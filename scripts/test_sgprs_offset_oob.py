"""Run directly on an idle gfx950 GPU:

HIP_VISIBLE_DEVICES=6 python oob_test.py --max-offset 12288
HIP_VISIBLE_DEVICES=6 python oob_test.py --offsets 0 4096 8192 0xfffff000

64 active lanes, source voffset=0, DWORDX4 per lane, no LDS readback.
--max-offset is exclusive, with a 4096-byte step (default: 1 MiB).
Each case runs in a fresh
process so a global-load fault does not stop the remaining tests. The logical
buffer ends at an unmapped virtual-address region, not another valid allocation.
"""

import argparse
import ctypes as ct
import os
from pathlib import Path
import resource
import subprocess
import sys

import torch
import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import const_expr, rocdl

BLOCK_SIZE = 64


@flyc.kernel(known_block_size=[BLOCK_SIZE, 1, 1])
def test_kernel(
    src: fx.Pointer,
    buffer_size: fx.Constexpr[int],
    is_buffer_load: fx.Constexpr[bool],
    soffset: fx.Int64,
):
    lds = (
        fx.SharedAllocator()
        .allocate(fx.Array[fx.Uint8, BLOCK_SIZE * 16, 16])
        .peek()
        .ptr
    )
    voffset = fx.thread_idx.x * 0
    if const_expr(is_buffer_load):
        rsrc = rocdl.get_buffer_rsrc(
            rocdl.make_buffer_ptr(src, num_records_bytes=buffer_size)
        )
        rocdl.raw_ptr_buffer_load_lds(
            rsrc,
            fx.to_llvm_ptr(lds),
            fx.Int32(16),
            voffset,
            fx.Int32(soffset),
            fx.Int32(0),
            fx.Int32(0),
        )
    else:
        rocdl.global_load_lds(
            fx.to_llvm_ptr(fx.add_offset(src, soffset + fx.Int64(voffset))),
            fx.to_llvm_ptr(fx.add_offset(lds, fx.thread_idx.x * 16)),
            16,
            0,
        )
    rocdl.s_waitcnt(vmcnt=0)


@flyc.jit
def launch_test(
    src: fx.Pointer,
    buffer_size: fx.Constexpr[int],
    is_buffer_load: fx.Constexpr[bool],
    soffset: fx.Int64,
):
    test_kernel(src, buffer_size, is_buffer_load, soffset).launch(
        grid=(1,), block=(BLOCK_SIZE,)
    )


class Location(ct.Structure):
    _fields_ = [("type", ct.c_int), ("id", ct.c_int)]


class AllocationProperties(ct.Structure):
    _fields_ = [
        ("type", ct.c_int),
        ("handle_types", ct.c_int),
        ("location", Location),
        ("win32_metadata", ct.c_void_p),
        ("alloc_flags", ct.c_uint32),
    ]


class AccessDescriptor(ct.Structure):
    _fields_ = [("location", Location), ("flags", ct.c_int)]


def run_single(buffer_size, mode, soffset):
    if not torch.version.hip or not torch.cuda.is_available():
        raise RuntimeError("A ROCm GPU is required")
    if not torch.cuda.get_device_properties(0).gcnArchName.startswith("gfx950"):
        raise RuntimeError("This DWORDX4 LDS DMA test requires gfx950")
    if mode == "buffer" and soffset >= 1 << 32:
        raise ValueError("Buffer soffset is 32 bit; refusing to truncate it")

    hip = ct.CDLL("/opt/rocm/lib/libamdhip64.so")
    hip.hipGetErrorString.argtypes = [ct.c_int]
    hip.hipGetErrorString.restype = ct.c_char_p

    def call(name, *args):
        # Every argument below has an explicit ctypes width, including VA sizes.
        error = getattr(hip, name)(*args)
        if error:
            print(
                f"{name}: HIP error {error}: {hip.hipGetErrorString(error).decode()}",
                file=sys.stderr,
                flush=True,
            )
            os._exit(1)  # Do not run HIP finalizers in a faulted context.

    size, u64 = ct.c_size_t, ct.c_ulonglong
    prop = AllocationProperties(type=1, location=Location(1, 0))
    granularity, reservation, handle = size(), ct.c_void_p(), ct.c_void_p()
    call("hipSetDevice", ct.c_int(0))
    call(
        "hipMemGetAllocationGranularity",
        ct.byref(granularity),
        ct.byref(prop),
        ct.c_int(0),
    )
    g = granularity.value
    if not g:
        raise RuntimeError("HIP reported zero VMM granularity")
    mapped_size = (buffer_size + g - 1) // g * g
    reserved_size = (
        mapped_size + (max(4096, soffset + BLOCK_SIZE * 16) + g - 1) // g * g
    )
    call(
        "hipMemAddressReserve",
        ct.byref(reservation),
        size(reserved_size),
        size(g),
        ct.c_void_p(),
        u64(0),
    )
    call("hipMemCreate", ct.byref(handle), size(mapped_size), ct.byref(prop), u64(0))
    call("hipMemMap", reservation, size(mapped_size), size(0), handle, u64(0))
    access = AccessDescriptor(location=Location(1, 0), flags=3)
    call("hipMemSetAccess", reservation, size(mapped_size), ct.byref(access), size(1))

    # Keep the last buffer_size bytes of the mapping; everything after is unmapped.
    base = reservation.value + mapped_size - buffer_size
    call("hipMemset", ct.c_void_p(base), ct.c_int(0x5A), size(buffer_size))
    call("hipDeviceSynchronize")
    pointer = flyc.from_c_void_p(fx.Uint8, base, assumed_align=16)
    print(
        f"LAUNCH {mode}: base={base:#x}, guard={base + buffer_size:#x}, "
        f"offset={soffset:#x}",
        flush=True,
    )
    flyc.compile(launch_test, pointer, buffer_size, mode == "buffer", soffset)
    call("hipDeviceSynchronize")
    call("hipMemUnmap", reservation, size(mapped_size))
    call("hipMemRelease", handle)
    call("hipMemAddressFree", reservation, size(reserved_size))
    print("NO_CRASH", flush=True)


def main():
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--buffer-size", type=int, default=BLOCK_SIZE * 16)
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument(
        "--max-offset",
        type=lambda value: int(value, 0),
        default=1024 * 1024,
        help="Exclusive scan limit in bytes; offsets advance by 4096",
    )
    selection.add_argument(
        "--offsets",
        nargs="+",
        type=lambda value: int(value, 0),
        help="Explicit byte offsets instead of scanning --max-offset",
    )
    parser.add_argument("--mode", choices=("buffer", "global", "both"), default="both")
    parser.add_argument("--single", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if not 512 <= args.buffer_size < 1 << 32 or args.buffer_size % 16:
        parser.error("--buffer-size must be a multiple of 16 in [512, 4 GiB)")
    if not 0 < args.max_offset <= 16 * 1024**3:
        parser.error("--max-offset must be in (0, 16 GiB]")
    if args.offsets is not None and any(
        offset < 0 or offset > 16 * 1024**3 or offset % 16 for offset in args.offsets
    ):
        parser.error("--offsets must be multiples of 16 in [0, 16 GiB]")
    if os.environ.get("COMPILE_ONLY") == "1":
        parser.error("Unset COMPILE_ONLY: this test must actually execute")
    if args.single:
        if args.mode == "both" or args.offsets is None or len(args.offsets) != 1:
            parser.error("--single requires one mode and one offset")
        if args.mode == "buffer" and args.offsets[0] >= 1 << 32:
            parser.error("Buffer soffset is 32 bit; values >=4 GiB cannot be encoded")
        run_single(args.buffer_size, args.mode, args.offsets[0])
        return

    modes = ("buffer", "global") if args.mode == "both" else (args.mode,)
    offsets = (
        args.offsets if args.offsets is not None else range(0, args.max_offset, 4096)
    )
    print(
        f"buffer_size={args.buffer_size}, lanes={BLOCK_SIZE}, voffset=lane*0, DWORDX4 -> LDS"
    )
    print("GPU faults below are observed outcomes; LDS values are not checked.")
    print(f"{'Offset (bytes)':>15}  {'Mode':<8}  {'Result':<27}  Exit", flush=True)
    for offset in offsets:
        for mode in modes:
            if mode == "buffer" and offset >= 1 << 32:
                print(
                    f"{offset:>15}  {mode:<8}  {'UNREPRESENTABLE_SOFFSET':<27}  -",
                    flush=True,
                )
                continue
            command = [
                sys.executable,
                str(Path(__file__).resolve()),
                "--single",
                "--buffer-size",
                str(args.buffer_size),
                "--mode",
                mode,
                "--offsets",
                str(offset),
            ]
            try:
                result = subprocess.run(
                    command, capture_output=True, text=True, timeout=120
                )
            except subprocess.TimeoutExpired:
                print(f"{offset:>15}  {mode:<8}  TIMEOUT; stopping", flush=True)
                raise SystemExit(1)
            combined = result.stdout + result.stderr
            if result.returncode == 0 and "NO_CRASH" in result.stdout:
                outcome = "NO_CRASH"
            elif (
                result.returncode != 0
                and "LAUNCH " in result.stdout
                and (
                    "HIP error 700:" in combined
                    or "Memory access fault by GPU" in combined
                )
            ):
                outcome = "GPU_MEMORY_FAULT"
            else:
                print(combined, file=sys.stderr, flush=True)
                raise SystemExit(
                    f"Unexpected error: {mode}, offset={offset}, exit={result.returncode}"
                )
            print(
                f"{offset:>15}  {mode:<8}  {outcome:<27}  {result.returncode}",
                flush=True,
            )


if __name__ == "__main__":
    main()
