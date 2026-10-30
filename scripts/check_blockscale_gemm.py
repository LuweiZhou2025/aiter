# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Accuracy-only sweep of the gfx950 FlyDSL 8-wave FP8 blockscale kernel.

Run from the repository root:
  python scripts/check_blockscale_gemm.py --collect-only
  python scripts/check_blockscale_gemm.py --device 1
  python scripts/check_blockscale_gemm.py --device 1 --stress --repeats 10
  python scripts/check_blockscale_gemm.py --shapes my_shapes.csv --results out.csv

The default shape file is created if absent; --collect-only refreshes it from
all plain/B-preshuffle tuned CSVs under aiter/configs, including model_configs.
MX-scale, group32, A-preshuffle and untuned tables are deliberately excluded.
Rows from every source architecture are included and deduplicated by
M,N,K,preshuffle. Edit/copy this file to select a smaller sweep.

By default, each shape invocation draws a fresh seed from OS randomness; the
actual seed is saved in the result CSV. --seed BASE opts into deterministic
shape-derived seeds (identical MNK data across weight layouts). --repeats reuses
that invocation's inputs to check repeated execution, rather than regenerating
them. To replay a CSV seed S, use --seed (S - M*1000003 - N*1009 - K).

This is not a dispatch/performance benchmark: the named FlyDSL candidate is
called directly, with no CK/ASM fallback. Reference operands are the SAME FP8
values dequantized with FP32 scales, then rounded to BF16 before F.linear.
The PyHIP calc_diff formula is 1 - 2*sum(x*y)/sum(x*x+y*y), accumulated in
FP64 chunks; smaller is better, and <= --threshold passes. Both-zero outputs
have diff=0; any nonfinite output/reference fails. BF16 operand rounding is
therefore included in the error (original pre-quantization error is not).

--stress allocates two independent buffers (--stress-gib is their TOTAL GiB)
and repeatedly copies between them in a graph on another stream. Defaults are
8 GiB total buffers and 64 copies: 512 GiB logical read+write traffic per check.
The copy event interval must contain the entire GEMM event interval on EVERY
repetition; STRESS_INCOMPLETE_COVERAGE is not a pass. CSV timing fields report
the minimum overlap/copy duration/start lead/end tail and maximum GEMM duration
across repetitions. Both lead and tail must be nonnegative for full coverage.
This verifies event-interval coverage, not hardware-level bandwidth or continuous
HBM saturation. Increase --stress-passes if the traffic ends before the GEMM.
Only run on a GPU you have reserved; the script does not lock shared GPUs.

A persistent spawned worker amortizes imports/JIT across shapes. On exceptions,
process crashes or per-case timeouts it is replaced before the next shape.
GPU/driver-wide faults may still require external recovery. Results (including
tracebacks) are flushed after each case; failures do not stop the sweep. Exit
status is 1 if any row is not PASS, 2 for setup/input errors. Ctrl-C stops it.
No TFLOPS are collected. Use --start/--limit to resume or smoke-test a subset.
"""

import argparse
import csv
import math
import multiprocessing as mp
import secrets
import sys
import time
import traceback
from collections import Counter
from multiprocessing.connection import Connection
from pathlib import Path
from typing import NamedTuple

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SHAPES = Path(__file__).with_name("blockscale_gemm_shapes.csv")
SHAPE_FIELDS = ("M", "N", "K", "preshuffle")
RESULT_FIELDS = SHAPE_FIELDS + (
    "status",
    "calc_diff",
    "max_abs",
    "rmse",
    "finite",
    "repeats_completed",
    "seed",
    "threshold",
    "stress_gib",
    "stress_passes",
    "stress_overlap_ms",
    "stress_start_lead_ms",
    "stress_end_tail_ms",
    "stress_copy_ms_min",
    "stress_gemm_ms_max",
    "stress_full_coverage",
    "error",
)


class Shape(NamedTuple):
    m: int
    n: int
    k: int
    preshuffle: bool


def read_shapes(path: Path, preshuffle: bool | None = None) -> list[Shape]:
    shapes = []
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        required = SHAPE_FIELDS if preshuffle is None else SHAPE_FIELDS[:3]
        if not set(required).issubset(reader.fieldnames or []):
            raise ValueError(f"{path}: missing required columns {required}")
        for line, row in enumerate(reader, 2):
            try:
                m, n, k = (int(row[key]) for key in SHAPE_FIELDS[:3])
                if min(m, n, k) <= 0:
                    raise ValueError("M, N and K must be positive")
                ps = preshuffle
                if ps is None:
                    flag = row["preshuffle"].strip().lower()
                    if flag not in ("0", "1", "false", "true"):
                        raise ValueError("preshuffle must be 0/1/false/true")
                    ps = flag in ("1", "true")
                shapes.append(Shape(m, n, k, ps))
            except (ValueError, TypeError, AttributeError) as exc:
                raise ValueError(f"{path}:{line}: {exc}") from exc
    if not shapes:
        raise ValueError(f"{path}: no shapes")
    return shapes


def collect_shapes(config_dir: Path, output: Path) -> None:
    shapes: set[Shape] = set()
    sources = []
    for ps, pattern in (
        (False, "*a8w8_blockscale_tuned_gemm*.csv"),
        (True, "*a8w8_blockscale_bpreshuffle_tuned_gemm*.csv"),
    ):
        paths = sorted(config_dir.rglob(pattern))
        if not paths:
            raise ValueError(f"{config_dir}: no tables matching {pattern}")
        for path in paths:
            shapes.update(read_shapes(path, ps))
            sources.append(path)
    if output.resolve() in {path.resolve() for path in sources}:
        raise ValueError("Shape output must not overwrite a tuned source table")
    with output.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(SHAPE_FIELDS)
        for m, n, k, ps in sorted(shapes, key=lambda s: (s.preshuffle, s.n, s.k, s.m)):
            writer.writerow((m, n, k, int(ps)))
    counts = Counter(s.preshuffle for s in shapes)
    print(
        f"Collected {len(shapes)} shapes from {len(sources)} tables: "
        f"plain={counts[False]}, preshuffle={counts[True]} -> {output}",
        flush=True,
    )


def comparison_stats(actual, reference) -> tuple[float, float, float, float, bool]:
    """FP64 sufficient statistics, matching PyHIP without a dependency on it."""
    import torch

    x, y = actual.double(), reference.double()
    finite = bool((torch.isfinite(x).all() & torch.isfinite(y).all()).item())
    delta = x - y
    return (
        (x * y).sum().item(),
        (x.square() + y.square()).sum().item(),
        delta.square().sum().item(),
        delta.abs().max().item(),
        finite,
    )


def finish_comparison(
    dot: float, denominator: float, squared_error: float, count: int, finite: bool
) -> tuple[float, float]:
    if not finite:
        return float("inf"), float("inf")
    diff = 1.0 - 2.0 * dot / denominator if denominator else 0.0
    return diff, math.sqrt(squared_error / count)


def stress_metrics(
    traffic_start: float, traffic_end: float, gemm_start: float, gemm_end: float
) -> dict[str, float | bool]:
    if not all(
        math.isfinite(t) for t in (traffic_start, traffic_end, gemm_start, gemm_end)
    ) or not (traffic_end > traffic_start and gemm_end > gemm_start):
        raise ValueError("Invalid or nonpositive GPU event interval")
    return {
        "stress_overlap_ms": max(
            0.0, min(traffic_end, gemm_end) - max(traffic_start, gemm_start)
        ),
        "stress_start_lead_ms": gemm_start - traffic_start,
        "stress_end_tail_ms": traffic_end - gemm_end,
        "stress_copy_ms_min": traffic_end - traffic_start,
        "stress_gemm_ms_max": gemm_end - gemm_start,
        "stress_full_coverage": traffic_start <= gemm_start and gemm_end <= traffic_end,
    }


class HbmTraffic:
    def __init__(self, gib: float, passes: int, device):
        import torch

        elements = int(gib * 2**30 / 2 / 4)
        self.src = torch.ones(elements, dtype=torch.float32, device=device)
        self.dst = torch.empty_like(self.src)
        self.stream = torch.cuda.Stream(device=device)
        self.stream.wait_stream(torch.cuda.current_stream(device))
        with torch.cuda.stream(self.stream):
            self.dst.copy_(self.src)
        self.stream.synchronize()
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph, stream=self.stream):
            for _ in range(passes):
                self.dst.copy_(self.src)

    def run(self, launch) -> dict[str, float | bool]:
        import torch

        origin, traffic_start, traffic_end, gemm_start, gemm_end = (
            torch.cuda.Event(enable_timing=True) for _ in range(5)
        )
        origin.record()
        self.stream.wait_event(origin)
        with torch.cuda.stream(self.stream):
            traffic_start.record()
            self.graph.replay()
            traffic_end.record()
        # Order only the start marker, not completion of the copy graph.
        torch.cuda.current_stream().wait_event(traffic_start)
        gemm_start.record()
        launch()
        gemm_end.record()
        gemm_end.synchronize()
        traffic_end.synchronize()
        return stress_metrics(
            origin.elapsed_time(traffic_start),
            origin.elapsed_time(traffic_end),
            origin.elapsed_time(gemm_start),
            origin.elapsed_time(gemm_end),
        )


def run_case(shape: Shape, args, traffic: HbmTraffic | None) -> dict:
    import torch
    import torch.nn.functional as F

    from aiter.ops.flydsl.gemm_a8w8_blockscale import run_gemm_a8w8_blockscale
    from aiter.ops.flydsl.gemm_tune.flydsl_gemm_a8w8_blockscale_common import (
        kernel_fits_shape,
        kernelInstance,
    )
    from aiter.ops.shuffle import shuffle_weight

    m, n, k, ps = shape
    ki = kernelInstance(ps)
    if not kernel_fits_shape(ki, m, n, k, "gfx950"):
        return {"status": "UNSUPPORTED", "error": "Outside 8-wave shape contract"}
    device = torch.device("cuda", args.device)
    seed = (
        secrets.randbits(63)
        if args.seed is None
        else (args.seed + m * 1000003 + n * 1009 + k) % (2**63)
    )
    generator = torch.Generator(device=device).manual_seed(seed)
    a = torch.randn((m, k), device=device, generator=generator).to(torch.float8_e4m3fn)
    b = torch.randn((n, k), device=device, generator=generator).to(torch.float8_e4m3fn)
    sa = torch.rand((m, k // 128), device=device, generator=generator) + 0.5
    sb = (
        torch.rand(((n + 127) // 128, k // 128), device=device, generator=generator)
        + 0.5
    )
    b_ref = torch.empty((n, k), device=device, dtype=torch.bfloat16)
    for row in range(0, n, 128):
        end = min(row + 128, n)
        b_ref[row:end] = (
            (
                b[row:end].float().reshape(end - row, k // 128, 128)
                * sb[row // 128, :, None]
            )
            .reshape(end - row, k)
            .to(torch.bfloat16)
        )
    weight = shuffle_weight(b, layout=(16, 16)) if ps else b
    scale_a = sa.T.contiguous() if ps else sa
    output = torch.empty((m, n), device=device, dtype=torch.bfloat16)

    def launch():
        run_gemm_a8w8_blockscale(
            a, weight, scale_a, sb, output, ki.name, preshuffle_b=ps
        )

    launch()
    torch.cuda.synchronize(device)
    result = {
        "status": "PASS",
        "calc_diff": 0.0,
        "max_abs": 0.0,
        "rmse": 0.0,
        "finite": True,
        "repeats_completed": 0,
        "seed": seed,
        "stress_overlap_ms": "",
        "stress_start_lead_ms": "",
        "stress_end_tail_ms": "",
        "stress_copy_ms_min": "",
        "stress_gemm_ms_max": "",
        "stress_full_coverage": "",
    }
    for iteration in range(args.repeats):
        output.fill_(float("nan"))
        if traffic is not None:
            metrics = traffic.run(launch)
            if iteration == 0:
                result.update(metrics)
            else:
                for field in (
                    "stress_overlap_ms",
                    "stress_start_lead_ms",
                    "stress_end_tail_ms",
                    "stress_copy_ms_min",
                ):
                    result[field] = min(result[field], metrics[field])
                result["stress_gemm_ms_max"] = max(
                    result["stress_gemm_ms_max"], metrics["stress_gemm_ms_max"]
                )
                result["stress_full_coverage"] &= metrics["stress_full_coverage"]
            if not metrics["stress_full_coverage"] and result["status"] == "PASS":
                result["status"] = "STRESS_INCOMPLETE_COVERAGE"
        else:
            launch()
            torch.cuda.synchronize(device)
        dot = denominator = squared_error = max_abs = 0.0
        finite = True
        for row in range(0, m, args.reference_rows):
            end = min(row + args.reference_rows, m)
            a_ref = (
                (
                    a[row:end].float().reshape(end - row, k // 128, 128)
                    * sa[row:end, :, None]
                )
                .reshape(end - row, k)
                .to(torch.bfloat16)
            )
            reference = F.linear(a_ref, b_ref)
            d, denom, sq, maximum, ok = comparison_stats(output[row:end], reference)
            dot += d
            denominator += denom
            squared_error += sq
            max_abs = max(max_abs, maximum) if ok else float("inf")
            finite &= ok
        diff, rmse = finish_comparison(dot, denominator, squared_error, m * n, finite)
        result["calc_diff"] = max(result["calc_diff"], diff)
        result["max_abs"] = max(result["max_abs"], max_abs)
        result["rmse"] = max(result["rmse"], rmse)
        result["finite"] &= finite
        result["repeats_completed"] = iteration + 1
        if not finite or diff > args.threshold:
            result["status"] = "FAIL"
    return result


def worker(connection: Connection, args) -> None:
    sys.path.insert(0, str(ROOT))
    try:
        import torch

        from aiter.jit.utils.chip_info import get_gfx

        torch.cuda.set_device(args.device)
        gfx = get_gfx()
        if gfx != "gfx950":
            raise RuntimeError(f"This 8-wave kernel requires gfx950, got {gfx}")
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
        traffic = (
            HbmTraffic(
                args.stress_gib, args.stress_passes, torch.device("cuda", args.device)
            )
            if args.stress
            else None
        )
        connection.send({"status": "READY", "device": torch.cuda.get_device_name()})
    except Exception:  # noqa: BLE001 - transport setup traceback to the parent
        connection.send({"status": "ERROR", "error": traceback.format_exc()})
        return
    with torch.inference_mode():
        while True:
            shape = connection.recv()
            if shape is None:
                return
            try:
                connection.send(run_case(shape, args, traffic))
            except Exception:  # noqa: BLE001 - isolate arbitrary JIT/HIP failures
                # A failed HIP call may poison the context. Report, then replace it.
                connection.send({"status": "ERROR", "error": traceback.format_exc()})
                return


def receive(connection: Connection, timeout: float) -> dict:
    if not connection.poll(timeout):
        return {"status": "TIMEOUT", "error": f"No response within {timeout:g}s"}
    try:
        return connection.recv()
    except (EOFError, ConnectionError) as exc:
        return {"status": "CRASH", "error": f"GPU worker exited: {exc!r}"}


def stop_worker(process, connection: Connection) -> None:
    if process.is_alive():
        process.terminate()
    process.join(timeout=5)
    if process.is_alive():
        process.kill()
        process.join(timeout=5)
    connection.close()
    process.close()


def sweep(shapes: list[Shape], args) -> int:
    context = mp.get_context("spawn")
    process = connection = None
    counts: Counter[str] = Counter()
    started = time.monotonic()
    with args.results.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=RESULT_FIELDS)
        writer.writeheader()
        f.flush()
        try:
            for index, shape in enumerate(shapes, 1):
                print(f"[{index}/{len(shapes)}] RUN {shape}", flush=True)
                if process is None:
                    connection, child = context.Pipe()
                    process = context.Process(target=worker, args=(child, args))
                    process.start()
                    child.close()
                    ready = receive(connection, args.timeout)
                    if ready["status"] != "READY":
                        raise RuntimeError(f"GPU worker setup failed: {ready}")
                    print(f"GPU {args.device}: {ready['device']}", flush=True)
                try:
                    connection.send(shape)
                    result = receive(connection, args.timeout)
                except (BrokenPipeError, ConnectionError) as exc:
                    result = {"status": "CRASH", "error": repr(exc)}
                row = dict(zip(SHAPE_FIELDS, (*shape[:3], int(shape.preshuffle))))
                row.update(
                    threshold=args.threshold,
                    stress_gib=args.stress_gib if args.stress else 0,
                    stress_passes=args.stress_passes if args.stress else 0,
                )
                row.update(result)
                writer.writerow(row)
                f.flush()
                counts[result["status"]] += 1
                eta = (time.monotonic() - started) / index * (len(shapes) - index)
                print(
                    f"[{index}/{len(shapes)}] {result['status']} "
                    f"diff={result.get('calc_diff', '-')} "
                    f"pass={counts['PASS']} issues={index - counts['PASS']} "
                    f"ETA={eta / 60:.1f}min",
                    flush=True,
                )
                if result.get("error"):
                    print(result["error"], file=sys.stderr, flush=True)
                if result["status"] in ("ERROR", "CRASH", "TIMEOUT"):
                    stop_worker(process, connection)
                    process = connection = None
        finally:
            if process is not None:
                stop_worker(process, connection)
    print(f"Summary: {dict(counts)}; results: {args.results}", flush=True)
    return int(counts["PASS"] != len(shapes))


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--shapes", type=Path, default=DEFAULT_SHAPES)
    parser.add_argument("--config-dir", type=Path, default=ROOT / "aiter/configs")
    parser.add_argument("--collect-only", action="store_true")
    parser.add_argument("--results", type=Path, default=Path("blockscale_results.csv"))
    parser.add_argument(
        "--device", type=int, default=0, help="Visible CUDA/HIP device index"
    )
    parser.add_argument("--threshold", type=float, default=1e-5)
    parser.add_argument(
        "--seed",
        type=int,
        help="Reproducible base seed; default: fresh random seed for each shape",
    )
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument(
        "--timeout", type=float, default=600, help="Seconds per case/startup"
    )
    parser.add_argument("--reference-rows", type=int, default=256)
    parser.add_argument("--start", type=int, default=0, help="Zero-based row offset")
    parser.add_argument(
        "--limit", type=int, help="Maximum rows; default: all remaining"
    )
    parser.add_argument("--stress", action="store_true")
    parser.add_argument(
        "--stress-gib", type=float, default=8.0, help="TOTAL buffer GiB (default: 8)"
    )
    parser.add_argument(
        "--stress-passes", type=int, default=64, help="Copies per check (default: 64)"
    )
    args = parser.parse_args()
    for name in ("threshold", "timeout", "stress_gib"):
        value = getattr(args, name)
        if not math.isfinite(value) or value <= 0:
            parser.error(f"--{name.replace('_', '-')} must be finite and positive")
    if min(args.repeats, args.reference_rows, args.stress_passes) < 1:
        parser.error("--repeats, --reference-rows and --stress-passes must be positive")
    if args.start < 0 or args.device < 0 or (args.limit is not None and args.limit < 1):
        parser.error("--start/--device must be nonnegative; --limit must be positive")
    if args.stress and args.stress_gib < 0.25:
        parser.error("--stress-gib must be >= 0.25 (use buffers larger than cache)")
    try:
        if args.collect_only or not args.shapes.exists():
            collect_shapes(args.config_dir, args.shapes)
        if args.collect_only:
            return 0
        if args.results.resolve() == args.shapes.resolve():
            raise ValueError("--results must not overwrite --shapes")
        shapes = read_shapes(args.shapes)
        shapes = shapes[args.start : args.start + args.limit if args.limit else None]
        if not shapes:
            raise ValueError("No shapes selected")
        print(f"Testing {len(shapes)} shapes; results -> {args.results}", flush=True)
        return sweep(shapes, args)
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr, flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
