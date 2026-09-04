"""Compare t0-alpha inference across MLX, PyTorch MPS, and PyTorch CPU."""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import json
import statistics
import subprocess
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np

BACKENDS = ("mlx-eager", "mlx-compiled", "torch-mps", "torch-cpu")
RESULT_MARKER = "T0_BENCHMARK_RESULT="
DEFAULT_WORKLOADS = ((1, 96, 32), (1, 512, 64), (8, 96, 32))


@dataclass(frozen=True)
class Workload:
    """One public-API forecast shape."""

    batch: int
    context: int
    horizon: int

    @property
    def label(self) -> str:
        return f"batch {self.batch}, context {self.context}, horizon {self.horizon}"


@dataclass(frozen=True)
class Timing:
    """Synchronized steady-state latency for one backend and workload."""

    backend: str
    workload: Workload
    minimum_ms: float
    median_ms: float
    maximum_ms: float
    checksum: float
    output_shape: tuple[int, ...]
    output: list[float]
    compile_and_first_run_ms: float | None = None


@dataclass(frozen=True)
class NumericalComparison:
    """Elementwise agreement with the PyTorch CPU reference output."""

    backend: str
    workload: Workload
    max_abs_error: float
    max_rel_error: float
    within_tolerance: bool


def _parse_workload(value: str) -> Workload:
    try:
        batch, context, horizon = (int(part) for part in value.split(","))
    except (TypeError, ValueError) as error:
        raise argparse.ArgumentTypeError("workload must be BATCH,CONTEXT,HORIZON") from error
    if batch < 1 or context < 1 or horizon < 1:
        raise argparse.ArgumentTypeError("workload dimensions must be positive")
    return Workload(batch, context, horizon)


def _measure(
    predict: Callable[[], Any],
    synchronize: Callable[[Any], None],
    to_numpy: Callable[[Any], np.ndarray],
    iterations: int,
) -> tuple[float, float, float, np.ndarray]:
    durations: list[float] = []
    forecast: Any = None
    for _ in range(iterations):
        synchronize(None)
        started = time.perf_counter()
        forecast = predict()
        synchronize(forecast)
        durations.append((time.perf_counter() - started) * 1000.0)
    values = to_numpy(forecast)
    return min(durations), statistics.median(durations), max(durations), values


def _benchmark_mlx(
    backend: str,
    checkpoint: Path,
    workloads: Sequence[Workload],
    warmup: int,
    iterations: int,
    seed: int,
) -> dict[str, Any]:
    import mlx.core as mx

    from t0_mlx import T0Forecaster

    model = T0Forecaster.from_pretrained(checkpoint)
    compiled = backend == "mlx-compiled"
    if compiled:
        model.compile()

    timings = []
    for workload in workloads:
        context = np.random.default_rng(seed).normal(size=(workload.batch, workload.context)).astype(np.float32)
        predict = partial(model.predict, context, horizon=workload.horizon)

        compile_and_first_run_ms = None
        if compiled:
            started = time.perf_counter()
            first = predict()
            mx.eval(first.quantiles)
            compile_and_first_run_ms = (time.perf_counter() - started) * 1000.0

        for _ in range(warmup):
            warm = predict()
            mx.eval(warm.quantiles)

        minimum, median, maximum, output = _measure(
            predict,
            lambda forecast: mx.synchronize() if forecast is None else mx.eval(forecast.quantiles),
            lambda forecast: np.asarray(forecast.quantiles),
            iterations,
        )
        timings.append(
            Timing(
                backend=backend,
                workload=workload,
                minimum_ms=minimum,
                median_ms=median,
                maximum_ms=maximum,
                checksum=float(output.sum()),
                output_shape=output.shape,
                output=output.reshape(-1).tolist(),
                compile_and_first_run_ms=compile_and_first_run_ms,
            )
        )

    return {
        "backend": backend,
        "framework": f"MLX {importlib.metadata.version('mlx')}",
        "device": str(mx.default_device()),
        "timings": [_timing_to_dict(timing) for timing in timings],
    }


def _benchmark_torch(
    backend: str,
    checkpoint: Path,
    workloads: Sequence[Workload],
    warmup: int,
    iterations: int,
    seed: int,
    torch_threads: int | None,
) -> dict[str, Any]:
    try:
        t0 = importlib.import_module("t0")
        import torch  # ty: ignore[unresolved-import]
    except ImportError as error:
        raise RuntimeError(
            "PyTorch benchmarks require the parity extra (`uv run --extra parity ...`) "
            "or the local tfc-t0 checkout (`uv run --with-editable .. ...`)."
        ) from error

    device = "mps" if backend == "torch-mps" else "cpu"
    if device == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("PyTorch MPS is not available on this machine")
    if torch_threads is not None:
        torch.set_num_threads(torch_threads)

    model = t0.T0Forecaster.from_pretrained(checkpoint).eval().to(device)
    synchronize = torch.mps.synchronize if device == "mps" else lambda: None
    timings = []
    for workload in workloads:
        context = np.random.default_rng(seed).normal(size=(workload.batch, workload.context)).astype(np.float32)
        predict = partial(model.predict, context, horizon=workload.horizon)
        for _ in range(warmup):
            predict()
        synchronize()

        minimum, median, maximum, output = _measure(
            predict,
            lambda _forecast: synchronize(),
            lambda forecast: forecast.quantiles.detach().cpu().numpy(),
            iterations,
        )
        timings.append(
            Timing(
                backend=backend,
                workload=workload,
                minimum_ms=minimum,
                median_ms=median,
                maximum_ms=maximum,
                checksum=float(output.sum()),
                output_shape=output.shape,
                output=output.reshape(-1).tolist(),
            )
        )

    return {
        "backend": backend,
        "framework": f"PyTorch {torch.__version__}",
        "device": device,
        "torch_threads": torch.get_num_threads(),
        "timings": [_timing_to_dict(timing) for timing in timings],
    }


def _timing_to_dict(timing: Timing) -> dict[str, Any]:
    value = asdict(timing)
    value["workload"] = asdict(timing.workload)
    return value


def _workload_key(timing: dict[str, Any]) -> tuple[int, int, int]:
    workload = timing["workload"]
    return workload["batch"], workload["context"], workload["horizon"]


def _compare_with_torch_cpu(
    results: Sequence[dict[str, Any]],
    *,
    rtol: float,
    atol: float,
) -> list[NumericalComparison]:
    """Compare every available backend elementwise with PyTorch CPU."""
    by_backend = {result["backend"]: result for result in results}
    reference = by_backend.get("torch-cpu")
    if reference is None:
        return []
    reference_timings = {_workload_key(timing): timing for timing in reference["timings"]}
    comparisons: list[NumericalComparison] = []
    for result in results:
        if result["backend"] == "torch-cpu":
            continue
        for timing in result["timings"]:
            key = _workload_key(timing)
            reference_timing = reference_timings[key]
            expected = np.asarray(reference_timing["output"], dtype=np.float32).reshape(
                reference_timing["output_shape"]
            )
            actual = np.asarray(timing["output"], dtype=np.float32).reshape(timing["output_shape"])
            if actual.shape != expected.shape:
                raise RuntimeError(
                    f"{result['backend']} returned {actual.shape} for {key}; PyTorch CPU returned {expected.shape}"
                )
            absolute = np.abs(actual - expected)
            relative = absolute / np.maximum(np.abs(expected), np.float32(1e-6))
            comparisons.append(
                NumericalComparison(
                    backend=result["backend"],
                    workload=Workload(*key),
                    max_abs_error=float(absolute.max(initial=0.0)),
                    max_rel_error=float(relative.max(initial=0.0)),
                    within_tolerance=bool(np.all(absolute <= atol + rtol * np.abs(expected))),
                )
            )
    return comparisons


def _remove_raw_outputs(results: Sequence[dict[str, Any]]) -> None:
    for result in results:
        for timing in result["timings"]:
            timing.pop("output", None)
            timing.pop("output_shape", None)


def _run_worker(args: argparse.Namespace) -> None:
    if args.backend.startswith("mlx-"):
        result = _benchmark_mlx(
            args.backend,
            args.checkpoint,
            args.workloads,
            args.warmup,
            args.iterations,
            args.seed,
        )
    else:
        result = _benchmark_torch(
            args.backend,
            args.checkpoint,
            args.workloads,
            args.warmup,
            args.iterations,
            args.seed,
            args.torch_threads,
        )
    print(f"{RESULT_MARKER}{json.dumps(result, separators=(',', ':'))}")


def _worker_command(args: argparse.Namespace, backend: str) -> list[str]:
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        str(args.checkpoint),
        "--backend",
        backend,
        "--warmup",
        str(args.warmup),
        "--iterations",
        str(args.iterations),
        "--seed",
        str(args.seed),
        "--worker",
    ]
    for workload in args.workloads:
        command.extend(["--workload", f"{workload.batch},{workload.context},{workload.horizon}"])
    if args.torch_threads is not None:
        command.extend(["--torch-threads", str(args.torch_threads)])
    return command


def _run_isolated(args: argparse.Namespace, backend: str) -> dict[str, Any]:
    completed = subprocess.run(_worker_command(args, backend), check=False, capture_output=True, text=True)
    if completed.returncode != 0:
        sys.stderr.write(completed.stdout)
        sys.stderr.write(completed.stderr)
        raise SystemExit(completed.returncode)
    for line in reversed(completed.stdout.splitlines()):
        if line.startswith(RESULT_MARKER):
            return json.loads(line.removeprefix(RESULT_MARKER))
    raise RuntimeError(f"benchmark worker {backend!r} returned no result")


def _render_markdown(
    results: Sequence[dict[str, Any]],
    comparisons: Sequence[NumericalComparison],
    *,
    warmup: int,
    iterations: int,
    seed: int,
    rtol: float,
    atol: float,
) -> str:
    frameworks = []
    for result in results:
        description = f"{result['framework']} on {result['device']}"
        if "torch_threads" in result:
            description += f" ({result['torch_threads']} CPU threads)"
        if description not in frameworks:
            frameworks.append(description)
    rows = [
        f"Python {sys.version.split()[0]}; {warmup} warm-ups; {iterations} timed calls; seed {seed}.",
        "Frameworks: " + "; ".join(frameworks) + ".",
        "",
        "| Backend | Shape | Minimum | Median | Maximum | Compile + first run | Checksum |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for result in results:
        for timing in result["timings"]:
            workload = timing["workload"]
            shape = f"batch {workload['batch']}, context {workload['context']}, horizon {workload['horizon']}"
            first = timing["compile_and_first_run_ms"]
            first_text = "—" if first is None else f"{first:.2f} ms"
            rows.append(
                f"| {timing['backend']} | {shape} | {timing['minimum_ms']:.2f} ms "
                f"| {timing['median_ms']:.2f} ms | {timing['maximum_ms']:.2f} ms "
                f"| {first_text} | {timing['checksum']:.6f} |"
            )
    if comparisons:
        rows.extend(
            [
                "",
                f"Elementwise output comparison with PyTorch CPU (`rtol={rtol:g}`, `atol={atol:g}`):",
                "",
                "| Backend | Shape | Maximum absolute error | Maximum relative error | Within tolerance |",
                "| --- | --- | ---: | ---: | ---: |",
            ]
        )
        for comparison in comparisons:
            rows.append(
                f"| {comparison.backend} | {comparison.workload.label} "
                f"| {comparison.max_abs_error:.3e} | {comparison.max_rel_error:.3e} "
                f"| {'yes' if comparison.within_tolerance else 'NO'} |"
            )
    return "\n".join(rows)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path, help="directory containing config.json and model.safetensors")
    parser.add_argument("--backend", choices=("all", *BACKENDS), default="all")
    parser.add_argument(
        "--workload",
        dest="workloads",
        action="append",
        type=_parse_workload,
        help="repeatable BATCH,CONTEXT,HORIZON tuple; defaults to three representative shapes",
    )
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=60)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--torch-threads", type=int)
    parser.add_argument("--parity-rtol", type=float, default=2e-5)
    parser.add_argument("--parity-atol", type=float, default=3e-4)
    parser.add_argument("--format", choices=("markdown", "json"), default="markdown")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    return parser


def main() -> None:
    args = _parser().parse_args()
    args.workloads = args.workloads or [Workload(*values) for values in DEFAULT_WORKLOADS]
    if args.warmup < 0:
        raise SystemExit("--warmup must be non-negative")
    if args.iterations < 1:
        raise SystemExit("--iterations must be positive")
    if args.torch_threads is not None and args.torch_threads < 1:
        raise SystemExit("--torch-threads must be positive")
    if args.parity_rtol < 0 or args.parity_atol < 0:
        raise SystemExit("--parity-rtol and --parity-atol must be non-negative")
    if not (args.checkpoint / "config.json").is_file() or not (args.checkpoint / "model.safetensors").is_file():
        raise SystemExit(f"{args.checkpoint} must contain config.json and model.safetensors")

    if args.worker:
        if args.backend == "all":
            raise SystemExit("worker mode requires one concrete backend")
        _run_worker(args)
        return

    backends = BACKENDS if args.backend == "all" else (args.backend,)
    results = [_run_isolated(args, backend) for backend in backends]
    comparisons = _compare_with_torch_cpu(results, rtol=args.parity_rtol, atol=args.parity_atol)
    _remove_raw_outputs(results)
    payload = {
        "python": sys.version.split()[0],
        "warmup": args.warmup,
        "iterations": args.iterations,
        "seed": args.seed,
        "parity_rtol": args.parity_rtol,
        "parity_atol": args.parity_atol,
        "backends": results,
        "comparisons": [
            {**asdict(comparison), "workload": asdict(comparison.workload)} for comparison in comparisons
        ],
    }
    if args.format == "json":
        print(json.dumps(payload, indent=2))
    else:
        print(
            _render_markdown(
                results,
                comparisons,
                warmup=args.warmup,
                iterations=args.iterations,
                seed=args.seed,
                rtol=args.parity_rtol,
                atol=args.parity_atol,
            )
        )
    failed = [comparison for comparison in comparisons if not comparison.within_tolerance]
    if failed:
        backends = ", ".join(sorted({comparison.backend for comparison in failed}))
        raise SystemExit(f"elementwise comparison with PyTorch CPU failed for: {backends}")


if __name__ == "__main__":
    main()
