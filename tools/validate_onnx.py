"""Check grouped ONNX exports of a t0 checkpoint against the PyTorch runtime.

Every case runs a batch of synthetic series through ONNX Runtime (CPU) and through
``T0Forecaster.predict`` at the checkpoint's native quantile levels. The reference runs in
float64, so it also covers series whose level is far from zero relative to their spread,
where a float32 reference is itself inexact. Drift is the absolute error divided by the
reference series' range over the horizon and all quantiles. An artifact passes when every
output is finite, no quantiles cross, and the drift stays within the gates.

    uv run --group onnx python tools/validate_onnx.py theforecastingcompany/t0-gamma out/*.onnx
"""

import argparse
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

from t0 import TimeSeries
from t0.torch.model import T0Forecaster

CONTEXTS = (1, 32, 97, 512, 8192)
HORIZONS = (32, 64, 1024)
MAX_MEAN_DRIFT = 0.02
MAX_POINT_DRIFT = 0.10
# Rows of one batch, one group each; the two "joint" rows share a group with a known-future covariate.
SERIES = ("dense", "gappy", "level", "joint-0", "joint-1")


def make_case(context: int, horizon: int, seed: int) -> dict[str, np.ndarray]:
    """The ONNX feeds of one case."""
    rng = np.random.default_rng(seed)
    steps = np.arange(context + horizon, dtype=np.float64)
    seasonal = np.sin(2 * np.pi * steps / 24)
    dense = 20 + 0.01 * steps + 5 * seasonal + rng.normal(0, 1, steps.size)
    gappy = 3 * np.cos(2 * np.pi * steps / 7) + rng.normal(0, 0.5, steps.size)
    gappy[rng.random(steps.size) < 0.2] = np.nan
    gappy[: context // 4] = np.nan
    level = 5e4 + 0.3 * seasonal + rng.normal(0, 0.05, steps.size)
    calendar = np.sin(2 * np.pi * steps / 168)
    joint = np.stack([100 + 10 * calendar + rng.normal(0, 1, steps.size), 50 - 4 * calendar])
    targets = np.concatenate([np.stack([dense, gappy, level]), joint])[:, :context]
    return {
        "target_context": targets.astype(np.float32),
        "target_group_ids": np.array([0, 1, 2, 3, 3], dtype=np.int32),
        "future_covariate_context": calendar[None, :context].astype(np.float32),
        "future_covariate_future": calendar[None, context:].astype(np.float32),
        "future_covariate_group_ids": np.array([3], dtype=np.int32),
    }


def reference(model: T0Forecaster, feeds: dict[str, np.ndarray]) -> np.ndarray:
    """``predict`` on the same rows, at the native quantile levels, in float64."""
    targets = torch.from_numpy(feeds["target_context"]).double()
    covariates = torch.from_numpy(
        np.concatenate([feeds["future_covariate_context"], feeds["future_covariate_future"]], axis=1)
    ).double()
    horizon = covariates.shape[1] - targets.shape[1]
    values = torch.cat([torch.cat([targets, torch.full((targets.shape[0], horizon), torch.nan)], dim=1), covariates])
    groups = np.concatenate([feeds["target_group_ids"], feeds["future_covariate_group_ids"]])
    target_rows = len(feeds["target_group_ids"])
    variate_type = torch.tensor([0] * target_rows + [2] * (len(groups) - target_rows))[:, None].expand_as(values)
    mask = torch.isnan(values).to(torch.int8) * 2
    mask[:target_rows, targets.shape[1] :] = 4
    series = TimeSeries(
        variates=torch.nan_to_num(values),
        mask=mask,
        group_ids=torch.from_numpy(groups).long()[:, None].expand_as(values).contiguous(),
        variate_type=variate_type.contiguous(),
    )
    forecast = model.predict(series, horizon=horizon, quantile_levels=model.config.quantile_levels)
    return forecast.quantiles.numpy()


def compare(candidate: np.ndarray, expected: np.ndarray) -> list[dict[str, float]]:
    """Per-row drift over the reference spread, and quantile crossings."""
    rows = []
    for got, want in zip(candidate, expected, strict=True):
        spread = max(float(want.max() - want.min()), 1e-6)
        drift = np.abs(got.astype(np.float64) - want) / spread
        crossings = int((np.diff(got, axis=-1) < -1e-4 * spread).sum())
        rows.append(
            {
                "mean_drift_over_spread": float(drift.mean()),
                "max_drift_over_spread": float(drift.max()),
                "crossings": crossings,
                "finite": bool(np.isfinite(got).all()),
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("checkpoint", help="Hugging Face repository id, or a local checkpoint directory")
    parser.add_argument("artifacts", type=Path, nargs="+", help="grouped .onnx exports of that checkpoint")
    args = parser.parse_args()

    model = T0Forecaster.from_pretrained(args.checkpoint).eval().double()
    sessions = {path: ort.InferenceSession(str(path), providers=["CPUExecutionProvider"]) for path in args.artifacts}
    reports = {path: {"cases": [], "failures": []} for path in args.artifacts}
    for seed, (context, horizon) in enumerate((c, h) for c in CONTEXTS for h in HORIZONS):
        feeds = make_case(context, horizon, seed)
        expected = reference(model, feeds)
        for path, session in sessions.items():
            candidate = session.run(["quantiles"], feeds)[0]
            for series, row in zip(SERIES, compare(candidate, expected), strict=True):
                case = {"context": context, "horizon": horizon, "series": series, **row}
                reports[path]["cases"].append(case)
                if (
                    not row["finite"]
                    or row["crossings"]
                    or row["mean_drift_over_spread"] > MAX_MEAN_DRIFT
                    or row["max_drift_over_spread"] > MAX_POINT_DRIFT
                ):
                    reports[path]["failures"].append(case)
        print(f"context={context} horizon={horizon} done", flush=True)

    for path, report in reports.items():
        cases = report["cases"]
        summary = {
            "artifact": path.name,
            "runtime": ort.__version__,
            "reference": "tfc-t0 T0Forecaster.predict, float64, native quantile levels",
            "quantile_levels": list(model.config.quantile_levels),
            "max_mean_drift": MAX_MEAN_DRIFT,
            "max_point_drift": MAX_POINT_DRIFT,
            "passed": not report["failures"],
            "worst_mean_drift": max(case["mean_drift_over_spread"] for case in cases),
            "worst_point_drift": max(case["max_drift_over_spread"] for case in cases),
            **report,
        }
        path.with_suffix(".validation.json").write_text(json.dumps(summary, indent=1) + "\n")
        print(
            f"{path.name}: passed={summary['passed']} cases={len(cases)} "
            f"worst mean drift={summary['worst_mean_drift']:.4%} worst point drift={summary['worst_point_drift']:.4%}"
        )


if __name__ == "__main__":
    main()
