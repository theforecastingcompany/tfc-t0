"""The backend benchmark must compare complete forecasts, not only checksums."""

from __future__ import annotations

import sys
from pathlib import Path

TOOLS = Path(__file__).parent.parent / "tools"
sys.path.insert(0, str(TOOLS))

from benchmark import _compare_with_torch_cpu, _remove_raw_outputs  # noqa: E402


def _result(backend: str, output: list[float]) -> dict:
    return {
        "backend": backend,
        "timings": [
            {
                "workload": {"batch": 1, "context": 4, "horizon": 2},
                "output_shape": [1, 2, 2],
                "output": output,
            }
        ],
    }


def test_elementwise_comparison_detects_compensating_checksum_error() -> None:
    # Both outputs sum to ten; a checksum-only guard would miss the corruption.
    results = [
        _result("torch-mps", [1.0, 1.0, 4.0, 4.0]),
        _result("torch-cpu", [1.0, 2.0, 3.0, 4.0]),
    ]
    [comparison] = _compare_with_torch_cpu(results, rtol=0.0, atol=1e-4)
    assert comparison.max_abs_error == 1.0
    assert not comparison.within_tolerance


def test_elementwise_comparison_accepts_small_backend_drift() -> None:
    results = [
        _result("torch-mps", [1.0, 2.00001, 3.0, 4.0]),
        _result("torch-cpu", [1.0, 2.0, 3.0, 4.0]),
    ]
    [comparison] = _compare_with_torch_cpu(results, rtol=0.0, atol=2e-5)
    assert comparison.within_tolerance

    _remove_raw_outputs(results)
    assert all("output" not in timing for result in results for timing in result["timings"])
