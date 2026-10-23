# Numerical parity

The MLX runtime (`t0.mlx`, installed with `pip install 'tfc-t0[mlx]'`) is
checked against the PyTorch runtime using the original FP32 `t0-alpha`
checkpoint. The entries below were recorded for `tfc-t0-mlx`, the standalone
MLX distribution that the `mlx` extra of `tfc-t0` supersedes.

To repeat the check, point the checkpoint-backed tests at a local directory
holding the original `config.json` and `model.safetensors`:

```bash
uv sync
MLX_ENABLE_TF32=0 T0_MLX_CHECKPOINT=/path/to/t0-alpha uv run pytest tests/mlx
```

## [0.1.0a1] - 2026-09-05

- Reference package: `tfc-t0` 0.3.1.
- Candidate package: `tfc-t0-mlx` 0.1.0a1; runtime unchanged from 0.1.0a0.
- Hugging Face revision: `7fbf2648f27133aa427f51d152cdaa35c0268f32`.
- Checkpoint SHA-256:
  `16c030d3fd70f06dc4238e9a8356e9b5a631d07f80f1bc76ba539991aed5897f`.
- Environment: Apple M5 Max, macOS 26.4, Python 3.12.13, MLX 0.32.2,
  PyTorch 2.13.0, NumPy 2.5.2.
- Result: all 40 tests passed, including checkpoint-backed parity, with
  the original tolerances unchanged.
- Precision: `MLX_ENABLE_TF32=0` was set before starting Python. MLX can use
  reduced-precision float32 matrix multiplication on supported hardware;
  the default M5 setting failed three of these strict parity tests. See
  [MLX numerical precision](https://ml-explore.github.io/mlx/build/html/usage/precision.html).

## [0.1.0a0] - 2026-09-04

- Reference package: `tfc-t0` 0.3.0.
- Candidate package: `tfc-t0-mlx` 0.1.0a0.
- Hugging Face revision:
  `7fbf2648f27133aa427f51d152cdaa35c0268f32`.
- Checkpoint SHA-256:
  `16c030d3fd70f06dc4238e9a8356e9b5a631d07f80f1bc76ba539991aed5897f`.
- Coverage: patch encoding, time and group attention, rotary embeddings,
  scaling, native and interpolated quantiles, missing observations, ragged
  padding, explicit groups, known-future covariates, compiled inference and
  autoregressive rollout.
- End-to-end tolerance: `rtol=2e-5`, `atol=3e-4` or tighter, depending on the
  path under test.
- PyTorch MPS versus CPU: maximum absolute error `2.623e-6` across the three
  published benchmark workloads; every element passed `rtol=2e-5`,
  `atol=3e-4`.

The checkpoint itself lives in the
[`theforecastingcompany/t0-alpha`](https://huggingface.co/theforecastingcompany/t0-alpha)
repository and is not redistributed here.

## ONNX exports

`tools/export_onnx.py` exports a checkpoint to the grouped ONNX interface of the
published `t0-*-onnx-*` artifacts. `tools/validate_onnx.py` checks each export
against `T0Forecaster.predict` in float64, at the native quantile levels. It runs
75 cases: contexts 1, 32, 97, 512 and 8192, horizons 32, 64 and 1024, and five
series per case. The series are a dense series, a series with gaps and leading
missing values, a series at a level of 50,000 that moves by less than 1, and two
series forecast jointly with a known-future covariate. Drift is the absolute
error divided by the reference series' range over the horizon and all quantiles.
An export passes when all outputs are finite, no quantiles cross, the mean drift
is at most 2% and the point drift is at most 10%.

To repeat the check:

```bash
uv sync --group onnx
uv run python tools/export_onnx.py theforecastingcompany/t0-gamma out/
uv run python tools/validate_onnx.py theforecastingcompany/t0-gamma out/*.onnx
```

### t0-gamma - 2026-10-23

- Reference package: `tfc-t0` 0.6.0.
- Hugging Face revision: `e08aee083778f1f32da4974c0a34276776a35c52`.
- Checkpoint SHA-256:
  `1603e255629be22ea45518b502f16c55d06821c7090a1a851673e779e9c245a6`.
- Environment: Linux x86_64 CPU, Python 3.12.3, PyTorch 2.14.1, ONNX 1.23.2,
  onnxscript 0.7.2, ONNX Runtime 1.31.0 (CPU execution provider). Opset 20.

| Export | Weights | Size | Worst mean drift | Worst point drift | Result |
| --- | --- | ---: | ---: | ---: | --- |
| `t0-gamma-grouped-fp32.onnx` | FP32 | 1023.0 MB | 0.0001% | 0.0016% | Passed 75 of 75 |
| `t0-gamma-grouped-fp16.onnx` | FP16 MatMul weights, FP32 compute | 512.4 MB | 0.038% | 0.82% | Passed 75 of 75 |
| `t0-gamma-grouped-int8.onnx` | INT8 transformer weights, FP32 compute | 269.2 MB | 0.62% | 54.1% | Failed 18 of 75 |

The INT8 export uses per-channel symmetric weights for the 96 transformer
matrices. All its failures are on point drift, at the 0.99 quantile, on contexts
of 512 steps or fewer. Most are on the one-step context. When some layers stay
in FP32, the worst point drift goes down but stays above the gate: 17.5% with
the first and last layers in FP32, and 14.1% when the down-projection of every
MLP also stays in FP32. Do not publish the INT8 export until a different
quantization method passes.

Browser execution (ONNX Runtime Web, WASM and WebGPU) was not checked.
