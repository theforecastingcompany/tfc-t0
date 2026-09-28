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
