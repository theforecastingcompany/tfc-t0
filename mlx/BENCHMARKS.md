# Backend benchmarks

This is a reproducible comparison of the first-party MLX port with the
first-party PyTorch implementation. Both implementations loaded the same
original FP32 `model.safetensors`.

## Environment

- Date: 2026-09-04
- Machine: 14-inch MacBook Pro (2021), Apple M1 Pro, 10 CPU cores
  (8 performance and 2 efficiency), 16 GB unified memory
- Operating system: macOS 15.4.1 (24E263)
- Python: 3.14.3
- MLX: 0.32.2, default GPU device
- PyTorch: 2.13.0, MPS and CPU devices, 8 CPU threads
- Hugging Face revision:
  `7fbf2648f27133aa427f51d152cdaa35c0268f32`
- Checkpoint SHA-256:
  `16c030d3fd70f06dc4238e9a8356e9b5a631d07f80f1bc76ba539991aed5897f`
- MLX package: `tfc-t0-mlx` 0.1.0a0
- PyTorch reference package: `tfc-t0` 0.3.0

## Method

Each backend ran in an isolated subprocess. The benchmark exercised the
public `T0Forecaster.predict()` API with identical NumPy inputs generated from
seed 0. It performed 10 warm-up forecasts followed by 60 measured forecasts
for each shape. GPU operations were synchronized inside the timed region.
The final forecast from every accelerated backend was compared elementwise
with PyTorch CPU using `rtol=2e-5` and `atol=3e-4`; the command fails if any
element exceeds that tolerance.

Model loading is excluded. The MLX compiled measurements also exclude the
shape-dependent compilation and first execution, which are reported
separately. All measurements use FP32 weights and request the default 0.1,
0.5, and 0.9 quantiles.

Run the comparison from this directory with the published PyTorch package:

```bash
uv run --extra parity python tools/benchmark.py /path/to/t0-alpha \
  --warmup 10 --iterations 60
```

For local development of both packages, replace `--extra parity` with
`--with-editable ..`. `--backend`, `--workload`, `--torch-threads`,
`--parity-rtol`, `--parity-atol`, and `--format json` support narrower or
machine-readable runs; see `python tools/benchmark.py --help`.

## Results

Steady-state latency in milliseconds:

| Shape | MLX eager | MLX compiled | PyTorch MPS | PyTorch CPU | Compiled MLX vs MPS |
| --- | ---: | ---: | ---: | ---: | ---: |
| batch 1, context 96, horizon 32 | 13.86 | **10.73** | 36.18 | 59.19 | **3.37x** |
| batch 1, context 512, horizon 64 | 14.64 | **10.90** | 39.09 | 74.81 | **3.59x** |
| batch 8, context 96, horizon 32 | 16.33 | **12.45** | 43.12 | 65.06 | **3.46x** |

Full synchronized timing output:

| Backend | Shape | Minimum | Median | Maximum | Compile + first run | Checksum |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| mlx-eager | batch 1, context 96, horizon 32 | 11.74 ms | 13.86 ms | 20.04 ms | — | 11.805859 |
| mlx-eager | batch 1, context 512, horizon 64 | 12.78 ms | 14.64 ms | 17.60 ms | — | -20.967939 |
| mlx-eager | batch 8, context 96, horizon 32 | 13.46 ms | 16.33 ms | 20.45 ms | — | -87.408371 |
| mlx-compiled | batch 1, context 96, horizon 32 | 9.04 ms | 10.73 ms | 13.75 ms | 149.88 ms | 11.805849 |
| mlx-compiled | batch 1, context 512, horizon 64 | 9.89 ms | 10.90 ms | 13.92 ms | 22.86 ms | -20.967934 |
| mlx-compiled | batch 8, context 96, horizon 32 | 11.09 ms | 12.45 ms | 35.40 ms | 28.69 ms | -87.408325 |
| torch-mps | batch 1, context 96, horizon 32 | 28.85 ms | 36.18 ms | 91.85 ms | — | 11.805881 |
| torch-mps | batch 1, context 512, horizon 64 | 29.42 ms | 39.09 ms | 57.18 ms | — | -20.967865 |
| torch-mps | batch 8, context 96, horizon 32 | 32.65 ms | 43.12 ms | 56.11 ms | — | -87.408264 |
| torch-cpu | batch 1, context 96, horizon 32 | 54.58 ms | 59.19 ms | 88.84 ms | — | 11.805872 |
| torch-cpu | batch 1, context 512, horizon 64 | 66.56 ms | 74.81 ms | 294.42 ms | — | -20.967855 |
| torch-cpu | batch 8, context 96, horizon 32 | 58.55 ms | 65.06 ms | 161.00 ms | — | -87.408127 |

The checksums make it easy to spot a different workload. Numerical correctness
uses the complete elementwise comparisons below and the broader
checkpoint-backed FP32 parity suite recorded in [`PARITY.md`](PARITY.md).

| Backend | Shape | Maximum absolute error | Maximum relative error | Within tolerance |
| --- | --- | ---: | ---: | ---: |
| MLX eager | batch 1, context 96, horizon 32 | 1.788e-6 | 8.574e-5 | yes |
| MLX eager | batch 1, context 512, horizon 64 | 1.907e-6 | 8.188e-6 | yes |
| MLX eager | batch 8, context 96, horizon 32 | 2.503e-6 | 1.161e-3 | yes |
| MLX compiled | batch 1, context 96, horizon 32 | 1.907e-6 | 8.574e-5 | yes |
| MLX compiled | batch 1, context 512, horizon 64 | 2.146e-6 | 8.545e-6 | yes |
| MLX compiled | batch 8, context 96, horizon 32 | 2.742e-6 | 1.357e-3 | yes |
| PyTorch MPS | batch 1, context 96, horizon 32 | 1.073e-6 | 1.102e-4 | yes |
| PyTorch MPS | batch 1, context 512, horizon 64 | 1.192e-6 | 6.628e-6 | yes |
| PyTorch MPS | batch 8, context 96, horizon 32 | 2.623e-6 | 5.607e-3 | yes |

## Interpretation

For these short patch-transformer workloads, MLX eager was 2.61–2.67x faster
than PyTorch MPS. Compiled MLX was 3.37–3.59x faster than PyTorch MPS and
5.23–6.86x faster than PyTorch CPU.

Compilation is best suited to repeated shapes. Its first-call cost is not part
of the steady-state comparison and may differ on a cold machine or after a
framework update. Results are specific to this hardware and software stack;
other batch sizes, horizons, and Apple chips should be measured independently.
