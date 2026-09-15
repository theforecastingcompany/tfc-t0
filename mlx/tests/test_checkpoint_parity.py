"""Parity gates against the original PyTorch model and checkpoint."""

import os
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest

from t0_mlx import T0Config
from t0_mlx import batch_series as mlx_batch_series
from t0_mlx.data import TimeSeries as MLXTimeSeries
from t0_mlx.layers import PatchEncoder
from t0_mlx.quantile import reduce_rollout_quantiles
from t0_mlx.scaler import CausalScaler as MLXCausalScaler

CHECKPOINT_ENV = "T0_MLX_CHECKPOINT"


def _checkpoint() -> Path:
    value = os.environ.get(CHECKPOINT_ENV)
    if value is None:
        pytest.skip(f"set {CHECKPOINT_ENV} to run checkpoint-backed parity")
    path = Path(value)
    if not (path / "config.json").is_file() or not (path / "model.safetensors").is_file():
        pytest.fail(f"{path} must contain config.json and model.safetensors")
    return path


def test_patch_encoder_matches_pytorch_checkpoint() -> None:
    torch = pytest.importorskip("torch")
    t0 = pytest.importorskip("t0")
    checkpoint = _checkpoint()
    config = T0Config.from_json(checkpoint / "config.json")

    reference = t0.T0Forecaster.from_pretrained(checkpoint).eval()
    candidate = PatchEncoder(config.embed_dim, config.patch_size)
    weights = mx.load(str(checkpoint / "model.safetensors"))
    prefix = "patch_encoder."
    candidate.load_weights(
        [(key.removeprefix(prefix), value) for key, value in weights.items() if key.startswith(prefix)]
    )
    candidate.eval()

    rng = np.random.default_rng(17)
    values = rng.normal(size=(3, 4, config.patch_size)).astype(np.float32)
    mask = np.zeros(values.shape, dtype=np.int8)
    mask[1, 0, :7] = 1
    mask[2, 2, 9:15] = 2
    values[mask != 0] = 0.0
    variate_type = np.zeros(values.shape, dtype=np.int64)
    variate_type[2] = 2

    with torch.inference_mode():
        expected = reference.patch_encoder(
            torch.from_numpy(values),
            torch.from_numpy(mask),
            torch.from_numpy(variate_type),
        ).numpy()
    actual = candidate(mx.array(values), mx.array(mask), mx.array(variate_type))
    mx.eval(actual)

    np.testing.assert_allclose(np.asarray(actual), expected, rtol=1e-5, atol=2e-5)


def test_scaler_and_rollout_quantiles_match_pytorch() -> None:
    torch = pytest.importorskip("torch")
    pytest.importorskip("t0")
    from t0.data import TimeSeries as TorchTimeSeries
    from t0.quantile import QuantileRolloutReducer
    from t0.scaler import CausalScaler as TorchCausalScaler

    values = np.array(
        [[1.0, 2.0, np.nan, 8.0, 13.0], [3.0, -2.0, 4.0, 7.0, 11.0]],
        dtype=np.float32,
    )
    torch_input = TorchTimeSeries.from_array(torch.from_numpy(values))
    mlx_input = MLXTimeSeries.from_context(mx.array(values))
    with torch.inference_mode():
        expected_scaled, expected_stats = TorchCausalScaler(patch_size=1, use_arcsinh=True).scale_input(torch_input)
    actual_scaled, actual_stats = MLXCausalScaler(use_arcsinh=True).scale_input(mlx_input)
    mx.eval(actual_scaled.variates, actual_stats.loc, actual_stats.scale)

    np.testing.assert_allclose(np.asarray(actual_scaled.variates), expected_scaled.variates.numpy(), atol=2e-6)
    np.testing.assert_allclose(np.asarray(actual_stats.loc), expected_stats.loc.numpy(), atol=2e-6)
    np.testing.assert_allclose(np.asarray(actual_stats.scale), expected_stats.scale.numpy(), atol=2e-6)

    rng = np.random.default_rng(31)
    predictions = rng.normal(size=(2, 3, 5, 7)).astype(np.float32)
    predicted_levels = [0.1, 0.25, 0.5, 0.75, 0.9]
    query_levels = [0.1, 0.5, 0.9]
    with torch.inference_mode():
        expected_quantiles = QuantileRolloutReducer(
            predicted_quantile_levels=torch.tensor(predicted_levels),
            query_quantile_levels=torch.tensor(query_levels),
        ).reduce(torch.from_numpy(predictions))
    actual_quantiles = reduce_rollout_quantiles(mx.array(predictions), predicted_levels, query_levels)
    mx.eval(actual_quantiles)
    np.testing.assert_allclose(np.asarray(actual_quantiles), expected_quantiles.numpy(), rtol=1e-6, atol=2e-6)


def test_attention_and_rotary_layers_match_pytorch_checkpoint() -> None:
    torch = pytest.importorskip("torch")
    t0 = pytest.importorskip("t0")
    from t0_mlx import T0Forecaster as MLXT0Forecaster

    checkpoint = _checkpoint()
    reference = t0.T0Forecaster.from_pretrained(checkpoint).eval()
    candidate = MLXT0Forecaster.from_pretrained(checkpoint).eval()
    rng = np.random.default_rng(29)

    head_dim = candidate.config.embed_dim // candidate.config.num_heads
    queries = rng.normal(size=(2, candidate.config.num_heads, 5, head_dim)).astype(np.float32)
    keys = rng.normal(size=queries.shape).astype(np.float32)
    with torch.inference_mode():
        expected_queries, expected_keys = reference.transformer.rotary_emb.rotate_queries_and_keys(
            torch.from_numpy(queries),
            torch.from_numpy(keys),
            seq_dim=-2,
        )
    rotary = candidate.transformer.layers[0].attention_block.attention.rotary
    assert rotary is not None
    actual_queries, actual_keys = rotary.rotate_queries_and_keys(mx.array(queries), mx.array(keys))
    mx.eval(actual_queries, actual_keys)
    np.testing.assert_allclose(np.asarray(actual_queries), expected_queries.numpy(), rtol=1e-5, atol=2e-5)
    np.testing.assert_allclose(np.asarray(actual_keys), expected_keys.numpy(), rtol=1e-5, atol=2e-5)

    hidden = rng.normal(size=(3, 5, candidate.config.embed_dim)).astype(np.float32)
    time_mask = np.broadcast_to(np.tril(np.ones((5, 5), dtype=bool)), (3, 1, 5, 5)).copy()
    with torch.inference_mode():
        expected_time = reference.transformer.layers[0].attention_block(
            torch.from_numpy(hidden), torch.from_numpy(time_mask)
        )
    actual_time = candidate.transformer.layers[0].attention_block(mx.array(hidden), mx.array(time_mask))
    mx.eval(actual_time)
    np.testing.assert_allclose(np.asarray(actual_time), expected_time.numpy(), rtol=2e-5, atol=2e-4)

    group_index = candidate.config.group_every_n - 1
    group_mask = np.ones((5, 1, 3, 3), dtype=bool)
    with torch.inference_mode():
        expected_group = reference.transformer.layers[group_index].attention_block(
            torch.from_numpy(hidden), torch.from_numpy(group_mask)
        )
    actual_group = candidate.transformer.layers[group_index].attention_block(mx.array(hidden), mx.array(group_mask))
    mx.eval(actual_group)
    np.testing.assert_allclose(np.asarray(actual_group), expected_group.numpy(), rtol=2e-5, atol=2e-4)


def test_single_pass_forecast_matches_pytorch_checkpoint() -> None:
    torch = pytest.importorskip("torch")
    t0 = pytest.importorskip("t0")
    from t0_mlx import T0Forecaster as MLXT0Forecaster

    checkpoint = _checkpoint()
    reference = t0.T0Forecaster.from_pretrained(checkpoint).eval()
    assert reference.scaler.patch_size == 1
    candidate = MLXT0Forecaster.from_pretrained(checkpoint)

    time = np.arange(97, dtype=np.float32)
    context = np.stack(
        [
            0.25 * time + 3.0,
            np.sin(time / 7.0) + 0.2 * np.cos(time / 3.0),
            np.sqrt(time + 1.0),
        ]
    ).astype(np.float32)
    context[2, [3, 4, 31, 80]] = np.nan
    quantiles = [0.1, 0.2, 0.5, 0.8, 0.9]

    with torch.inference_mode():
        expected = reference.predict(context, horizon=64, quantiles=quantiles).quantiles.numpy()
    actual = np.asarray(candidate.predict(context, horizon=64, quantiles=quantiles).quantiles)
    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=2e-4)

    compiled = np.asarray(candidate.compile().predict(context, horizon=64, quantiles=quantiles).quantiles)
    np.testing.assert_allclose(compiled, expected, rtol=1e-5, atol=2e-4)

    candidate.uncompile()
    multivariate = context[:2].reshape(1, 2, -1)
    with torch.inference_mode():
        expected_multivariate = reference.predict(multivariate, horizon=32, quantiles=[0.1, 0.5, 0.9]).quantiles.numpy()
    actual_multivariate = np.asarray(candidate.predict(multivariate, horizon=32, quantiles=[0.1, 0.5, 0.9]).quantiles)
    np.testing.assert_allclose(actual_multivariate, expected_multivariate, rtol=1e-5, atol=2e-4)

    grouped = context
    group_ids = np.array([0, 0, 1])
    with torch.inference_mode():
        expected_grouped = reference.predict(
            grouped,
            horizon=32,
            quantiles=[0.1, 0.5, 0.9],
            group_ids=group_ids,
        ).quantiles.numpy()
    actual_grouped = np.asarray(
        candidate.predict(grouped, horizon=32, quantiles=[0.1, 0.5, 0.9], group_ids=group_ids).quantiles
    )
    np.testing.assert_allclose(actual_grouped, expected_grouped, rtol=1e-5, atol=2e-4)

    ragged_series = [context[0, :17], np.stack([context[1, :65], context[2, :65]])]
    torch_context, torch_mask, torch_groups = t0.batch_series(ragged_series)
    mlx_context, mlx_mask, mlx_groups = mlx_batch_series(ragged_series)
    with torch.inference_mode():
        expected_ragged = reference.predict(
            torch_context,
            horizon=32,
            quantiles=[0.1, 0.5, 0.9],
            mask=torch_mask,
            group_ids=torch_groups,
        ).quantiles.numpy()
    actual_ragged = np.asarray(
        candidate.predict(
            mlx_context,
            horizon=32,
            quantiles=[0.1, 0.5, 0.9],
            mask=mlx_mask,
            group_ids=mlx_groups,
        ).quantiles
    )
    np.testing.assert_allclose(actual_ragged, expected_ragged, rtol=2e-5, atol=3e-4)
    actual_short_alone = np.asarray(
        candidate.predict(ragged_series[0], horizon=32, quantiles=[0.1, 0.5, 0.9]).quantiles
    )[0]
    np.testing.assert_allclose(actual_ragged[0], actual_short_alone, rtol=2e-5, atol=3e-4)

    rng = np.random.default_rng(23)
    future = rng.normal(size=(3, 2, context.shape[-1] + 32)).astype(np.float32)
    future[1, 0, [4, 103]] = np.nan
    with torch.inference_mode():
        expected_future = reference.predict(
            context,
            horizon=32,
            quantiles=[0.1, 0.5, 0.9],
            future_covariates=future,
        ).quantiles.numpy()
    actual_future = np.asarray(
        candidate.predict(context, horizon=32, quantiles=[0.1, 0.5, 0.9], future_covariates=future).quantiles
    )
    np.testing.assert_allclose(actual_future, expected_future, rtol=1e-5, atol=2e-4)

    reference.max_horizon = 64
    candidate.max_horizon = 64
    rollout_context = context[:1, :64]
    with torch.inference_mode():
        expected_rollout = reference.predict(
            rollout_context,
            horizon=96,
            quantiles=[0.1, 0.5, 0.9],
        ).quantiles.numpy()
    actual_rollout = np.asarray(candidate.predict(rollout_context, horizon=96, quantiles=[0.1, 0.5, 0.9]).quantiles)
    np.testing.assert_allclose(actual_rollout, expected_rollout, rtol=2e-5, atol=3e-4)

    rollout_future = rng.normal(size=(1, 1, rollout_context.shape[-1] + 96)).astype(np.float32)
    with torch.inference_mode():
        expected_future_rollout = reference.predict(
            rollout_context,
            horizon=96,
            quantiles=[0.1, 0.5, 0.9],
            future_covariates=rollout_future,
        ).quantiles.numpy()
    actual_future_rollout = np.asarray(
        candidate.predict(
            rollout_context,
            horizon=96,
            quantiles=[0.1, 0.5, 0.9],
            future_covariates=rollout_future,
        ).quantiles
    )
    np.testing.assert_allclose(actual_future_rollout, expected_future_rollout, rtol=2e-5, atol=3e-4)
