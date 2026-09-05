import logging

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest

from t0_mlx import Forecast, MaskType, T0Config, T0Forecaster, batch_series
from t0_mlx import __all__ as t0_all
from t0_mlx.data import TimeSeries, VariateType
from t0_mlx.model import _sanitize_predictions
from t0_mlx.scaler import CausalScaler


@pytest.fixture
def tiny_model() -> T0Forecaster:
    config = T0Config(
        embed_dim=16,
        num_layers=3,
        num_heads=4,
        mlp_hidden_dim=32,
        patch_size=4,
        group_every_n=3,
        dropout=0.0,
        quantile_levels=(0.1, 0.5, 0.9),
    )
    model = T0Forecaster.from_config(config)
    model.eval()
    return model


def test_top_level_api_matches_tfc_t0() -> None:
    assert t0_all == ["Forecast", "MaskType", "T0Config", "T0Forecaster", "batch_series"]


def test_scaler_uses_checkpoint_trained_per_timestep_statistics() -> None:
    values = mx.array([[1.0, 3.0, 5.0, 7.0]], dtype=mx.float32)
    model_input = TimeSeries(
        variates=values,
        mask=mx.full(values.shape, MaskType.VALID, dtype=mx.int8),
        group_ids=mx.zeros(values.shape, dtype=mx.int32),
        variate_type=mx.full(values.shape, VariateType.TARGET, dtype=mx.int32),
    )

    scaled, loc_scale = CausalScaler(use_arcsinh=False).scale_input(model_input)

    # PyTorch T0Forecaster uses CausalScaler(patch_size=1), so the running
    # location is retained at every timestep rather than frozen per model patch.
    np.testing.assert_allclose(np.asarray(loc_scale.loc), [[1.0, 2.0, 3.0, 4.0]])
    assert loc_scale.loc.shape == values.shape
    assert scaled.variates.shape == values.shape

    # Inverse scaling selects the statistic at the model patch's right edge.
    prediction = mx.zeros((1, 1, 4, 1), dtype=mx.float32)
    restored = CausalScaler(use_arcsinh=False).rescale_predictions(prediction, loc_scale, model_patch_size=4)
    np.testing.assert_allclose(np.asarray(restored), 4.0)


def test_predict_matches_public_shape_contract(tiny_model: T0Forecaster) -> None:
    context = np.arange(15, dtype=np.float32).reshape(3, 5)
    forecast = tiny_model.predict(context, horizon=7)
    assert forecast.quantiles.shape == (3, 7, 3)
    assert forecast.median.shape == (3, 7)
    assert forecast.quantile_levels == (0.1, 0.5, 0.9)
    assert np.isfinite(np.asarray(forecast.quantiles)).all()


def test_predict_accepts_one_dimensional_context(tiny_model: T0Forecaster) -> None:
    forecast = tiny_model.predict([1.0, 2.0, 3.0], horizon=2, quantiles=[0.5])
    assert forecast.quantiles.shape == (1, 2, 1)


def test_predict_handles_missing_values(tiny_model: T0Forecaster) -> None:
    forecast = tiny_model.predict(np.array([1.0, np.nan, 3.0], dtype=np.float32), horizon=2)
    assert np.isfinite(np.asarray(forecast.quantiles)).all()


def test_non_finite_predictions_are_replaced_and_logged(caplog: pytest.LogCaptureFixture) -> None:
    predictions = mx.array([float("nan"), float("inf"), -float("inf"), 2.0])
    with caplog.at_level(logging.WARNING, logger="t0_mlx.model"):
        sanitized = _sanitize_predictions(predictions)

    np.testing.assert_array_equal(np.asarray(sanitized), [0.0, 0.0, 0.0, 2.0])
    assert "replaced 3 non-finite prediction values with 0.0" in caplog.text


def test_forecast_interpolates_median() -> None:
    forecast = Forecast(
        quantiles=mx.array([[[1.0, 5.0]]], dtype=mx.float32),
        quantile_levels=(0.1, 0.9),
    )
    np.testing.assert_allclose(np.asarray(forecast.median), [[3.0]])


@pytest.mark.parametrize(
    "kwargs",
    [{"future_covariates": np.zeros((1, 1, 7), dtype=np.float32), "group_ids": np.array([0])}],
)
def test_unimplemented_structures_fail_explicitly(tiny_model: T0Forecaster, kwargs) -> None:
    with pytest.raises(ValueError, match="cannot be combined"):
        tiny_model.predict([1.0, 2.0, 3.0], horizon=4, **kwargs)


def test_multivariate_context_preserves_batch_and_variate_axes(tiny_model: T0Forecaster) -> None:
    forecast = tiny_model.predict(np.ones((2, 3, 8), dtype=np.float32), horizon=4)
    assert forecast.quantiles.shape == (2, 3, 4, 3)


def test_explicit_groups_are_accepted(tiny_model: T0Forecaster) -> None:
    forecast = tiny_model.predict(
        np.ones((4, 8), dtype=np.float32),
        horizon=4,
        group_ids=np.array([0, 0, 1, 2]),
    )
    assert forecast.quantiles.shape == (4, 4, 3)


def test_future_covariates_are_accepted(tiny_model: T0Forecaster) -> None:
    context = np.ones((2, 8), dtype=np.float32)
    future = np.ones((2, 3, 12), dtype=np.float32)
    forecast = tiny_model.predict(context, horizon=4, future_covariates=future)
    assert forecast.quantiles.shape == (2, 4, 3)


def test_autoregressive_rollout_extends_past_max_horizon(tiny_model: T0Forecaster) -> None:
    tiny_model.max_horizon = 8
    forecast = tiny_model.predict([1.0, 2.0, 3.0], horizon=12)
    assert forecast.quantiles.shape == (1, 12, 3)
    assert np.isfinite(np.asarray(forecast.quantiles)).all()


def test_max_horizon_must_align_to_patch_size(tiny_model: T0Forecaster) -> None:
    tiny_model.max_horizon = 6
    with pytest.raises(ValueError, match="positive multiple"):
        tiny_model.predict([1.0, 2.0, 3.0], horizon=8)


def test_mask_cannot_mark_nan_valid(tiny_model: T0Forecaster) -> None:
    with pytest.raises(ValueError, match="marks NaN"):
        tiny_model.predict([1.0, np.nan, 3.0], horizon=2, mask=np.zeros(3, dtype=np.int8))


def test_group_ids_must_match_flat_rows(tiny_model: T0Forecaster) -> None:
    with pytest.raises(ValueError, match="one id per target row"):
        tiny_model.predict(np.ones((2, 8), dtype=np.float32), horizon=2, group_ids=np.array([0]))


def test_local_pretrained_round_trip(tiny_model: T0Forecaster, tmp_path) -> None:
    tiny_model.save_pretrained(tmp_path)

    restored = T0Forecaster.from_pretrained(tmp_path)
    expected = tiny_model.predict([1.0, 2.0, 3.0], horizon=3).quantiles
    actual = restored.predict([1.0, 2.0, 3.0], horizon=3).quantiles
    mx.eval(expected, actual)
    np.testing.assert_array_equal(np.asarray(actual), np.asarray(expected))
    assert not restored.training
    assert isinstance(restored, nn.Module)


def test_batch_series_matches_public_contract() -> None:
    context, mask, group_ids = batch_series(
        [
            np.array([1.0, 2.0, 3.0]),
            np.array([[4.0, np.nan], [5.0, 6.0]]),
        ]
    )
    assert context.shape == (3, 3)
    np.testing.assert_array_equal(np.asarray(group_ids), [0, 1, 1])
    np.testing.assert_array_equal(
        np.asarray(mask),
        [
            [MaskType.VALID, MaskType.VALID, MaskType.VALID],
            [MaskType.PAD, MaskType.VALID, MaskType.MISSING],
            [MaskType.PAD, MaskType.VALID, MaskType.VALID],
        ],
    )


def test_compiled_prediction_matches_eager(tiny_model: T0Forecaster) -> None:
    context = np.arange(9, dtype=np.float32)
    eager = tiny_model.predict(context, horizon=5).quantiles
    compiled = tiny_model.compile().predict(context, horizon=5).quantiles
    mx.eval(eager, compiled)
    np.testing.assert_allclose(np.asarray(compiled), np.asarray(eager), rtol=1e-5, atol=1e-5)
    assert tiny_model.uncompile() is tiny_model
