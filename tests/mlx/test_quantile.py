"""IQF exponential tails wired through the MLX runtime's predict()."""

import math

import numpy as np
import pytest

from t0.mlx import T0Config, T0Forecaster

TRAINED_LEVELS = (0.1, 0.25, 0.5, 0.75, 0.9)


def reference_tail(knot_levels, knot_values, level: float) -> float:
    """Park et al. (2022) Eq. 6/9 tails pinned through the outermost knots."""
    k, v = knot_levels, knot_values
    if level < k[0]:
        slope = max((v[1] - v[0]) / math.log(k[1] / k[0]), 0.0)
        return v[0] + slope * math.log(level / k[0])
    slope = max((v[-1] - v[-2]) / math.log((1.0 - k[-2]) / (1.0 - k[-1])), 0.0)
    return v[-1] + slope * math.log((1.0 - k[-1]) / (1.0 - level))


def _model(quantile_levels: tuple[float, ...]) -> T0Forecaster:
    config = T0Config(
        embed_dim=16,
        num_layers=3,
        num_heads=4,
        mlp_hidden_dim=32,
        patch_size=4,
        group_every_n=3,
        dropout=0.0,
        quantile_levels=quantile_levels,
    )
    model = T0Forecaster.from_config(config)
    model.eval()
    return model


@pytest.fixture
def tiny_model() -> T0Forecaster:
    return _model((0.1, 0.5, 0.9))


@pytest.fixture
def published_model() -> T0Forecaster:
    """Trained levels wider than the requested interior, as the published model's are."""
    return _model(TRAINED_LEVELS)


CONTEXT = [1.0, 2.0, 3.0, 4.0]


def test_predict_returns_only_the_requested_columns(published_model: T0Forecaster) -> None:
    forecast = published_model.predict(CONTEXT, horizon=8, quantile_levels=(0.01, 0.5, 0.99))
    assert np.asarray(forecast.quantiles).shape[-1] == 3
    assert forecast.quantile_levels == (0.01, 0.5, 0.99)


def test_predict_tail_quantiles(tiny_model: T0Forecaster) -> None:
    """Tail levels through predict() follow the tails pinned by the trained-level columns."""
    requested = (0.01, 0.1, 0.5, 0.9, 0.99)
    forecast = tiny_model.predict(CONTEXT, horizon=8, quantile_levels=requested)
    knots = (0.1, 0.5, 0.9)  # this model trains three levels; both tails pin through its outermost pairs
    values = np.asarray(forecast.quantiles)
    knot_columns = [values[..., requested.index(knot)] for knot in knots]
    for level, column in ((0.01, values[..., 0]), (0.99, values[..., -1])):
        for step in range(values.shape[1]):
            expected = reference_tail(knots, [float(c[0, step]) for c in knot_columns], level)
            assert column[0, step] == pytest.approx(expected, rel=1e-5, abs=1e-5)


def test_predict_output_is_monotone(published_model: T0Forecaster) -> None:
    forecast = published_model.predict(CONTEXT, horizon=8, quantile_levels=(0.01, 0.1, 0.5, 0.9, 0.99))
    assert (np.diff(np.asarray(forecast.quantiles), axis=-1) >= -1e-6).all()


def test_single_pass_interior_unchanged_by_tail_request(published_model: T0Forecaster) -> None:
    """Below max_horizon each level is interpolated independently, so the interior cannot move."""
    interior = published_model.predict(CONTEXT, horizon=8, quantile_levels=(0.1, 0.5, 0.9))
    with_tails = published_model.predict(CONTEXT, horizon=8, quantile_levels=(0.01, 0.1, 0.5, 0.9, 0.99))
    assert np.asarray(with_tails.quantiles)[..., 1:4] == pytest.approx(np.asarray(interior.quantiles))


def test_rollout_interior_unchanged_when_the_request_holds_the_knots(published_model: T0Forecaster) -> None:
    """Under rollout the grid is unchanged only if the request already carries the pinning knots."""
    published_model.max_horizon = 8
    interior = published_model.predict(CONTEXT, horizon=20, quantile_levels=TRAINED_LEVELS)
    with_tails = published_model.predict(CONTEXT, horizon=20, quantile_levels=(0.01, *TRAINED_LEVELS, 0.99))
    assert np.asarray(with_tails.quantiles)[..., 1:6] == pytest.approx(np.asarray(interior.quantiles))


@pytest.mark.parametrize("requested", [(0.01, 0.5), (0.5, 0.99), (0.01, 0.99), (0.01,)])
def test_predict_handles_one_sided_tail_requests(published_model: T0Forecaster, requested: tuple[float, ...]) -> None:
    """One tail without the other, and tails without any interior level, still assemble."""
    forecast = published_model.predict(CONTEXT, horizon=8, quantile_levels=requested)
    values = np.asarray(forecast.quantiles)
    assert forecast.quantile_levels == requested
    assert values.shape[-1] == len(requested)
    assert np.isfinite(values).all()
