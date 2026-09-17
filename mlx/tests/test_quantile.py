"""IQF exponential tails: the level grid, the tail math, and the predict() wiring."""

import math

import mlx.core as mx
import numpy as np
import pytest

from t0_mlx import T0Config, T0Forecaster
from t0_mlx.quantile import (
    extend_quantile_tails,
    extrapolate_quantiles,
    get_rollout_quantile_levels,
    select_tail_knots,
)

TRAINED_LEVELS = (0.1, 0.25, 0.5, 0.75, 0.9)
TAIL_KNOTS = (0.1, 0.25, 0.75, 0.9)
TAIL_LEVELS = (0.01, 0.02, 0.05, 0.95, 0.98, 0.99)
N_LEFT = 3
# Closed-form values for TAIL_LEVELS pinned through TAIL_KNOTS on `knot_values`, shared
# verbatim with the PyTorch suite so both runtimes answer to one oracle.
GOLDEN_TAILS = [
    [-15.129416, -7.564708, 2.435292, 57.564708, 67.564708, 75.129416],
    [-15.051766, -12.025883, -8.025883, 12.538825, 18.538825, 23.077650],
]


def reference_tail(knot_levels: tuple[float, ...], knot_values: list[float], level: float) -> float:
    """Park et al. (2022) Eq. 6/9 tails pinned through the outermost knots."""
    k = knot_levels
    v = knot_values
    if level < k[0]:
        slope = max((v[1] - v[0]) / math.log(k[1] / k[0]), 0.0)
        return v[0] + slope * math.log(level / k[0])
    slope = max((v[-1] - v[-2]) / math.log((1.0 - k[-2]) / (1.0 - k[-1])), 0.0)
    return v[-1] + slope * math.log((1.0 - k[-1]) / (1.0 - level))


@pytest.fixture
def knot_values() -> mx.array:
    return mx.array([[10.0, 20.0, 40.0, 50.0], [-5.0, -1.0, 2.0, 8.0]], dtype=mx.float32)


@pytest.fixture
def extended(knot_values: mx.array) -> np.ndarray:
    return np.asarray(extend_quantile_tails(TAIL_LEVELS, TAIL_KNOTS, knot_values))


def test_matches_reference_formula(extended: np.ndarray, knot_values: mx.array) -> None:
    values = np.asarray(knot_values)
    for row in range(values.shape[0]):
        for column, level in enumerate(TAIL_LEVELS):
            expected = reference_tail(TAIL_KNOTS, list(values[row]), level)
            assert extended[row, column] == pytest.approx(expected, rel=1e-6)


def test_matches_the_shared_golden_values(extended: np.ndarray) -> None:
    np.testing.assert_allclose(extended, np.array(GOLDEN_TAILS), rtol=1e-5, atol=1e-5)


def test_allocates_only_the_tail_columns(extended: np.ndarray, knot_values: mx.array) -> None:
    assert extended.shape == (knot_values.shape[0], len(TAIL_LEVELS))


def test_each_tail_is_strictly_monotone(extended: np.ndarray) -> None:
    assert (np.diff(extended[:, :N_LEFT], axis=-1) > 0).all()
    assert (np.diff(extended[:, N_LEFT:], axis=-1) > 0).all()


def test_tails_extend_strictly_past_the_boundary_knots(extended: np.ndarray, knot_values: mx.array) -> None:
    values = np.asarray(knot_values)
    assert (extended[:, :N_LEFT] < values[:, :1]).all()
    assert (extended[:, N_LEFT:] > values[:, -1:]).all()


@pytest.mark.parametrize("levels", [(0.01, 0.02), (0.98, 0.99), TAIL_LEVELS])
def test_one_sided_tail_sets(levels: tuple[float, ...], knot_values: mx.array) -> None:
    extended = np.asarray(extend_quantile_tails(levels, TAIL_KNOTS, knot_values))
    assert extended.shape == (knot_values.shape[0], len(levels))
    assert np.isfinite(extended).all()


def test_degenerate_equal_knots_stay_flat_and_finite() -> None:
    extended = extend_quantile_tails(TAIL_LEVELS, TAIL_KNOTS, mx.full((2, 4), 7.0))
    assert np.asarray(extended) == pytest.approx(np.full((2, len(TAIL_LEVELS)), 7.0))


def test_crossed_knots_clamp_to_the_flat_tail() -> None:
    extended = np.asarray(extend_quantile_tails(TAIL_LEVELS, TAIL_KNOTS, mx.array([[10.0, 4.0, 40.0, 12.0]])))
    assert extended[:, :N_LEFT] == pytest.approx(np.full((1, N_LEFT), 10.0))
    assert extended[:, N_LEFT:] == pytest.approx(np.full((1, len(TAIL_LEVELS) - N_LEFT), 12.0))


@pytest.mark.parametrize(
    "requested, expected",
    [
        ((0.01, 0.5, 0.99), TAIL_KNOTS),
        ((0.1, 0.5, 0.9), None),
        ((0.2, 0.8), None),
        ((0.95,), TAIL_KNOTS),
    ],
)
def test_select_tail_knots(requested: tuple[float, ...], expected: tuple[float, ...] | None) -> None:
    assert select_tail_knots(TRAINED_LEVELS, requested) == expected


def test_select_tail_knots_with_float32_trained_levels() -> None:
    levels32 = [float(level) for level in mx.array(TRAINED_LEVELS, dtype=mx.float32)]
    assert select_tail_knots(levels32, (0.1, 0.5, 0.9)) is None
    assert select_tail_knots(levels32, (0.01,)) == TAIL_KNOTS


@pytest.mark.parametrize(
    "trained, requested, expected",
    [
        (TRAINED_LEVELS, (0.01, 0.5, 0.99), (0.1, 0.25, 0.5, 0.75, 0.9)),
        (TRAINED_LEVELS, (0.1, 0.5, 0.9), (0.1, 0.5, 0.9)),
        (TRAINED_LEVELS, (0.01, 0.99), TAIL_KNOTS),
        (TRAINED_LEVELS, (0.01, 0.25, 0.5), (0.1, 0.25, 0.5, 0.75, 0.9)),
        ((0.1, 0.9), (0.01, 0.5), (0.1, 0.5, 0.9)),
        ((0.1, 0.5, 0.9), (0.01, 0.5), (0.1, 0.5, 0.9)),
    ],
)
def test_get_rollout_quantile_levels(
    trained: tuple[float, ...], requested: tuple[float, ...], expected: tuple[float, ...]
) -> None:
    assert get_rollout_quantile_levels(trained, requested) == expected


def test_rollout_levels_never_contain_a_tail_level() -> None:
    rollout = get_rollout_quantile_levels(TRAINED_LEVELS, (0.001, 0.01, 0.5, 0.99, 0.999))
    assert not [level for level in rollout if level < 0.1 or level > 0.9]


def test_extrapolate_quantiles_subsets_back_to_the_request() -> None:
    """The rollout grid is wider than the request, so the result must be subset back."""
    rollout = get_rollout_quantile_levels(TRAINED_LEVELS, (0.01, 0.5, 0.99))
    predictions = mx.broadcast_to(mx.arange(len(rollout), dtype=mx.float32), (2, len(rollout)))
    out = extrapolate_quantiles((0.01, 0.5, 0.99), rollout, predictions, TRAINED_LEVELS)
    assert out.shape == (2, 3)
    assert np.asarray(out)[:, 1] == pytest.approx(np.asarray(predictions)[:, rollout.index(0.5)])


def test_extrapolate_quantiles_returns_interior_requests_untouched() -> None:
    rollout = get_rollout_quantile_levels(TRAINED_LEVELS, (0.1, 0.5, 0.9))
    predictions = mx.broadcast_to(mx.arange(3, dtype=mx.float32), (2, 3))
    out = extrapolate_quantiles((0.1, 0.5, 0.9), rollout, predictions, TRAINED_LEVELS)
    assert np.asarray(out) == pytest.approx(np.asarray(predictions))


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


def test_single_trained_level_serves_interior_requests() -> None:
    """A model with one trained level cannot pin a tail, but interior requests still work."""
    assert select_tail_knots((0.5,), (0.5,)) is None


def test_single_trained_level_rejects_a_tail_request() -> None:
    with pytest.raises(ValueError, match="at least two trained quantile levels"):
        select_tail_knots((0.5,), (0.01, 0.5))
