# Copyright 2026 The Forecasting Company
# SPDX-License-Identifier: Apache-2.0

"""Tests for the ``non_negative`` quantile clip on ``T0Forecaster.predict``."""

import pytest
import torch

from t0.config import T0Config
from t0.data import MaskType, TimeSeries, VariateType
from t0.model.model import T0Forecaster
from t0.model.rollout import RolloutManager


@pytest.fixture
def tiny_model() -> T0Forecaster:
    torch.manual_seed(2)
    config = T0Config(
        embed_dim=32,
        num_layers=2,
        num_heads=4,
        mlp_hidden_dim=64,
        patch_size=8,
        group_every_n=1,
        dropout=0.0,
        quantile_levels=(0.1, 0.5, 0.9),
    )
    return T0Forecaster.from_config(config).eval()


def test_lower_bound_clip_selects_rows_by_observed_sign() -> None:
    """A row is clipped only when its observed cells hold no negative value."""
    variates = torch.tensor(
        [
            [1.0, 2.0, 0.0, 3.0],  # every observation non-negative -> clip
            [1.0, -2.0, 3.0, 4.0],  # a negative observation -> no clip
            [-9.0, 1.0, 2.0, 3.0],  # the negative cell is PAD, not observed -> clip
        ]
    )
    mask = torch.tensor(
        [
            [MaskType.VALID, MaskType.VALID, MaskType.VALID, MaskType.VALID],
            [MaskType.VALID, MaskType.VALID, MaskType.VALID, MaskType.VALID],
            [MaskType.PAD, MaskType.VALID, MaskType.VALID, MaskType.VALID],
        ],
        dtype=torch.int8,
    )
    series = TimeSeries(
        variates=variates,
        mask=mask,
        group_ids=torch.zeros((3, 4), dtype=torch.long),
        variate_type=torch.full((3, 4), VariateType.TARGET, dtype=torch.long),
    )
    target_rows = series.variate_type[:, 0] == VariateType.TARGET

    clip = RolloutManager(model=None)._lower_bound_clip(series, target_rows)
    clipped = clip(torch.full((3, 2, 3), -5.0))

    assert (clipped[0] == 0.0).all()
    assert (clipped[1] == -5.0).all()
    assert (clipped[2] == 0.0).all()


def test_non_negative_defaults_off(tiny_model: T0Forecaster) -> None:
    """The default leaves the forecast identical to an explicit opt-out."""
    context = torch.rand(3, 40)
    default = tiny_model.predict(context, horizon=16, quantiles=[0.1, 0.5, 0.9])
    opted_out = tiny_model.predict(context, horizon=16, quantiles=[0.1, 0.5, 0.9], non_negative=False)
    assert torch.equal(default.quantiles, opted_out.quantiles)


def test_non_negative_clips_a_negative_forecast(tiny_model: T0Forecaster) -> None:
    """A non-negative context whose forecast dips below 0 is clipped to ``max(base, 0)``."""
    context = torch.rand(4, 40)
    base = tiny_model.predict(context, horizon=16, quantiles=[0.1, 0.5, 0.9])
    clipped = tiny_model.predict(context, horizon=16, quantiles=[0.1, 0.5, 0.9], non_negative=True)

    assert base.quantiles.min() < 0  # the model does forecast below zero here
    assert clipped.quantiles.min() >= 0
    # A single decode block has no rollout feed-back, so the clip is exactly a floor at 0.
    assert torch.equal(clipped.quantiles, base.quantiles.clamp_min(0.0))


def test_non_negative_leaves_a_negative_context_row_untouched(tiny_model: T0Forecaster) -> None:
    """A series with a negative observation keeps its unclipped forecast."""
    context = torch.rand(2, 40)
    context[1, 3] = -1.0
    base = tiny_model.predict(context, horizon=16, quantiles=[0.1, 0.5, 0.9])
    clipped = tiny_model.predict(context, horizon=16, quantiles=[0.1, 0.5, 0.9], non_negative=True)

    assert torch.equal(clipped.quantiles[1], base.quantiles[1])
    assert clipped.quantiles[0].min() >= 0


def test_non_negative_holds_through_the_rollout(tiny_model: T0Forecaster) -> None:
    """The clip stays non-negative when the horizon runs the autoregressive rollout."""
    tiny_model.max_horizon = 16  # force a horizon past one block
    context = torch.rand(2, 40)
    clipped = tiny_model.predict(context, horizon=40, quantiles=[0.1, 0.5, 0.9], non_negative=True)
    assert clipped.quantiles.min() >= 0
