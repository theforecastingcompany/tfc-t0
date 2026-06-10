"""Functions to plot T0 quantile forecasts.

Requires the ``plot`` extra: ``pip install "tfc-t0[plot]"``.
"""

import matplotlib.pyplot as plt
import numpy as np
from jaxtyping import Float
from matplotlib.figure import Figure

from t0.model import Forecast
from t0.utils.style import (
    COLOR_CONTEXT,
    COLOR_FCD_LINE,
    COLOR_FORECAST,
    COLOR_MISSING_SHADE,
    COLOR_PREDICTION_INTERVAL,
    COLOR_TARGET,
    MASK_SHADE_ALPHA,
    PREDICTION_INTERVAL_ALPHA,
    use_t0_style,
)

use_t0_style()


def _shade_missing_regions(ax, missing: np.ndarray, steps: np.ndarray) -> None:
    """Shade contiguous missing-data (NaN) regions on a plot axis."""
    if len(missing) == 0 or not np.any(missing):
        return
    half_step = (steps[1] - steps[0]) / 2 if len(steps) > 1 else 0.5
    in_region = False
    start = 0
    for i, m in enumerate(missing):
        if m and not in_region:
            start = i
            in_region = True
        elif not m and in_region:
            ax.axvspan(
                steps[start] - half_step,
                steps[i - 1] + half_step,
                alpha=MASK_SHADE_ALPHA,
                color=COLOR_MISSING_SHADE,
                zorder=0,
            )
            in_region = False
    if in_region:
        ax.axvspan(
            steps[start] - half_step, steps[-1] + half_step, alpha=MASK_SHADE_ALPHA, color=COLOR_MISSING_SHADE, zorder=0
        )


def _first_non_nan_index(values: np.ndarray) -> int:
    """Return index of the first non-NaN value, or 0 if all NaN/empty."""
    if len(values) == 0:
        return 0
    non_nan = ~np.isnan(values)
    if not np.any(non_nan):
        return 0
    return int(np.argmax(non_nan))


def plot_forecast(
    context: Float[np.ndarray, " context_length"],
    forecast: Forecast,
    target: Float[np.ndarray, " horizon"] | None = None,
    series: int = 0,
    title: str | None = None,
    max_context: int | None = None,
) -> Figure:
    """Plot one series' context, quantile forecast, and optional ground truth.

    Args:
        context: Past observations for the series (NaN = missing).
        forecast: A ``Forecast`` returned by ``T0Forecaster.predict``.
        target: Optional ground-truth values over the forecast horizon.
        series: Row of ``forecast`` to plot (for batched predictions).
        title: Optional plot title.
        max_context: If set, show only the most recent ``max_context``
            context observations.

    Returns:
        The matplotlib ``Figure``.
    """
    quantiles = forecast.quantiles[series].cpu().numpy()  # (horizon, Q), levels sorted ascending
    quantile_levels = np.asarray(forecast.quantile_levels, dtype=np.float32)
    horizon = quantiles.shape[0]

    context = np.asarray(context, dtype=np.float32)
    if max_context is not None:
        context = context[-max_context:]
    first_valid = _first_non_nan_index(context)
    context = context[first_valid:]
    context_steps = np.arange(first_valid, first_valid + len(context))
    forecast_steps = (
        np.arange(context_steps[-1] + 1, context_steps[-1] + 1 + horizon) if len(context) else np.arange(horizon)
    )

    fig, ax = plt.subplots(layout="constrained")

    ax.plot(context_steps, context, color=COLOR_CONTEXT, label="context")
    _shade_missing_regions(ax, np.isnan(context), context_steps)

    if target is not None:
        ax.plot(
            forecast_steps,
            np.asarray(target, dtype=np.float32)[:horizon],
            color=COLOR_TARGET,
            label="target (ground truth)",
        )

    ax.axvline(
        context_steps[-1] if len(context) else 0,
        color=COLOR_FCD_LINE,
        alpha=0.5,
        linestyle="--",
        label="forecast creation date",
    )

    ax.fill_between(
        forecast_steps,
        quantiles[:, 0],
        quantiles[:, -1],
        alpha=PREDICTION_INTERVAL_ALPHA,
        color=COLOR_PREDICTION_INTERVAL,
        label="prediction interval",
    )

    # Label only the outermost levels and the one closest to the median.
    alpha_colors = np.where(quantile_levels < 0.5, quantile_levels, 1 - quantile_levels)
    p50_idx = int(np.argmin(np.abs(quantile_levels - 0.5)))
    labeled_indices = {0, p50_idx, len(quantile_levels) - 1}
    for idx, (q_level, q_values, alpha) in enumerate(zip(quantile_levels, quantiles.T, alpha_colors, strict=True)):
        label = f"{q_level:.2f}" if idx in labeled_indices else None
        ax.plot(forecast_steps, q_values, color=COLOR_FORECAST, alpha=max(float(alpha), 0.15) * 2, label=label)

    ax.set_xlabel("Time Steps")
    if title is not None:
        fig.suptitle(title, y=0.99)
    # Single-row legend between the title and the axes.
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.95),
        ncol=len(handles),
        frameon=False,
        fontsize=10,
    )
    layout_engine = fig.get_layout_engine()
    assert layout_engine is not None  # plt.subplots(layout="constrained") always sets one
    layout_engine.set(rect=(0, 0, 1, 0.89))  # ty: ignore[unknown-argument]  # matplotlib stub omits ConstrainedLayoutEngine kwargs
    return fig
