"""T0 matplotlib style utilities.

The rcParams live in ``t0.mplstyle`` (matplotlib's native style format,
shipped with the package); this module loads it and defines the semantic
color palette. Both follow the chart design of Retrocast
(https://retrocast.com), The Forecasting Company's forecasting platform.
"""

from importlib import resources

try:
    import matplotlib.pyplot as plt
except ImportError as err:  # pragma: no cover - exercised only without the extra
    raise ImportError('t0.utils requires the plot extra: pip install "tfc-t0[plot]"') from err


def use_t0_style() -> None:
    """Activate the T0 matplotlib style globally.

    Example::

        from t0.utils.style import use_t0_style

        use_t0_style()
        # All subsequent plots use the T0 style
    """
    with resources.as_file(resources.files("t0.utils") / "t0.mplstyle") as style_path:
        plt.style.use(style_path)


# Semantic color palette (Retrocast light mode) — use these instead of
# hardcoded color strings.
COLOR_TARGET = "#000000"
COLOR_FORECAST = "#0DA846"
COLOR_CONTEXT = "#a1a1aa"

# Quantile / confidence bands
COLOR_PREDICTION_INTERVAL = "#0DA846"  # same green, rendered with low alpha
PREDICTION_INTERVAL_ALPHA = 0.10

# Auxiliary
COLOR_FCD_LINE = "#71717a"
COLOR_TEXT = "#333333"

# Missing-data shading
MASK_SHADE_ALPHA = 0.12
COLOR_MISSING_SHADE = "red"
