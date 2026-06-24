"""t0-alpha model package: the forecaster, its layers, and the rollout.

``T0Forecaster.predict`` returns a ``Forecast`` — the container holding the
predicted ``quantiles`` tensor, the requested ``quantile_levels``, and a
derived ``median``.
"""

from t0.model.model import Forecast, T0Forecaster

__all__ = ["Forecast", "T0Forecaster"]
