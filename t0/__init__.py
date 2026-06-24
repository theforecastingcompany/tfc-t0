"""Open-weights t0-alpha forecasting model.

Public API:
    - T0Forecaster — ``nn.Module`` backbone with ``from_pretrained`` /
      ``save_pretrained`` (via ``huggingface_hub.ModelHubMixin``) and the
      user-facing ``predict(context, horizon, quantiles)``.
    - Forecast — the value ``predict`` returns (``quantiles`` + ``median``).
    - T0Config — frozen dataclass; ``T0Config.medium()`` for the
      published t0-alpha checkpoint.
"""

from t0.config import T0Config
from t0.model import Forecast, T0Forecaster

__all__ = ["Forecast", "T0Config", "T0Forecaster"]
