"""The MLX runtime of t0, for Apple silicon. Requires the ``mlx`` extra: ``pip install 'tfc-t0[mlx]'``."""

try:
    import mlx.core
except ImportError as error:
    raise ImportError(
        "t0.mlx is the MLX runtime of t0 for Apple silicon. Install through pip install 'tfc-t0[mlx]' "
        "or use the PyTorch runtime with `from t0 import T0Forecaster`"
    ) from error

from t0 import data
from t0._ops import register
from t0.config import T0Config
from t0.data import MaskType, TimeSeries, VariateType
from t0.forecast import Forecast
from t0.mlx._ops import MLXOps
from t0.mlx.model import T0Forecaster

_OPS = MLXOps()
register(mlx.core.array, _OPS)


def batch_series(series):
    """``t0.data.batch_series`` into MLX arrays, from MLX arrays, NumPy arrays or lists; see there."""
    return data.batch_series(series, ops=_OPS)


__all__ = ["Forecast", "MaskType", "T0Config", "T0Forecaster", "TimeSeries", "VariateType", "batch_series"]
