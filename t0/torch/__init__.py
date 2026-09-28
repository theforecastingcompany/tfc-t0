"""The PyTorch runtime of t0. Requires the ``torch`` extra: ``pip install 'tfc-t0[torch]'``."""

try:
    import torch
except ImportError as error:
    raise ImportError(
        "t0.torch is the PyTorch runtime of t0. Install through pip install 'tfc-t0[torch]' "
        "or use the MLX runtime with `from t0.mlx import T0Forecaster`"
    ) from error

from t0 import data
from t0._ops import register
from t0.data import MaskType, TimeSeries, VariateType
from t0.torch._ops import TorchOps
from t0.torch.model import Forecast, T0Forecaster

_OPS = TorchOps()
register(torch.Tensor, _OPS)


def batch_series(series):
    """``t0.data.batch_series`` into tensors, from tensors, NumPy arrays or lists; see there."""
    return data.batch_series(series, ops=_OPS)


__all__ = ["Forecast", "MaskType", "T0Forecaster", "TimeSeries", "VariateType", "batch_series"]
