"""``ArrayOps`` on native MLX calls."""

from typing import Any

import mlx.core as mx


class MLXOps:
    """The shared array operations for the ``mx.array`` backend."""

    float32, int8, int32, int64 = mx.float32, mx.int8, mx.int32, mx.int64

    def asarray(self, obj: Any, /, *, dtype: Any = None, device: Any = None) -> mx.array:
        return mx.array(obj, dtype=dtype)

    def full(self, shape: tuple[int, ...], fill_value: Any, *, dtype: Any = None, device: Any = None) -> mx.array:
        return mx.full(shape, fill_value, dtype=dtype)

    def arange(self, start: int, /, *, dtype: Any = None, device: Any = None) -> mx.array:
        return mx.arange(start) if dtype is None else mx.arange(start, dtype=dtype)

    def zeros_like(self, x: mx.array, /) -> mx.array:
        return mx.zeros_like(x)

    def ones_like(self, x: mx.array, /) -> mx.array:
        return mx.ones_like(x)

    def astype(self, x: mx.array, dtype: Any, /) -> mx.array:
        return x.astype(dtype)

    def reshape(self, x: mx.array, /, shape: tuple[int, ...]) -> mx.array:
        return x.reshape(shape)

    def expand_dims(self, x: mx.array, /, *, axis: int) -> mx.array:
        return mx.expand_dims(x, axis)

    def squeeze(self, x: mx.array, /, axis: int) -> mx.array:
        return mx.squeeze(x, axis)

    def permute_dims(self, x: mx.array, /, axes: tuple[int, ...]) -> mx.array:
        return mx.transpose(x, axes)

    def broadcast_to(self, x: mx.array, /, shape: tuple[int, ...]) -> mx.array:
        return mx.broadcast_to(x, shape)

    def concat(self, arrays: list[mx.array], /, *, axis: int = 0) -> mx.array:
        return mx.concatenate(arrays, axis=axis)

    def repeat(self, x: mx.array, repeats: int, /, *, axis: int | None = None) -> mx.array:
        return mx.repeat(x, repeats, axis=axis)

    def take(self, x: mx.array, indices: mx.array, /, *, axis: int | None = None) -> mx.array:
        return mx.take(x, indices, axis=axis)

    def take_along_axis(self, x: mx.array, indices: mx.array, /, *, axis: int = -1) -> mx.array:
        return mx.take_along_axis(x, indices, axis=axis)

    def where(self, condition: mx.array, x1: Any, x2: Any, /) -> mx.array:
        return mx.where(condition, x1, x2)

    def clip(self, x: mx.array, /, min: Any = None, max: Any = None) -> mx.array:
        return mx.clip(x, min, max)

    def minimum(self, x1: mx.array, x2: mx.array, /) -> mx.array:
        return mx.minimum(x1, x2)

    def sum(self, x: mx.array, /, *, axis: int | None = None, keepdims: bool = False) -> mx.array:
        return mx.sum(x, axis=axis, keepdims=keepdims)

    def any(self, x: mx.array, /, *, axis: int | None = None, keepdims: bool = False) -> mx.array:
        return mx.any(x, axis=axis, keepdims=keepdims)

    def cumulative_sum(self, x: mx.array, /, *, axis: int | None = None) -> mx.array:
        return mx.cumsum(x, axis=axis)

    def argmax(self, x: mx.array, /, *, axis: int | None = None, keepdims: bool = False) -> mx.array:
        return mx.argmax(x, axis=axis, keepdims=keepdims)

    def argsort(self, x: mx.array, /, *, axis: int = -1) -> mx.array:
        return mx.argsort(x, axis=axis)

    def isnan(self, x: mx.array, /) -> mx.array:
        return mx.isnan(x)

    def isfinite(self, x: mx.array, /) -> mx.array:
        return mx.isfinite(x)

    def sqrt(self, x: mx.array, /) -> mx.array:
        return mx.sqrt(x)

    def asinh(self, x: mx.array, /) -> mx.array:
        return mx.arcsinh(x)

    def sinh(self, x: mx.array, /) -> mx.array:
        return mx.sinh(x)

    def device(self, x: mx.array, /) -> None:
        return None  # unified memory: arrays have no device

    def to_device(self, x: mx.array, device: Any, /) -> mx.array:
        return x  # unified memory: arrays have no device

    def row_sum(self, x: mx.array, /) -> mx.array:
        return mx.sum(x, axis=-1, keepdims=True)
