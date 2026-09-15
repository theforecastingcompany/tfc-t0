# Copyright 2026 The Forecasting Company
# The layer structure follows the first-party t0 PyTorch implementation. Its
# patched-transformer and rotary backbone is adapted from Datadog's Toto
# (https://github.com/DataDog/toto); the patch encoding and variate-attention
# patterns build on Chronos-2
# (https://github.com/amazon-science/chronos-forecasting).
# Copyright 2025 Datadog, Inc.
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""MLX-native leaf layers used by t0-alpha."""

import math
from collections.abc import Callable, Sequence

import mlx.core as mx
import mlx.nn as nn

from t0_mlx.data import MaskType, TimeSeries, VariateType


class MLP(nn.Module):
    """Two-layer perceptron with an activation between its projections."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        dropout: float = 0.0,
        activation: Callable[[mx.array], mx.array] = nn.relu,
    ):
        super().__init__()
        self.hidden_layer = nn.Linear(input_size, hidden_size)
        self.output_layer = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(dropout)
        self.activation = activation

    def __call__(self, x: mx.array) -> mx.array:
        return self.output_layer(self.dropout(self.activation(self.hidden_layer(x))))


class ResidualBlock(nn.Module):
    """MLP summed with a learned projection of its input."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        dropout: float = 0.0,
        activation: Callable[[mx.array], mx.array] = nn.relu,
    ):
        super().__init__()
        self.mlp = MLP(input_size, hidden_size, output_size, dropout, activation)
        self.residual_layer = nn.Linear(input_size, output_size)

    def __call__(self, x: mx.array) -> mx.array:
        return self.mlp(x) + self.residual_layer(x)


class PatchEncoder(nn.Module):
    """Encode values, within-patch time, validity, and variate role."""

    def __init__(self, embed_dim: int, patch_size: int):
        super().__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.projection = ResidualBlock(patch_size * 3, embed_dim, embed_dim)
        self.type_embeddings = nn.Embedding(3, embed_dim)

    def __call__(self, values: mx.array, mask: mx.array, variate_type: mx.array) -> mx.array:
        if values.ndim != 3 or values.shape[-1] != self.patch_size:
            raise ValueError(f"values must be [variates, patches, {self.patch_size}]")
        if mask.shape != values.shape or variate_type.shape != values.shape:
            raise ValueError("mask and variate_type must match values")
        total_variates, n_patches, _ = values.shape
        validity = (mask == 0).astype(values.dtype)
        time_index = mx.arange(self.patch_size, dtype=values.dtype) / self.patch_size
        time_index = mx.broadcast_to(time_index, (total_variates, n_patches, self.patch_size))
        embedded = self.projection(mx.concatenate([values, time_index, validity], axis=-1))
        type_ids = mx.maximum(variate_type[:, :, 0], mx.array(0, dtype=variate_type.dtype))
        return embedded + self.type_embeddings(type_ids)


class RMSNorm(nn.Module):
    """RMS normalization with the checkpoint's 1e-8 epsilon."""

    def __init__(self, dims: int, eps: float = 1e-8):
        super().__init__()
        self.scale = mx.ones((dims,))
        self.eps = eps

    def __call__(self, x: mx.array) -> mx.array:
        return x * mx.rsqrt(mx.mean(mx.square(x), axis=-1, keepdims=True) + self.eps) * self.scale


class SwiGLU(nn.Module):
    """SwiGLU with the checkpoint's gate-first projection ordering."""

    def __call__(self, x: mx.array) -> mx.array:
        gate, values = mx.split(x, 2, axis=-1)
        return nn.silu(gate) * values


class QuantileHead(nn.Module):
    """Turn raw decoder outputs into non-decreasing quantiles."""

    def __init__(self, quantile_levels: Sequence[float]):
        super().__init__()
        self.quantile_levels = mx.array(sorted(quantile_levels), dtype=mx.float32)

    @property
    def n_quantiles(self) -> int:
        return self.quantile_levels.size

    def __call__(self, x: mx.array) -> mx.array:
        first = x[..., :1]
        if x.shape[-1] == 1:
            return first
        remaining = first + mx.cumsum(nn.softplus(x[..., 1:]), axis=-1)
        return mx.concatenate([first, remaining], axis=-1)


class Patcher(nn.Module):
    """Left-pad and reshape time steps into contiguous patches."""

    def __init__(self, patch_size: int):
        super().__init__()
        self.patch_size = patch_size

    def pad(self, model_input: TimeSeries) -> TimeSeries:
        pad_len = (-model_input.seq_len) % self.patch_size
        if pad_len == 0:
            return model_input
        rows = model_input.variates.shape[0]
        return TimeSeries(
            variates=mx.concatenate(
                [mx.zeros((rows, pad_len), dtype=model_input.variates.dtype), model_input.variates], axis=1
            ),
            mask=mx.concatenate([mx.full((rows, pad_len), MaskType.PAD, dtype=mx.int8), model_input.mask], axis=1),
            group_ids=mx.concatenate(
                [mx.full((rows, pad_len), -1, dtype=model_input.group_ids.dtype), model_input.group_ids], axis=1
            ),
            variate_type=mx.concatenate(
                [mx.full((rows, pad_len), -1, dtype=model_input.variate_type.dtype), model_input.variate_type], axis=1
            ),
        )

    def patch(self, values: mx.array) -> mx.array:
        return values.reshape(*values.shape[:-1], -1, self.patch_size)


def _rotate_half(x: mx.array) -> mx.array:
    paired = x.reshape(*x.shape[:-1], -1, 2)
    return mx.stack([-paired[..., 1], paired[..., 0]], axis=-1).reshape(x.shape)


class TimeAwareRotaryEmbedding:
    """The checkpoint's temporal RoPE with XPos query/key scaling."""

    def __init__(self, dims: int, theta: float = 10_000.0, scale_base: float = 512.0):
        self.dims = dims
        self.theta = theta
        self.scale_base = scale_base

    def rotate_queries_and_keys(self, queries: mx.array, keys: mx.array) -> tuple[mx.array, mx.array]:
        seq_len = queries.shape[-2]
        positions = mx.arange(seq_len, dtype=queries.dtype)
        frequencies = 1.0 / (self.theta ** (mx.arange(0, self.dims, 2, dtype=mx.float32) / self.dims))
        angles = mx.repeat(positions[:, None] * frequencies[None, :], 2, axis=-1)
        cosine = mx.cos(angles)
        sine = mx.sin(angles)

        scale_frequencies = (mx.arange(0, self.dims, 2, dtype=mx.float32) + 0.4 * self.dims) / (1.4 * self.dims)
        center = mx.floor(mx.max(positions) / 2.0)
        power = (positions - center) / self.scale_base
        half_scale = scale_frequencies[None, :] ** power[:, None]
        scale = mx.concatenate([half_scale, half_scale], axis=-1).astype(queries.dtype)

        rotated_queries = (queries * cosine + _rotate_half(queries) * sine) * scale
        rotated_keys = (keys * cosine + _rotate_half(keys) * sine) / scale
        return rotated_queries.astype(queries.dtype), rotated_keys.astype(keys.dtype)


class SelfAttention(nn.Module):
    """Shared QKV projections for temporal and variate-axis attention."""

    def __init__(self, embed_dim: int, num_heads: int, rotary: TimeAwareRotaryEmbedding | None = None):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.rotary = rotary
        self.wQKV = nn.Linear(embed_dim, embed_dim * 3)
        self.wO = nn.Linear(embed_dim, embed_dim)
        self.q_norm = RMSNorm(self.head_dim)
        self.k_norm = RMSNorm(self.head_dim)

    def __call__(self, x: mx.array, attn_mask: mx.array) -> mx.array:
        batch, seq_len, _ = x.shape
        qkv = self.wQKV(x).reshape(batch, seq_len, 3, self.num_heads, self.head_dim)
        qkv = qkv.transpose(2, 0, 3, 1, 4)
        queries, keys, values = qkv[0], qkv[1], qkv[2]
        queries, keys = self.q_norm(queries), self.k_norm(keys)
        if self.rotary is not None:
            queries, keys = self.rotary.rotate_queries_and_keys(queries, keys)
        attended = mx.fast.scaled_dot_product_attention(
            queries,
            keys,
            values,
            scale=1.0 / math.sqrt(self.head_dim),
            mask=attn_mask,
        )
        return self.wO(attended.transpose(0, 2, 1, 3).reshape(batch, seq_len, self.embed_dim))


class TimeSelfAttentionBlock(nn.Module):
    """Pre-norm time-axis attention with a residual connection."""

    def __init__(self, embed_dim: int, num_heads: int, dropout: float, rotary: TimeAwareRotaryEmbedding):
        super().__init__()
        self.norm = RMSNorm(embed_dim)
        self.attention = SelfAttention(embed_dim, num_heads, rotary)
        self.dropout = nn.Dropout(dropout)

    def __call__(self, x: mx.array, attn_mask: mx.array) -> mx.array:
        return x + self.dropout(self.attention(self.norm(x), attn_mask))


class VariateSelfAttentionBlock(nn.Module):
    """Pre-norm variate-axis attention with a residual connection."""

    def __init__(self, embed_dim: int, num_heads: int, dropout: float):
        super().__init__()
        self.norm = RMSNorm(embed_dim)
        self.attention = SelfAttention(embed_dim, num_heads)
        self.dropout = nn.Dropout(dropout)

    def __call__(self, x: mx.array, attn_mask: mx.array) -> mx.array:
        flipped = x.transpose(1, 0, 2)
        attended = self.attention(self.norm(flipped), attn_mask)
        return x + self.dropout(attended).transpose(1, 0, 2)


class TransformerLayer(nn.Module):
    """One attention block followed by a pre-norm SwiGLU feed-forward."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        mlp_hidden_dim: int,
        dropout: float,
        is_group: bool,
        rotary: TimeAwareRotaryEmbedding,
    ):
        super().__init__()
        self.is_group = is_group
        if is_group:
            self.attention_block = VariateSelfAttentionBlock(embed_dim, num_heads, dropout)
        else:
            self.attention_block = TimeSelfAttentionBlock(embed_dim, num_heads, dropout, rotary)
        self.norm = RMSNorm(embed_dim)
        self.mlp = [
            nn.Linear(embed_dim, 2 * mlp_hidden_dim),
            SwiGLU(),
            nn.Linear(mlp_hidden_dim, embed_dim),
            nn.Dropout(dropout),
        ]

    def __call__(self, x: mx.array, time_mask: mx.array, group_mask: mx.array) -> mx.array:
        x = self.attention_block(x, group_mask if self.is_group else time_mask)
        residual = self.mlp[0](self.norm(x))
        residual = self.mlp[1](residual)
        residual = self.mlp[2](residual)
        return x + self.mlp[3](residual)


def _reduce_patch_metadata(metadata: mx.array, patched_mask: mx.array) -> mx.array:
    real = patched_mask != MaskType.PAD
    first_real = mx.argmax(real.astype(mx.int32), axis=-1, keepdims=True)
    reduced = mx.take_along_axis(metadata, first_real, axis=-1).squeeze(-1)
    return mx.where(mx.any(real, axis=-1), reduced, mx.array(-1, dtype=metadata.dtype))


def _build_attention_masks(
    patched_group_ids: mx.array, patched_variate_type: mx.array, patched_mask: mx.array
) -> tuple[mx.array, mx.array]:
    patch_group_ids = _reduce_patch_metadata(patched_group_ids, patched_mask)
    patch_variate_type = _reduce_patch_metadata(patched_variate_type, patched_mask)
    attendable = mx.any(patched_mask != MaskType.PAD, axis=-1)

    valid = patch_group_ids >= 0
    same_time_document = (
        (patch_group_ids[:, :, None] == patch_group_ids[:, None, :]) & valid[:, :, None] & valid[:, None, :]
    )
    patches = patch_group_ids.shape[1]
    causal = mx.tril(mx.ones((patches, patches), dtype=mx.bool_), k=0)
    time_mask = same_time_document & causal[None, :, :]
    future_query = patch_variate_type == VariateType.FUTURE
    time_mask = mx.where(future_query[:, :, None], same_time_document, time_mask)
    time_mask = time_mask & attendable[:, None, :]

    ids_by_patch = patch_group_ids.transpose(1, 0)
    valid_by_patch = valid.transpose(1, 0)
    group_mask = (
        (ids_by_patch[:, :, None] == ids_by_patch[:, None, :]) & valid_by_patch[:, :, None] & valid_by_patch[:, None, :]
    )
    return time_mask[:, None, :, :], group_mask[:, None, :, :]


class Transformer(nn.Module):
    """Alternating time- and variate-attention stack."""

    def __init__(
        self,
        num_layers: int,
        embed_dim: int,
        num_heads: int,
        mlp_hidden_dim: int,
        dropout: float,
        group_every_n: int,
    ):
        super().__init__()
        rotary = TimeAwareRotaryEmbedding(embed_dim // num_heads)
        self.layers = [
            TransformerLayer(
                embed_dim,
                num_heads,
                mlp_hidden_dim,
                dropout,
                is_group=group_every_n > 0 and (index + 1) % group_every_n == 0,
                rotary=rotary,
            )
            for index in range(num_layers)
        ]
        self.out_norm = RMSNorm(embed_dim)

    def __call__(
        self,
        x: mx.array,
        patched_group_ids: mx.array,
        patched_variate_type: mx.array,
        patched_mask: mx.array,
    ) -> mx.array:
        time_mask, group_mask = _build_attention_masks(patched_group_ids, patched_variate_type, patched_mask)
        for layer in self.layers:
            x = layer(x, time_mask, group_mask)
        return self.out_norm(x)
