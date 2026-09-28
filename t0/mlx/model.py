# Copyright 2026 The Forecasting Company
# The architecture follows the first-party t0 PyTorch implementation, which is
# inspired by Datadog's Toto patched-transformer backbone
# (https://github.com/DataDog/toto). The autoregressive quantile rollout follows
# Chronos-2's inference approach
# (https://github.com/amazon-science/chronos-forecasting).
# Copyright 2025 Datadog, Inc.
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""MLX-native t0-alpha model and inference API."""

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import mlx.core as mx
import mlx.nn as nn
from huggingface_hub import ModelHubMixin, hf_hub_download
from huggingface_hub.constants import SAFETENSORS_SINGLE_FILE

from t0.config import ScalerEpsMode, T0Config
from t0.data import TimeSeries
from t0.forecaster import DEFAULT_MAX_HORIZON, ForecasterBase
from t0.mlx.layers import PatchEncoder, QuantileHead, ResidualBlock, Transformer

__all__ = ["DEFAULT_MAX_HORIZON", "T0Forecaster"]


class T0Forecaster(
    ForecasterBase,
    # Before nn.Module: mlx.nn.Module is a dict, whose __new__ would shadow the mixin's,
    # and the mixin's __new__ is what records the constructor arguments for config.json.
    ModelHubMixin,
    nn.Module,
    library_name="tfc-t0",
    repo_url="https://github.com/theforecastingcompany/tfc-t0",
    pipeline_tag="time-series-forecasting",
    license="apache-2.0",
    tags=["time-series", "forecasting", "foundation-models", "pretrained-models", "safetensors", "mlx"],
):
    """Open-weights t0-alpha forecasting backbone for MLX inference.

    Construct with explicit hyperparameters, or via ``from_config`` /
    ``from_pretrained``. ``embed_dim`` must be divisible by ``num_heads`` and
    ``group_every_n`` must divide ``num_layers``. ``predict`` is shared with the
    PyTorch runtime; ``compile`` traces its forward passes for repeated shapes.
    """

    def __init__(
        self,
        embed_dim: int,
        num_layers: int,
        num_heads: int,
        mlp_hidden_dim: int,
        patch_size: int,
        group_every_n: int,
        dropout: float,
        quantile_levels: Sequence[float],
        scaler_use_arcsinh: bool = True,
        scaler_eps: float = 0.1,
        scaler_eps_mode: ScalerEpsMode = "variance_offset",
        **_: Any,
    ):
        super().__init__()
        # Building the config validates every hyperparameter.
        self._init_shared(
            T0Config(
                embed_dim=embed_dim,
                num_layers=num_layers,
                num_heads=num_heads,
                mlp_hidden_dim=mlp_hidden_dim,
                patch_size=patch_size,
                group_every_n=group_every_n,
                dropout=dropout,
                quantile_levels=tuple(quantile_levels),
                scaler_use_arcsinh=scaler_use_arcsinh,
                scaler_eps=scaler_eps,
                scaler_eps_mode=scaler_eps_mode,
            )
        )
        self.head = QuantileHead(quantile_levels)
        self.patch_encoder = PatchEncoder(embed_dim, patch_size)
        self.transformer = Transformer(
            num_layers,
            embed_dim,
            num_heads,
            mlp_hidden_dim,
            dropout,
            group_every_n,
        )
        self.decoder = ResidualBlock(embed_dim, embed_dim, patch_size * self.head.n_quantiles)
        self._compiled_single_pass = None

    def __setattr__(self, name: str, value: Any) -> None:
        # mlx.nn.Module files dict values into its parameter tree; the hub mixin's captured
        # config (a dict) is bookkeeping, not a parameter.
        if name.startswith("_hub_mixin"):
            object.__setattr__(self, name, value)
        else:
            super().__setattr__(name, value)

    @classmethod
    def _from_pretrained(
        cls,
        *,
        model_id: str,
        revision: str | None,
        cache_dir: str | Path | None,
        force_download: bool,
        local_files_only: bool,
        token: str | bool | None,
        **model_kwargs: Any,
    ) -> "T0Forecaster":
        """Build from ``config.json`` (read by the mixin into ``model_kwargs``) and load the safetensors weights.

        The published PyTorch tensor layouts are already compatible with MLX, so loading is
        strict and does not transpose or numerically convert any model weight.
        """
        directory = Path(model_id).expanduser()
        if directory.is_dir():
            weights_path = directory / SAFETENSORS_SINGLE_FILE
        else:
            weights_path = Path(
                hf_hub_download(
                    repo_id=model_id,
                    filename=SAFETENSORS_SINGLE_FILE,
                    revision=revision,
                    cache_dir=cache_dir,
                    force_download=force_download,
                    token=token,
                    local_files_only=local_files_only,
                )
            )
        if not weights_path.is_file():
            raise FileNotFoundError(f"{model_id} must contain {SAFETENSORS_SINGLE_FILE}")
        model = cls(**model_kwargs)
        model.load_weights(str(weights_path), strict=True)
        model.eval()
        mx.eval(model.parameters())
        return model

    def _save_pretrained(self, save_directory: Path) -> None:
        """Write the weights; the mixin writes ``config.json`` from the constructor arguments."""
        self.save_weights(str(save_directory / SAFETENSORS_SINGLE_FILE))

    def __call__(self, model_input: TimeSeries) -> mx.array:
        """Return per-patch values on the model's native quantile grid."""
        # Runs inside the compiled single pass, so the layout comes from shapes, not data: the
        # whole width is read as context (the rollout's windows are aligned already; an
        # unaligned input is left-padded).
        return self._backbone(model_input, context_end=model_input.seq_len)

    def compile(self, *, shapeless: bool = False) -> "T0Forecaster":
        """Compile the scaling, model, and rescaling graph for repeated inference."""
        self._compiled_single_pass = mx.compile(
            self._single_pass_arrays,
            inputs=self.state,
            shapeless=shapeless,
        )
        return self

    def uncompile(self) -> "T0Forecaster":
        """Return to eager MLX execution."""
        self._compiled_single_pass = None
        return self

    def _single_pass(self, window: TimeSeries) -> mx.array:
        if self._compiled_single_pass is None:
            return super()._single_pass(window)
        return self._compiled_single_pass(window.variates, window.mask, window.group_ids, window.variate_type)

    def _single_pass_arrays(
        self,
        variates: mx.array,
        mask: mx.array,
        group_ids: mx.array,
        variate_type: mx.array,
    ) -> mx.array:
        # mx.compile traces array arguments, so the window crosses the boundary as its four fields.
        return ForecasterBase._single_pass(self, TimeSeries(variates, mask, group_ids, variate_type))

    def _as_array(self, value: Any, *, floating: bool) -> mx.array:
        return mx.array(value, dtype=mx.float32) if floating else mx.array(value)

    def _device(self) -> None:
        return None  # unified memory: arrays have no device

    def _materialize(self, value: mx.array) -> mx.array:
        mx.eval(value)
        return value
