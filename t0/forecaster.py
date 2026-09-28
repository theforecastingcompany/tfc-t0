"""What every runtime's ``T0Forecaster`` does the same way: the forward pass wiring and ``predict``.

A runtime's forecaster mixes in ``ForecasterBase`` beside its framework's module class,
builds its parametric layers (``patch_encoder``, ``transformer``, ``decoder``, ``head``),
and supplies the few hooks that differ between frameworks: array conversion, the device,
the scope ``predict`` runs in, and materializing lazy results.
"""

import contextlib
import dataclasses
import sys
from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Any

from t0._ops import ops_for
from t0.config import T0Config
from t0.data import TimeSeries
from t0.forecast import Forecast
from t0.levels import get_rollout_quantile_levels, validate_quantile_levels
from t0.patcher import Patcher
from t0.quantile import extrapolate_quantiles, sanitize_predictions
from t0.rollout import rollout
from t0.scaler import CausalScaler
from t0.types import VariateType

if sys.version_info >= (3, 11):
    from typing import Self
else:
    from typing_extensions import Self

__all__ = ["DEFAULT_MAX_HORIZON", "ForecasterBase"]

# Steps predicted per forward pass; longer horizons continue with the rollout.
DEFAULT_MAX_HORIZON = 1024


class ForecasterBase(ABC):
    """The runtime-independent half of ``T0Forecaster``."""

    config: T0Config
    patch_size: int
    max_horizon: int
    patcher: Patcher
    scaler: CausalScaler
    # Parametric layers, built by each runtime with its own framework.
    patch_encoder: Any
    transformer: Any
    decoder: Any
    head: Any

    @classmethod
    def from_config(cls, config: T0Config) -> Self:
        """Build a fresh, randomly initialized model from a config."""
        return cls(**dataclasses.asdict(config))

    def _init_shared(self, config: T0Config) -> None:
        """Set the config and the stateless components; call once the framework module is initialized."""
        # config.json is the single serialized source of truth.
        self.config = config
        self.patch_size = config.patch_size
        # Not in the serialized config — callers may tune it (keep a multiple of patch_size).
        self.max_horizon = DEFAULT_MAX_HORIZON
        self.patcher = Patcher(config.patch_size)
        self.scaler = CausalScaler(
            use_arcsinh=config.scaler_use_arcsinh, eps=config.scaler_eps, eps_mode=config.scaler_eps_mode
        )

    @abstractmethod
    def _as_array(self, value: Any, *, floating: bool) -> Any:
        """``value`` as the runtime's array, on the model's device; float32 when ``floating``."""

    @abstractmethod
    def _device(self) -> Any:
        """Where the model's arrays live; ``None`` where arrays have no device."""

    def _predict_scope(self) -> contextlib.AbstractContextManager:
        """The context the rollout's forward passes run in (e.g. mixed precision)."""
        return contextlib.nullcontext()

    def _materialize(self, value: Any) -> Any:
        """Force lazily evaluated results before returning them."""
        return value

    def _backbone(self, model_input: TimeSeries, context_end: int | None = None) -> Any:
        """Predict per-patch quantiles for every patch position (``[variates, patches, patch_size, quantiles]``)."""
        padded = self.patcher.pad(model_input, context_end)
        values, mask, variate_type, group_ids = (
            self.patcher.patch(field) for field in (padded.variates, padded.mask, padded.variate_type, padded.group_ids)
        )
        embeddings = self.patch_encoder(values, mask, variate_type)
        embeddings = self.transformer(embeddings, group_ids, variate_type, mask)
        decoded = self.decoder(embeddings)
        decoded = ops_for(decoded).reshape(decoded, (*decoded.shape[:-1], self.patch_size, self.head.n_quantiles))
        return self.head(decoded)

    def _single_pass(self, window: TimeSeries) -> Any:
        """Scale a window, run the model, and rescale its per-patch predictions to data space."""
        scaled, loc_scale = self.scaler.scale_input(window)
        return self.scaler.rescale_predictions(self(scaled), loc_scale, self.patch_size)  # ty: ignore[call-non-callable]

    def _predict_step(self, window: TimeSeries, horizon: int) -> Any:
        """One pass over a window; the ``horizon`` steps after its context end, ``[rows, horizon, quantiles]``."""
        predictions = self._single_pass(window)
        context_patches = (window.seq_len - horizon) // self.patch_size
        predictions = predictions[:, context_patches - 1 : context_patches - 1 + horizon // self.patch_size]
        return ops_for(predictions).reshape(predictions, (predictions.shape[0], horizon, self.head.n_quantiles))

    def predict(
        self,
        model_input: Any,
        horizon: int,
        quantile_levels: Sequence[float] = (0.1, 0.5, 0.9),
        *,
        context_length: int | None = None,
        future_covariates: Any = None,
        mask: Any = None,
        group_ids: Any = None,
    ) -> Forecast:
        """Forecast ``horizon`` future timesteps, continuing auto-regressively past ``max_horizon``.

        Args:
            model_input: A ``TimeSeries``, or past observations as ``[T]`` / ``[B, T]``
                (independent univariate series) or ``[B, V, T]`` (multivariate, jointly
                forecast), as NumPy arrays or the runtime's arrays. An array is converted with
                ``TimeSeries.from_array``. NaN marks a missing observation unless ``mask``
                says otherwise.
            horizon: Number of future timesteps to forecast.
            quantile_levels: Quantile levels to return, sorted ascending in ``(0, 1)``; levels
                between trained ones are interpolated, levels beyond them extrapolated on
                exponential tails.
            context_length: Width of the right-aligned historical context. Defaults to the
                start of the forecast region.
            future_covariates: Array inputs only. ``[B, F, T + horizon]`` covariates known over
                the context and horizon (e.g. calendar features); conditioned on but not
                forecast. NaN over the horizon is 0.
            mask: Array inputs only. ``MaskType`` values shaped like the context: ``MISSING``
                for an absent observation, ``PAD`` for a cell that only pads a shorter series
                out to the batch's width. Defaults to reading every NaN as missing. Only
                all-``PAD`` patches are left unattended.
            group_ids: Array inputs only. One id per context row, marking which rows are
                variates of the same series; rows sharing an id are forecast jointly. Defaults
                to one series per sample. Cannot be combined with ``future_covariates``.

        Returns:
            Quantiles shaped ``[B, horizon, Q]`` or ``[B, V, horizon, Q]`` for array inputs,
            and ``[target_rows, horizon, Q]`` -- the input's row order -- for a ``TimeSeries``.
            Always float32, finite, and an array of the runtime (on the model's device).

        Raises:
            ValueError: ``horizon`` or ``quantile_levels`` are out of range, ``context_length`` does
                not fit the input width, the input has no target row, or an array-only
                argument is passed alongside a ``TimeSeries``.
        """
        if horizon < 1:
            raise ValueError(f"horizon must be >= 1, got {horizon}")
        quantile_levels = validate_quantile_levels(quantile_levels)

        batch_shape = None
        if isinstance(model_input, TimeSeries):
            for name, value in (("future_covariates", future_covariates), ("mask", mask), ("group_ids", group_ids)):
                if value is not None:
                    raise ValueError(f"{name} applies to array inputs; build it into the TimeSeries instead")
            series = model_input.to(self._device())
            # Ids and roles index embeddings and seed per-path group ids (id * n_paths + path),
            # so a narrow integer dtype would overflow: widen them once, here.
            ops = ops_for(series.variates)
            series = dataclasses.replace(
                series,
                group_ids=ops.astype(series.group_ids, ops.int64),
                variate_type=ops.astype(series.variate_type, ops.int64),
            )
            # One host read of the row roles serves every check below and the rollout.
            row_types = [int(kind) for kind in series.variate_type[:, -1].tolist()]
            target_rows = [row for row, kind in enumerate(row_types) if kind == VariateType.TARGET]
            has_future = VariateType.FUTURE in row_types
            if context_length is None:
                context_length = self.patcher.context_end(series)
        else:
            series, batch_shape, n_targets, context_width = self._series_from_array(
                model_input, horizon, future_covariates, mask, group_ids
            )
            # from_array puts the targets first and the forecast region after the context.
            target_rows = list(range(n_targets))
            has_future = series.variates.shape[0] > n_targets
            if context_length is None:
                context_length = context_width

        if not 1 <= context_length <= series.seq_len:
            raise ValueError(
                f"context_length must be between 1 and the input width ({series.seq_len}), got {context_length}"
            )
        if not target_rows:
            raise ValueError("model_input must contain at least one target row")
        # An input either stops at the context or carries the forecast region. Known future
        # covariates must span that region, so they rule the context-only width out.
        widths = [context_length + horizon]
        if not has_future:
            widths.append(context_length)
        if series.seq_len not in widths:
            raise ValueError(
                f"model_input width must be {' or '.join(str(w) for w in sorted(widths))} "
                f"for context_length={context_length} and horizon={horizon}, got {series.seq_len}"
            )

        rollout_levels = get_rollout_quantile_levels(self.config.quantile_levels, quantile_levels)
        with self._predict_scope():
            predictions = rollout(
                series,
                self._predict_step,
                horizon,
                context_length,
                self.patch_size,
                self.max_horizon,
                self.config.quantile_levels,
                rollout_levels,
                target_rows,
            )
        predictions = sanitize_predictions(predictions)
        predictions = extrapolate_quantiles(quantile_levels, rollout_levels, predictions, self.config.quantile_levels)
        if batch_shape is not None:
            predictions = ops_for(predictions).reshape(predictions, (*batch_shape, *predictions.shape[1:]))
        return Forecast(quantiles=self._materialize(predictions), quantile_levels=tuple(quantile_levels))

    def _series_from_array(
        self, context: Any, horizon: int, future_covariates: Any, mask: Any, group_ids: Any
    ) -> tuple[TimeSeries, tuple[int, int] | None, int, int]:
        """A ``TimeSeries`` from raw arrays, with the batch shape to restore, the target count and the context width."""
        context = self._as_array(context, floating=True)
        ops = ops_for(context)
        if context.ndim == 1:
            context = ops.expand_dims(context, axis=0)
        if context.ndim not in (2, 3):
            raise ValueError(f"context must be [T], [B, T] or [B, V, T]; got shape {tuple(context.shape)}")

        future = None
        if future_covariates is not None:
            future = self._as_array(future_covariates, floating=True)
            expected_width = context.shape[-1] + horizon
            if future.ndim != 3 or future.shape[0] != context.shape[0] or future.shape[2] != expected_width:
                raise ValueError(
                    f"future_covariates must be [B={context.shape[0]}, F, T+horizon={expected_width}]; "
                    f"got shape {tuple(future.shape)}"
                )
        groups = None if group_ids is None else self._as_array(group_ids, floating=False)
        mask_array = None
        if mask is not None:
            mask_array = self._as_array(mask, floating=False)
            # Mirror the 1-D promotion applied to `context` so the shapes line up.
            if mask_array.ndim == 1:
                mask_array = ops.expand_dims(mask_array, axis=0)

        batch_shape = (context.shape[0], context.shape[1]) if context.ndim == 3 else None
        n_targets = context.shape[0] * (context.shape[1] if context.ndim == 3 else 1)
        series = TimeSeries.from_array(context, future, mask=mask_array, group_ids=groups)
        return series, batch_shape, n_targets, context.shape[-1]
