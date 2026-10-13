"""GluonTS-style predictor wrapping ``T0Forecaster`` for benchmark evaluation.

Designed for benchmarks built on GluonTS test splits (e.g. GIFT-Eval):
``predict`` consumes test-data input entries (dicts with ``target``,
``start``, and optionally ``item_id``) and returns one ``QuantileForecast``
per entry, ready for ``gluonts.model.evaluate_model``.

Requires the ``evaluation`` extra: ``pip install "tfc-t0[evaluation]"``.
"""

import logging
from collections.abc import Iterable, Sequence
from typing import TypedDict

import numpy as np
import torch

try:
    from gluonts.model.forecast import QuantileForecast
    from tqdm.auto import tqdm
except ImportError as err:  # pragma: no cover - exercised only without the extra
    raise ImportError('t0.evaluation requires the evaluation extra: pip install "tfc-t0[evaluation]"') from err

from t0.torch import batch_series
from t0.torch.model import T0Forecaster

logger = logging.getLogger(__name__)


class InferenceKwargs(TypedDict):
    """The inference adjustments ``T0Predictor`` passes to every ``predict`` call."""

    trim_leading_missing: bool
    symmetric_averaging: bool
    clip_nonnegative: bool
    snap_integers: bool


class T0Predictor:
    """Batched ``T0Forecaster`` predictor over GluonTS test-data entries.

    Args:
        model: A loaded ``T0Forecaster`` (already on its target device).
        prediction_length: Forecast horizon in timesteps.
        quantile_levels: Quantile levels to forecast. Defaults to the
            levels use for training the model. Other levels are interpolated or, beyond the
            trained range, extrapolated.
        context_length: If set, keep only the most recent ``context_length``
            observations of each series.
        batch_size: Entries per forward pass.
        show_progress: Display a progress bar over the entries while
            predicting.
        trim_leading_missing, symmetric_averaging, clip_nonnegative, snap_integers: Passed to
            every ``predict`` call (see ``T0Forecaster.predict``). On by default, the
            settings of the published benchmark scores.
        cross_inference: Forecast up to this many series of one test window together,
            as the variates of one group, so the model reads across them. ``0`` forecasts
            each series on its own. Two windows of the same item are never grouped.
    """

    def __init__(
        self,
        model: T0Forecaster,
        prediction_length: int,
        *,
        quantile_levels: Sequence[float] | None = None,
        context_length: int | None = None,
        batch_size: int = 64,
        show_progress: bool = False,
        trim_leading_missing: bool = True,
        symmetric_averaging: bool = True,
        clip_nonnegative: bool = True,
        snap_integers: bool = True,
        cross_inference: int = 0,
    ):
        self.model = model
        self.prediction_length = prediction_length
        if quantile_levels is None:
            quantile_levels = [float(q) for q in model.config.quantile_levels]
        self.quantile_levels = list(quantile_levels)
        self.context_length = context_length
        self.batch_size = batch_size
        self.show_progress = show_progress
        self.inference_kwargs: InferenceKwargs = {
            "trim_leading_missing": trim_leading_missing,
            "symmetric_averaging": symmetric_averaging,
            "clip_nonnegative": clip_nonnegative,
            "snap_integers": snap_integers,
        }
        self.cross_inference = cross_inference

    def predict(self, test_data_input: Iterable[dict]) -> list[QuantileForecast]:
        """Forecast every entry, batching for efficiency.

        Entries within a batch are bucketed by context length so each
        forward pass runs on series of equal length; a CUDA/MPS
        out-of-memory error halves the batch size and retries.
        """
        entries = list(test_data_input)
        if self.cross_inference > 1:
            return self._predict_grouped(entries)
        batch_size = self.batch_size
        forecasts: list[QuantileForecast] = []
        with tqdm(
            total=len(entries), desc="predict", unit="series", leave=False, disable=not self.show_progress
        ) as bar:
            i = 0
            while i < len(entries):
                batch = entries[i : i + batch_size]
                try:
                    forecasts.extend(self._predict_batch(batch))
                except (torch.cuda.OutOfMemoryError, RuntimeError) as err:
                    if "out of memory" not in str(err).lower() or batch_size <= 1:
                        raise
                    batch_size //= 2
                    logger.warning("out of memory at batch_size %d, reducing to %d", batch_size * 2, batch_size)
                    continue
                i += len(batch)
                bar.update(len(batch))
        return forecasts

    def _contexts_from_entries(self, batch: list[dict]) -> list[np.ndarray]:
        """Turn GluonTS test entries into the model's context arrays.

        Each entry's ``target`` history becomes one univariate context,
        truncated to the most recent ``context_length`` observations.
        Dynamic covariates (``feat_dynamic_real`` / ``past_feat_dynamic_real``)
        that some GIFT-Eval datasets carry are intentionally dropped: the
        benchmark scores the target channel only, matching how the model is
        evaluated in production. (The model itself accepts covariates via a
        multivariate context, but the benchmark path stays target-only.)
        """
        contexts = [np.asarray(entry["target"], dtype=np.float32) for entry in batch]
        if self.context_length is not None:
            contexts = [c[-self.context_length :] for c in contexts]
        return contexts

    def _predict_batch(self, batch: list[dict]) -> list[QuantileForecast]:
        contexts = self._contexts_from_entries(batch)

        # Bucket by length so each forward runs on equal-length series and no
        # series is padded up to the batch maximum. The model is sensitive to
        # left-padding (it shifts the patch grid), so padding short series to a
        # long one measurably degrades them — bucketing avoids that entirely.
        by_length: dict[int, list[int]] = {}
        for i, c in enumerate(contexts):
            by_length.setdefault(len(c), []).append(i)

        quantiles = self._empty_quantiles(len(batch))
        for indices in by_length.values():
            out = self.model.predict(
                np.stack([contexts[i] for i in indices]),
                horizon=self.prediction_length,
                quantile_levels=self.quantile_levels,
                **self.inference_kwargs,
            )
            quantiles[indices] = out.quantiles.cpu().numpy()

        return self._forecasts(batch, quantiles)

    def _predict_grouped(self, entries: list[dict]) -> list[QuantileForecast]:
        """Forecast up to ``cross_inference`` series of one window together."""
        contexts = self._contexts_from_entries(entries)
        # An entry's window is its occurrence count among the entries of the same item.
        seen: dict[str, int] = {}
        by_window: dict[int, list[int]] = {}
        for i, entry in enumerate(entries):
            item = str(entry.get("item_id"))
            by_window.setdefault(seen.get(item, 0), []).append(i)
            seen[item] = seen.get(item, 0) + 1
        groups = [
            rows[start : start + self.cross_inference]
            for _, rows in sorted(by_window.items())
            for start in range(0, len(rows), self.cross_inference)
        ]
        groups_per_call = max(1, self.batch_size // self.cross_inference)
        device = next(self.model.parameters()).device
        quantiles = self._empty_quantiles(len(entries))
        with tqdm(
            total=len(entries), desc="predict", unit="series", leave=False, disable=not self.show_progress
        ) as bar:
            for start in range(0, len(groups), groups_per_call):
                chunk = groups[start : start + groups_per_call]
                blocks = []
                for members in chunk:
                    width = max(len(contexts[i]) for i in members)
                    block = np.full((len(members), width), np.nan, dtype=np.float32)
                    for row, i in enumerate(members):
                        block[row, width - len(contexts[i]) :] = contexts[i]
                    blocks.append(block)
                context, mask, group_ids = batch_series(blocks)
                out = self.model.predict(
                    context.to(device),
                    horizon=self.prediction_length,
                    quantile_levels=self.quantile_levels,
                    mask=mask.to(device),
                    group_ids=group_ids.to(device),
                    **self.inference_kwargs,
                )
                quantiles[[i for members in chunk for i in members]] = out.quantiles.cpu().numpy()
                bar.update(sum(len(members) for members in chunk))
        return self._forecasts(entries, quantiles)

    def _empty_quantiles(self, n: int) -> np.ndarray:
        return np.empty((n, self.prediction_length, len(self.quantile_levels)), dtype=np.float32)

    def _forecasts(self, entries: list[dict], quantiles: np.ndarray) -> list[QuantileForecast]:
        """One ``QuantileForecast`` per entry from ``[entries, horizon, Q]`` quantiles."""
        forecasts = []
        for i, entry in enumerate(entries):
            item_id = entry.get("item_id")
            forecasts.append(
                QuantileForecast(
                    forecast_arrays=quantiles[i].T,  # (Q, horizon)
                    forecast_keys=[str(q) for q in self.quantile_levels],
                    start_date=entry["start"] + len(entry["target"]),
                    item_id=str(item_id) if item_id is not None else None,
                )
            )
        return forecasts
