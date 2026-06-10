"""GluonTS-style predictor wrapping ``T0Forecaster`` for benchmark evaluation.

Designed for benchmarks built on GluonTS test splits (e.g. GIFT-Eval):
``predict`` consumes test-data input entries (dicts with ``target``,
``start``, and optionally ``item_id``) and returns one ``QuantileForecast``
per entry, ready for ``gluonts.model.evaluate_model``.

Requires the ``evaluation`` extra: ``pip install "tfc-t0[evaluation]"``.
"""

import logging
from collections.abc import Iterable, Sequence

import numpy as np
import torch

try:
    from gluonts.model.forecast import QuantileForecast
    from tqdm.auto import tqdm
except ImportError as err:  # pragma: no cover - exercised only without the extra
    raise ImportError('t0.evaluation requires the evaluation extra: pip install "tfc-t0[evaluation]"') from err

from t0.model import T0Forecaster

logger = logging.getLogger(__name__)


class T0Predictor:
    """Batched ``T0Forecaster`` predictor over GluonTS test-data entries.

    Args:
        model: A loaded ``T0Forecaster`` (already on its target device).
        prediction_length: Forecast horizon in timesteps.
        quantile_levels: Quantile levels to forecast. Defaults to the
            model's trained levels; other levels are interpolated by
            ``T0Forecaster.predict``.
        context_length: If set, keep only the most recent ``context_length``
            observations of each series.
        batch_size: Entries per forward pass.
        show_progress: Display a progress bar over the entries while
            predicting.
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
    ):
        self.model = model
        self.prediction_length = prediction_length
        if quantile_levels is None:
            quantile_levels = [float(q) for q in model.config.quantile_levels]
        self.quantile_levels = list(quantile_levels)
        self.context_length = context_length
        self.batch_size = batch_size
        self.show_progress = show_progress

    def predict(self, test_data_input: Iterable[dict]) -> list[QuantileForecast]:
        """Forecast every entry, batching for efficiency.

        Entries within a batch are bucketed by context length so each
        forward pass runs on series of equal length; a CUDA/MPS
        out-of-memory error halves the batch size and retries.
        """
        entries = list(test_data_input)
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

        quantiles = np.empty((len(batch), self.prediction_length, len(self.quantile_levels)), dtype=np.float32)
        for indices in by_length.values():
            out = self.model.predict(
                np.stack([contexts[i] for i in indices]),
                horizon=self.prediction_length,
                quantiles=self.quantile_levels,
            )
            quantiles[indices] = out.quantiles.cpu().numpy()

        forecasts = []
        for i, entry in enumerate(batch):
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
