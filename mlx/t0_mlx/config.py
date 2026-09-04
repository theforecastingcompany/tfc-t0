"""Architecture configuration for T0Forecaster.

A plain frozen dataclass. ``T0Config.medium()`` returns the hyperparameters of
the published t0-alpha checkpoint.
"""

import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

if sys.version_info >= (3, 11):
    from typing import Self
else:
    from typing_extensions import Self


@dataclass(frozen=True)
class T0Config:
    """Hyperparameters for an instance of T0Forecaster.

    ``quantile_levels`` must be a non-empty, ascending tuple of unique floats
    in ``(0, 1)``; use ``T0Config.medium()`` for the published configuration.
    """

    embed_dim: int
    num_layers: int
    num_heads: int
    mlp_hidden_dim: int
    patch_size: int
    group_every_n: int
    dropout: float
    quantile_levels: tuple[float, ...]
    scaler_use_arcsinh: bool = True

    def __post_init__(self) -> None:
        positive = {
            "embed_dim": self.embed_dim,
            "num_layers": self.num_layers,
            "num_heads": self.num_heads,
            "mlp_hidden_dim": self.mlp_hidden_dim,
            "patch_size": self.patch_size,
        }
        for name, value in positive.items():
            if value < 1:
                raise ValueError(f"{name} must be positive, got {value}")
        if self.embed_dim % self.num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        if self.group_every_n > 0 and self.num_layers % self.group_every_n != 0:
            raise ValueError("group_every_n must divide num_layers")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError(f"dropout must be in [0, 1), got {self.dropout}")
        if not self.quantile_levels:
            raise ValueError("quantile_levels must be non-empty")
        if tuple(sorted(set(self.quantile_levels))) != self.quantile_levels:
            raise ValueError("quantile_levels must be sorted ascending without duplicates")
        if any(not 0.0 < level < 1.0 for level in self.quantile_levels):
            raise ValueError("each quantile level must be in (0, 1)")

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> Self:
        """Construct from the JSON-compatible configuration mapping."""
        fields = {
            "embed_dim",
            "num_layers",
            "num_heads",
            "mlp_hidden_dim",
            "patch_size",
            "group_every_n",
            "dropout",
            "quantile_levels",
            "scaler_use_arcsinh",
        }
        filtered = {key: value for key, value in values.items() if key in fields}
        if "quantile_levels" in filtered:
            filtered["quantile_levels"] = tuple(filtered["quantile_levels"])
        return cls(**filtered)

    @classmethod
    def from_json(cls, path: str | Path) -> Self:
        """Load a configuration from `config.json`."""
        with Path(path).open(encoding="utf-8") as config_file:
            values = json.load(config_file)
        if not isinstance(values, dict):
            raise ValueError("config must contain a JSON object")
        return cls.from_dict(values)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible representation."""
        values = asdict(self)
        values["quantile_levels"] = list(self.quantile_levels)
        return values

    @classmethod
    def medium(cls) -> Self:
        """Return the published t0-alpha configuration."""
        return cls(
            embed_dim=512,
            num_layers=24,
            num_heads=8,
            mlp_hidden_dim=2048,
            patch_size=32,
            group_every_n=3,
            dropout=0.1,
            quantile_levels=(0.1, 0.25, 0.5, 0.75, 0.9),
            scaler_use_arcsinh=True,
        )
