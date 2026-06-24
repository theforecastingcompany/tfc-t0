"""Architecture configuration for T0Forecaster.

A plain frozen dataclass. ``T0Config.medium()`` returns the
hyperparameters of the published t0-alpha checkpoint.
"""

from dataclasses import dataclass
from typing import Self


@dataclass(frozen=True)
class T0Config:
    """Hyperparameters for an instance of T0Forecaster.

    ``quantile_levels`` must be a non-empty tuple of floats in ``(0, 1)``;
    use ``T0Config.medium()`` for the published configuration.

    Attributes:
        embed_dim: Transformer hidden size.
        num_layers: Number of transformer layers.
        num_heads: Number of attention heads. ``embed_dim`` must be divisible
            by ``num_heads``.
        mlp_hidden_dim: Hidden size of each transformer block's feedforward.
        patch_size: Number of timesteps per patch — the model's decode
            step width. ``T0Forecaster.predict()`` decodes whole blocks of
            patches per forward pass.
        group_every_n: Frequency of GROUP attention layers within the stack.
            Must divide ``num_layers``. ``-1`` disables group attention.
        dropout: Dropout probability throughout the transformer (training
            only — ignored at inference).
        quantile_levels: Quantile levels the model was trained to emit.
            The inference rollout interpolates these to whatever the user
            requests.
        scaler_use_arcsinh: Whether the scaler applies arcsinh after
            mean/std normalization.
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
        if not self.quantile_levels:
            raise ValueError("quantile_levels must be a non-empty tuple of floats in (0, 1)")
        for q in self.quantile_levels:
            if not (0.0 < q < 1.0):
                raise ValueError(f"each quantile must be in (0, 1); got {q}")

    @classmethod
    def medium(cls) -> Self:
        """The published t0-alpha configuration."""
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
