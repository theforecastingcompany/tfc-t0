"""Architecture configuration for T0Forecaster.

A plain frozen dataclass. ``T0Config.medium()`` and ``T0Config.large()``
return the hyperparameters of the two published checkpoints, t0-alpha and
t0-beta respectively.
"""

import sys
from dataclasses import dataclass
from typing import Literal

if sys.version_info >= (3, 11):
    from typing import Self
else:
    from typing_extensions import Self

# How the scaler keeps a standard deviation away from zero: "variance_offset"
# adds eps to the variance before the square root, "std_clamp" lower-bounds the
# square root at eps. A checkpoint is only valid under the one it trained with.
ScalerEpsMode = Literal["variance_offset", "std_clamp"]


@dataclass(frozen=True)
class T0Config:
    """Hyperparameters for an instance of T0Forecaster.

    ``quantile_levels`` must be a non-empty tuple of floats in ``(0, 1)``;
    use ``T0Config.medium()`` for t0-alpha and ``T0Config.large()`` for
    t0-beta.

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
            The inference rollout interpolates between these, and extrapolates on
            exponential tails beyond them, to whatever the user requests.
        scaler_use_arcsinh: Whether the scaler applies arcsinh after
            mean/std normalization.
        scaler_eps: Keeps the scale away from zero so a flat stretch of a
            series can't divide by it. ``scaler_eps_mode`` decides how, and
            only ``"std_clamp"`` makes this the exact lower bound.
        scaler_eps_mode: How ``scaler_eps`` reaches the causal statistic.
            ``"variance_offset"`` computes ``sqrt(variance + eps)``;
            ``"std_clamp"`` computes ``sqrt(variance).clamp(min=eps)``. A
            checkpoint must be run under the one it was trained with — they
            agree wherever a series' local standard deviation sits well above
            ``eps`` and diverge below it. The global (known-future) statistic
            always clamps.
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
    # Defaults reproduce t0-alpha, whose published config.json predates both
    # fields — from_pretrained on it must stay bit-identical.
    scaler_eps: float = 0.1
    scaler_eps_mode: ScalerEpsMode = "variance_offset"

    def __post_init__(self) -> None:
        if not self.quantile_levels:
            raise ValueError("quantile_levels must be a non-empty tuple of floats in (0, 1)")
        for q in self.quantile_levels:
            if not (0.0 < q < 1.0):
                raise ValueError(f"each quantile must be in (0, 1); got {q}")
        if self.scaler_eps <= 0.0:
            raise ValueError(f"scaler_eps must be positive; got {self.scaler_eps}")
        if self.scaler_eps_mode not in ("variance_offset", "std_clamp"):
            raise ValueError(f"scaler_eps_mode must be 'variance_offset' or 'std_clamp'; got {self.scaler_eps_mode!r}")

    @classmethod
    def medium(cls) -> Self:
        """The 102M-parameter configuration, published as t0-alpha."""
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
            scaler_eps=0.1,
            scaler_eps_mode="variance_offset",
        )

    @classmethod
    def large(cls) -> Self:
        """The 256M-parameter configuration, published as t0-beta."""
        return cls(
            embed_dim=1024,
            num_layers=24,
            num_heads=8,
            mlp_hidden_dim=2048,
            patch_size=32,
            group_every_n=3,
            dropout=0.1,
            quantile_levels=(
                0.01,
                0.05,
                0.1,
                0.15,
                0.2,
                0.25,
                0.3,
                0.35,
                0.4,
                0.45,
                0.5,
                0.55,
                0.6,
                0.65,
                0.7,
                0.75,
                0.8,
                0.85,
                0.9,
                0.95,
                0.99,
            ),
            scaler_use_arcsinh=True,
            scaler_eps=0.01,
            scaler_eps_mode="std_clamp",
        )
