"""Validate that the original t0-alpha checkpoint loads directly in MLX."""

import argparse
import hashlib
from pathlib import Path
from typing import cast

import mlx.core as mx
from mlx.utils import tree_flatten

from t0_mlx import T0Config, T0Forecaster


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_checkpoint(source: Path) -> tuple[T0Config, str]:
    """Verify that every checkpoint tensor matches the MLX parameter tree."""
    config_path = source / "config.json"
    weights_path = source / "model.safetensors"
    if not config_path.is_file() or not weights_path.is_file():
        raise FileNotFoundError(f"{source} must contain config.json and model.safetensors")

    config = T0Config.from_json(config_path)
    expected = dict(tree_flatten(T0Forecaster.from_config(config).parameters()))
    loaded = mx.load(str(weights_path))
    if not isinstance(loaded, dict):
        raise ValueError("expected model.safetensors to contain a named tensor mapping")
    actual = cast(dict[str, mx.array], loaded)
    if set(expected) != set(actual):
        missing = sorted(set(expected) - set(actual))
        unexpected = sorted(set(actual) - set(expected))
        raise ValueError(f"checkpoint key mismatch: missing={missing}, unexpected={unexpected}")
    mismatched = {
        key: (expected[key].shape, actual[key].shape) for key in expected if expected[key].shape != actual[key].shape
    }
    if mismatched:
        raise ValueError(f"checkpoint shape mismatch: {mismatched}")
    non_fp32 = sorted(key for key, value in actual.items() if value.dtype != mx.float32)
    if non_fp32:
        raise ValueError(f"expected an FP32 checkpoint; other dtypes found at {non_fp32}")
    return config, _sha256(weights_path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path, help="directory containing the original checkpoint")
    args = parser.parse_args()

    config, digest = validate_checkpoint(args.checkpoint)
    print(f"validated {config.num_layers} layers; model.safetensors sha256={digest}")


if __name__ == "__main__":
    main()
