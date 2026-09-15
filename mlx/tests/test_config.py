import json

import pytest

from t0_mlx import T0Config


def test_medium_matches_published_checkpoint() -> None:
    config = T0Config.medium()
    assert config.embed_dim == 512
    assert config.num_layers == 24
    assert config.group_every_n == 3
    assert config.quantile_levels == (0.1, 0.25, 0.5, 0.75, 0.9)


def test_json_round_trip(tmp_path) -> None:
    path = tmp_path / "config.json"
    path.write_text(json.dumps(T0Config.medium().to_dict()))
    assert T0Config.from_json(path) == T0Config.medium()


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"embed_dim": 513}, "divisible"),
        ({"group_every_n": 5}, "divide"),
        ({"quantile_levels": (0.5, 0.1)}, "sorted"),
    ],
)
def test_invalid_config_is_rejected(override, message) -> None:
    values = T0Config.medium().to_dict()
    values.update(override)
    with pytest.raises(ValueError, match=message):
        T0Config.from_dict(values)
