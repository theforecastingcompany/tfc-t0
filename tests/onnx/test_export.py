"""The grouped ONNX graph must forecast what ``T0Forecaster.predict`` forecasts."""

import dataclasses

import numpy as np
import onnxruntime as ort
import pytest
import torch
from export_onnx import export_fp32, save, to_int8
from validate_onnx import compare, make_case, reference

from t0 import T0Config
from t0.torch.model import T0Forecaster


def _small_model(zero_no_value: bool) -> T0Forecaster:
    config = dataclasses.replace(
        T0Config.large(),
        embed_dim=64,
        num_layers=3,
        num_heads=2,
        mlp_hidden_dim=128,
        xpos_scale_layout="halves",
        scaler_zero_no_value=zero_no_value,
    )
    torch.manual_seed(0)
    return T0Forecaster.from_config(config).eval()


@pytest.mark.parametrize("zero_no_value", [True, False], ids=["alpha-beta", "gamma"])
def test_fp32_graph_matches_predict(tmp_path, zero_no_value: bool) -> None:
    model = _small_model(zero_no_value)
    path = tmp_path / "grouped-fp32.onnx"
    save(export_fp32(model), path)
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    model = model.double()
    # Both ends of the dynamic shapes: a one-step context, an unaligned one and the longest horizon.
    for context, horizon in ((1, 32), (97, 64), (300, 1024)):
        feeds = make_case(context, horizon, seed=context)
        rows = compare(session.run(["quantiles"], feeds)[0], reference(model, feeds))
        for row in rows:
            assert row["finite"]
            assert row["max_drift_over_spread"] < 1e-3, (context, horizon, row)


def test_int8_quantizes_only_transformer_matrices() -> None:
    model = _small_model(zero_no_value=False)
    quantized = to_int8(export_fp32(model))
    dequantized = [node for node in quantized.graph.node if node.op_type == "DequantizeLinear"]
    # wQKV, wO and the two MLP matrices of every layer.
    assert len(dequantized) == 4 * model.config.num_layers
    assert all(np.prod(init.dims) > 0 for init in quantized.graph.initializer)
