"""Export a t0 checkpoint to the grouped ONNX interface, in FP32, FP16 or INT8 weights.

The graph has the interface of the published ``t0-*-onnx-*`` artifacts:

| Input/output | Type | Shape |
| --- | --- | --- |
| ``target_context`` | float32 | ``[target_rows, context_steps]`` |
| ``target_group_ids`` | int32 | ``[target_rows]`` |
| ``future_covariate_context`` | float32 | ``[covariate_rows, context_steps]`` |
| ``future_covariate_future`` | float32 | ``[covariate_rows, 32 * horizon_patches]`` |
| ``future_covariate_group_ids`` | int32 | ``[covariate_rows]`` |
| ``quantiles`` | float32 | ``[target_rows, 32 * horizon_patches, quantiles]`` |

NaN marks a missing observation. The graph runs one forward pass, so the horizon is at
most ``DEFAULT_MAX_HORIZON``, and returns the checkpoint's native quantile levels. It
applies the checkpoint's ``config.json`` conventions through ``T0Forecaster``, so it
needs the ``onnx`` dependency group of ``tfc-t0>=0.6.0``.

    uv run --group onnx python tools/export_onnx.py theforecastingcompany/t0-gamma out/ --precision fp32 fp16 int8
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import onnx
import torch
from onnx import TensorProto, helper, numpy_helper
from torch import Tensor, nn

from t0 import TimeSeries
from t0.forecaster import DEFAULT_MAX_HORIZON
from t0.torch.model import T0Forecaster
from t0.types import MaskType, VariateType

OPSET = 20
INPUT_NAMES = [
    "target_context",
    "target_group_ids",
    "future_covariate_context",
    "future_covariate_future",
    "future_covariate_group_ids",
]
PRECISIONS = ("fp32", "fp16", "int8")
# MatMul weights with fewer elements stay FP32 in the FP16 export.
FP16_MIN_ELEMENTS = 1 << 16


class GroupedGraph(nn.Module):
    """The model, from raw grouped arrays to the native quantiles of the forecast region."""

    def __init__(self, model: T0Forecaster):
        super().__init__()
        self.model = model

    def forward(
        self,
        target_context: Tensor,
        target_group_ids: Tensor,
        future_covariate_context: Tensor,
        future_covariate_future: Tensor,
        future_covariate_group_ids: Tensor,
    ) -> Tensor:
        model = self.model
        patch_size = model.patch_size
        n_targets, context = target_context.shape
        n_covariates, horizon = future_covariate_future.shape
        rows = n_targets + n_covariates
        pad_left = (patch_size - context % patch_size) % patch_size
        device = target_context.device

        values = torch.cat(
            [
                torch.cat([target_context, target_context.new_zeros((n_targets, horizon))], dim=1),
                torch.cat([future_covariate_context, future_covariate_future], dim=1),
            ],
            dim=0,
        )
        is_future = torch.cat(
            [
                torch.zeros(n_targets, dtype=torch.bool, device=device),
                torch.ones(n_covariates, dtype=torch.bool, device=device),
            ]
        )
        columns = torch.arange(context + horizon, device=device)
        withheld = ~is_future[:, None] & (columns >= context)[None, :]
        missing = torch.isnan(values) & ~withheld
        mask = torch.full_like(values, MaskType.VALID, dtype=torch.int64)
        mask = torch.where(missing, torch.full_like(mask, MaskType.MISSING), mask)
        mask = torch.where(withheld, torch.full_like(mask, MaskType.WITHHELD), mask)
        values = torch.where(torch.isnan(values), torch.zeros_like(values), values)

        # Shift each row by its first observation before the scaler's float32 running sums,
        # which lose a level far from zero relative to its spread. The scaler's statistics
        # move with the shift, so the shift is exact: it starts at the first observation on
        # causal rows (before it the causal mean is zero) and covers every cell of a future
        # row (whose mean is global). Predictions get the shift back.
        observed = mask == MaskType.VALID
        has_observation = observed.any(dim=1)
        first = torch.argmax(observed.to(torch.int32), dim=1)
        offset = torch.where(has_observation, values.gather(1, first[:, None])[:, 0], torch.zeros_like(values[:, 0]))
        shift_from = torch.where(is_future, torch.zeros_like(first), first)
        shifted = columns[None, :] >= shift_from[:, None]
        values = torch.where(shifted, values - offset[:, None], values)

        group_ids = torch.cat([target_group_ids, future_covariate_group_ids]).to(torch.int64)
        variate_type = torch.where(
            is_future, torch.full_like(group_ids, VariateType.FUTURE), torch.full_like(group_ids, VariateType.TARGET)
        )
        values = nn.functional.pad(values, (pad_left, 0), value=0.0)
        mask = nn.functional.pad(mask, (pad_left, 0), value=int(MaskType.PAD)).to(torch.int8)
        group_ids = nn.functional.pad(group_ids[:, None].expand(rows, context + horizon), (pad_left, 0), value=-1)
        variate_type = nn.functional.pad(variate_type[:, None].expand(rows, context + horizon), (pad_left, 0), value=-1)

        window = TimeSeries(variates=values, mask=mask, group_ids=group_ids, variate_type=variate_type)
        scaled, loc_scale = model.scaler.scale_input(window)
        patched = [
            model.patcher.patch(field)
            for field in (scaled.variates, scaled.mask, scaled.variate_type, scaled.group_ids)
        ]
        patched_values, patched_mask, patched_type, patched_groups = patched
        embeddings = model.patch_encoder(patched_values, patched_mask, patched_type)
        embeddings = model.transformer(embeddings, patched_groups, patched_type, patched_mask)
        decoded = model.decoder(embeddings)
        decoded = decoded.reshape(*decoded.shape[:-1], patch_size, model.head.n_quantiles)
        predictions = model.scaler.rescale_predictions(model.head(decoded), loc_scale, patch_size)
        predictions = predictions + offset[:, None, None, None]

        context_patches = (pad_left + context) // patch_size
        horizon_patches = horizon // patch_size
        predictions = predictions[:n_targets, context_patches - 1 : context_patches - 1 + horizon_patches]
        predictions = predictions.reshape(n_targets, horizon, model.head.n_quantiles)
        return torch.where(torch.isfinite(predictions), predictions, torch.zeros_like(predictions))


def example_inputs(context: int = 97, horizon: int = 64) -> tuple[Tensor, ...]:
    """Two target rows in one group, one known-future covariate in that group."""
    steps = torch.arange(context + horizon, dtype=torch.float32)
    targets = torch.stack([torch.sin(steps[:context] / 6), torch.cos(steps[:context] / 9)]) + 10
    covariate = torch.sin(2 * torch.pi * steps / 7)[None, :]
    return (
        targets,
        torch.tensor([7, 7], dtype=torch.int32),
        covariate[:, :context].contiguous(),
        covariate[:, context:].contiguous(),
        torch.tensor([7], dtype=torch.int32),
    )


def export_fp32(model: T0Forecaster) -> onnx.ModelProto:
    """Trace the grouped graph with dynamic rows, context and horizon."""
    model = model.eval().float()
    target_rows = torch.export.Dim("target_rows", min=1)
    covariate_rows = torch.export.Dim("covariate_rows", min=0)
    context_steps = torch.export.Dim("context_steps", min=1, max=8192)
    horizon_patches = torch.export.Dim("horizon_patches", min=1, max=DEFAULT_MAX_HORIZON // model.patch_size)
    dynamic_shapes = {
        "target_context": {0: target_rows, 1: context_steps},
        "target_group_ids": {0: target_rows},
        "future_covariate_context": {0: covariate_rows, 1: context_steps},
        "future_covariate_future": {0: covariate_rows, 1: model.patch_size * horizon_patches},
        "future_covariate_group_ids": {0: covariate_rows},
    }
    with torch.inference_mode():
        program = torch.onnx.export(
            GroupedGraph(model),
            example_inputs(),
            input_names=INPUT_NAMES,
            output_names=["quantiles"],
            dynamic_shapes=dynamic_shapes,
            opset_version=OPSET,
            dynamo=True,
            optimize=True,
        )
    return program.model_proto


def _matmul_weights(model: onnx.ModelProto) -> dict[str, tuple[onnx.TensorProto, str]]:
    """Float 2-D initializers that feed the weight input of a MatMul, with the module scope of that MatMul."""
    initializers = {init.name: init for init in model.graph.initializer}
    weights = {}
    for node in model.graph.node:
        init = initializers.get(node.input[1]) if node.op_type == "MatMul" else None
        if init is not None and init.data_type == TensorProto.FLOAT and len(init.dims) == 2:
            scope = next((prop.value for prop in node.metadata_props if prop.key == "namespace"), "")
            weights[init.name] = (init, scope)
    return weights


def _with_weights(source: onnx.ModelProto, replaced: dict[str, tuple[list, onnx.NodeProto]]) -> onnx.ModelProto:
    """A copy of ``source`` where each named initializer is rebuilt in FP32 by a node from new initializers."""
    model = onnx.ModelProto()
    model.CopyFrom(source)
    graph = model.graph
    kept = [init for init in graph.initializer if init.name not in replaced]
    nodes = list(graph.node)
    del graph.initializer[:], graph.node[:]
    graph.initializer.extend(kept + [init for initializers, _ in replaced.values() for init in initializers])
    graph.node.extend([node for _, node in replaced.values()] + nodes)
    return model


def to_fp16(source: onnx.ModelProto) -> onnx.ModelProto:
    """Store the large MatMul weights in FP16 and cast them back to FP32 for compute."""
    replaced = {}
    for name, (init, _) in _matmul_weights(source).items():
        if int(np.prod(init.dims)) >= FP16_MIN_ELEMENTS:
            stored = numpy_helper.from_array(numpy_helper.to_array(init).astype(np.float16), f"{name}_fp16")
            replaced[name] = ([stored], helper.make_node("Cast", [stored.name], [name], to=TensorProto.FLOAT))
    return _with_weights(source, replaced)


def to_int8(source: onnx.ModelProto) -> onnx.ModelProto:
    """Store the transformer's MatMul weights as per-output-channel symmetric INT8.

    The patch encoder and the decoder stay FP32, which keeps the extreme quantiles close
    to the FP32 graph.
    """
    replaced = {}
    for name, (init, scope) in _matmul_weights(source).items():
        if "/model.transformer" not in scope:
            continue
        weight = numpy_helper.to_array(init)
        scale = np.abs(weight).max(axis=0) / 127.0
        scale = np.where(scale > 0, scale, 1.0).astype(np.float32)
        quantized = np.clip(np.round(weight / scale), -127, 127).astype(np.int8)
        stored = [
            numpy_helper.from_array(quantized, f"{name}_int8"),
            numpy_helper.from_array(scale, f"{name}_scale"),
            numpy_helper.from_array(np.zeros_like(quantized[0]), f"{name}_zero_point"),
        ]
        dequantize = helper.make_node("DequantizeLinear", [init.name for init in stored], [name], axis=1)
        replaced[name] = (stored, dequantize)
    return _with_weights(source, replaced)


def save(model: onnx.ModelProto, path: Path) -> None:
    """Drop the exporter's debug metadata (module scopes, local stack traces), check and save."""
    for node in model.graph.node:
        del node.metadata_props[:]
        node.doc_string = ""
    onnx.checker.check_model(model)
    onnx.save(model, str(path))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("checkpoint", help="Hugging Face repository id, or a local checkpoint directory")
    parser.add_argument("output", type=Path, help="directory to write the .onnx files to")
    parser.add_argument("--name", help="artifact prefix; defaults to the checkpoint directory name")
    parser.add_argument("--precision", nargs="+", choices=PRECISIONS, default=list(PRECISIONS))
    args = parser.parse_args()

    name = args.name or Path(args.checkpoint).name
    model = T0Forecaster.from_pretrained(args.checkpoint).eval()
    args.output.mkdir(parents=True, exist_ok=True)
    fp32 = export_fp32(model)
    builders = {"fp32": lambda: fp32, "fp16": lambda: to_fp16(fp32), "int8": lambda: to_int8(fp32)}
    # Build every variant before saving any: saving strips the module scopes that to_int8 reads.
    variants = {precision: builders[precision]() for precision in args.precision}
    for precision, variant in variants.items():
        artifact = args.output / f"{name}-grouped-{precision}.onnx"
        save(variant, artifact)
        print(json.dumps({"artifact": artifact.name, "bytes": artifact.stat().st_size, "sha256": sha256(artifact)}))


if __name__ == "__main__":
    main()
