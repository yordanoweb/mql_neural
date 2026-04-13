"""
ONNX export helpers.

Contract (never break):
  Input : float32[1, WINDOW_SIZE * N_FEATURES]
  Output: float32[1, 2]  — softmax [P(sell), P(buy)]
  Metadata: feature_names (comma-sep), window_size, n_features
"""

import onnx
import onnxruntime as rt
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType


def export(model, feature_cols: list[str], window: int, output_path: str) -> None:
    """Convert a fitted sklearn model to ONNX and save with required metadata."""
    n_flat = window * len(feature_cols)
    initial_type = [('float_input', FloatTensorType([None, n_flat]))]
    onnx_model = convert_sklearn(
        model, initial_types=initial_type,
        options={type(model): {'zipmap': False}}
    )

    for key, val in [
        ('feature_names', ','.join(feature_cols)),
        ('window_size',   str(window)),
        ('n_features',    str(len(feature_cols))),
    ]:
        p = onnx_model.metadata_props.add()
        p.key, p.value = key, val

    onnx.save(onnx_model, output_path)
    _verify(output_path)


def _verify(path: str) -> None:
    sess = rt.InferenceSession(path)
    inp  = sess.get_inputs()[0]
    out  = sess.get_outputs()[0]
    print(f"Saved : {path}")
    print(f"Input : {inp.name} {inp.shape} {inp.type}")
    print(f"Output: {out.name} {out.shape} {out.type}")
    assert out.shape == [None, 2] or out.shape[1] == 2, "Output must be [*, 2]"
