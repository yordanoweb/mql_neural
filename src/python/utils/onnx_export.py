"""
ONNX export helpers.

Contract (never break):
  Input : float32[1, WINDOW_SIZE * N_FEATURES]
  Output: float32[1, 3]  — softmax [P(hold), P(buy), P(sell)]
  Metadata: feature_names (comma-sep), window_size, n_features
"""

import onnx
import onnxruntime as rt
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType


def export(model, feature_cols: list[str], window: int, output_path: str) -> None:
    """Convert a fitted sklearn model to ONNX and save with required metadata."""
    from sklearn.pipeline import Pipeline
    n_flat = window * len(feature_cols)
    initial_type = [('float_input', FloatTensorType([None, n_flat]))]

    # target the final estimator for zipmap option, whether wrapped in Pipeline or not
    estimator = model.steps[-1][1] if isinstance(model, Pipeline) else model
    onnx_model = convert_sklearn(
        model, initial_types=initial_type,
        options={type(estimator): {'zipmap': False}},
        target_opset=17,
    )

    for key, val in [
        ('feature_names', ','.join(feature_cols)),
        ('window_size',   str(window)),
        ('n_features',    str(len(feature_cols))),
    ]:
        p = onnx_model.metadata_props.add()
        p.key, p.value = key, val

    onnx.checker.check_model(onnx_model)
    onnx.save(onnx_model, output_path)
    _verify(output_path)


def _verify(path: str) -> None:
    sess    = rt.InferenceSession(path)
    inp     = sess.get_inputs()[0]
    outputs = {o.name: o for o in sess.get_outputs()}
    print(f"Saved : {path}")
    print(f"Input : {inp.name} {inp.shape} {inp.type}")
    for o in sess.get_outputs():
        print(f"Output: {o.name} {o.shape} {o.type}")
    assert 'probabilities' in outputs and outputs['probabilities'].shape[1] == 3, \
        f"Expected 'probabilities' output with shape [*, 3]. Got: {list(outputs.keys())}"
