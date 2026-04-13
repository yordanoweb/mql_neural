"""
Inspect an ONNX model: show all metadata, tensor shapes, and feature names.

Usage:
    python query_onnx_model.py path/to/model.onnx
"""

import argparse
import sys

import onnx
import onnxruntime as rt


def main() -> None:
    parser = argparse.ArgumentParser(description="Query ONNX model metadata.")
    parser.add_argument("onnx_path", help="Path to the ONNX file")
    args = parser.parse_args()

    try:
        model = onnx.load(args.onnx_path)
    except Exception as exc:
        sys.stderr.write(f"Cannot load model: {exc}\n")
        sys.exit(1)

    # --- tensor shapes via onnxruntime ---
    sess = rt.InferenceSession(args.onnx_path)
    inp  = sess.get_inputs()[0]
    out  = sess.get_outputs()[0]
    print(f"Input : {inp.name}  shape={inp.shape}  type={inp.type}")
    print(f"Output: {out.name}  shape={out.shape}  type={out.type}")

    # --- all metadata ---
    meta = {p.key: p.value for p in model.metadata_props}
    if not meta:
        print("\nNo metadata found.")
        sys.exit(0)

    print("\nMetadata:")
    for k, v in meta.items():
        print(f"  {k}: {v}")

    # --- feature names (structured) ---
    if 'feature_names' in meta:
        names = [n.strip() for n in meta['feature_names'].split(',') if n.strip()]
        print(f"\nFeatures ({len(names)}):")
        for i, n in enumerate(names, 1):
            print(f"  {i:>2}. {n}")


if __name__ == "__main__":
    main()
