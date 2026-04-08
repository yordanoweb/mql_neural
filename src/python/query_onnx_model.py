"""
Read an ONNX file and try to recover the original feature column names.
"""

import argparse
import sys
from typing import List

import onnx


def _parse_metadata(model: onnx.ModelProto) -> List[str]:
    """Look for a metadata entry named 'feature_names'."""
    for prop in model.metadata_props:
        if prop.key == "feature_names":
            # Expect a comma‑separated list
            names = [n.strip() for n in prop.value.split(",") if n.strip()]
            return names
    return []


def _fallback_from_doc(model: onnx.ModelProto) -> List[str]:
    """Some exporters put a CSV list into model.doc_string."""
    if model.doc_string:
        parts = [p.strip() for p in model.doc_string.split(",") if p.strip()]
        if parts:
            return parts
    return []


def _names_from_inputs(model: onnx.ModelProto) -> List[str]:
    """If the model has N separate inputs, their names are the feature names."""
    return [inp.name for inp in model.graph.input]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Recover feature column names from an ONNX model."
    )
    parser.add_argument("onnx_path", help="Path to the ONNX file")
    args = parser.parse_args()

    try:
        model = onnx.load(args.onnx_path)
    except Exception as exc:
        sys.stderr.write(f"Unable to load ONNX model: {exc}\n")
        sys.exit(1)

    # 1️⃣ Try explicit metadata
    names = _parse_metadata(model)

    # 2️⃣ If not found, try doc_string
    if not names:
        names = _fallback_from_doc(model)

    # 3️⃣ If still empty, maybe each feature was exported as a distinct input
    if not names:
        names = _names_from_inputs(model)

    if names:
        print("Recovered feature names:")
        for i, n in enumerate(names, start=1):
            print(f"  {i}. {n}")
    else:
        print("⚠️  No feature names could be recovered.")
        print("   • If you have control over the export step, add a metadata property")
        print("     called 'feature_names' with a comma‑separated list.")
        print("   • Alternatively, export each column as a separate ONNX input.")


if __name__ == "__main__":
    main()
