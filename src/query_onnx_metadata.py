import argparse
import json
import onnx


def parse_args():
    parser = argparse.ArgumentParser(description="Query and print ONNX metadata.")
    parser.add_argument("--onnx-file", required=True, help="Path to ONNX model file")
    parser.add_argument("--raw-json", action="store_true", help="Print metadata as JSON map")
    return parser.parse_args()


def maybe_parse_json(value):
    try:
        return json.loads(value)
    except (json.JSONDecodeError, TypeError):
        return value


def main():
    args = parse_args()
    model = onnx.load(args.onnx_file)

    metadata_map = {}
    for prop in model.metadata_props:
        metadata_map[prop.key] = maybe_parse_json(prop.value)

    if args.raw_json:
        print(json.dumps(metadata_map, indent=2, sort_keys=True))
        return

    print(f"ONNX file: {args.onnx_file}")
    print(f"Metadata entries: {len(metadata_map)}")
    if len(metadata_map) == 0:
        print("No metadata found.")
        return

    for key in sorted(metadata_map.keys()):
        value = metadata_map[key]
        if isinstance(value, (dict, list)):
            value = json.dumps(value, sort_keys=True)
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
