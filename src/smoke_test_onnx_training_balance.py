import argparse
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import onnxruntime as ort
import pandas as pd

from indicators import calculate_rsi as rsi


def parse_args():
    parser = argparse.ArgumentParser(description="Smoke test: ONNX trainer should not collapse to one-sided predictions.")
    parser.add_argument("--rows", type=int, default=1200, help="Synthetic rows to generate (default: 1200)")
    parser.add_argument("--window", type=int, default=20, help="Training window size (default: 20)")
    parser.add_argument("--pip-unit", type=float, default=0.01, help="Pip unit used in features (default: 0.01)")
    parser.add_argument("--rsi-period", type=int, default=14, help="RSI period (default: 14)")
    parser.add_argument("--n-iter", type=int, default=5, help="Random search iterations (default: 5)")
    parser.add_argument("--n-splits", type=int, default=3, help="TimeSeriesSplit folds (default: 3)")
    parser.add_argument("--n-jobs", type=int, default=-1, help="Parallel jobs for trainer (default: -1)")
    return parser.parse_args()


def build_synthetic_ohlc(rows: int) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    base = 18000.0
    close = [base]
    # Alternating directional regime prevents trivial one-class labeling.
    for i in range(1, rows):
        direction = 1.0 if (i % 4 in (0, 1)) else -1.0
        step = direction * (2.5 + rng.normal(0.0, 0.4))
        close.append(close[-1] + step)
    close = np.array(close)
    open_ = np.concatenate(([close[0]], close[:-1]))
    high = np.maximum(open_, close) + np.abs(rng.normal(0.8, 0.2, size=rows))
    low = np.minimum(open_, close) - np.abs(rng.normal(0.8, 0.2, size=rows))
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close})


def calculate_rsi(series: pd.Series, period: int) -> pd.Series:
    return pd.Series(rsi(series.values.tolist(), period), index=series.index)


def build_windows(df: pd.DataFrame, window: int, pip_unit: float, rsi_period: int):
    feat = df.copy()
    feat["feat_body"] = (feat["close"] - feat["open"]) / pip_unit
    feat["feat_range"] = (feat["high"] - feat["low"]) / pip_unit
    feat["feat_rsi"] = calculate_rsi(feat["close"], rsi_period) / 100.0
    feat.dropna(inplace=True)
    features = ["feat_body", "feat_range", "feat_rsi"]
    X = []
    for i in range(window, len(feat) - 1):
        X.append(feat[features].iloc[i - window : i].values.flatten())
    return np.array(X).astype(np.float32)


def main():
    args = parse_args()

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp = Path(tmp_dir)
        csv_path = tmp / "US100_M15_synthetic_smoke.csv"
        onnx_path = tmp / "smoke_model.onnx"

        df = build_synthetic_ohlc(args.rows)
        df.to_csv(csv_path, index=False)

        cmd = [
            "python",
            str(Path(__file__).with_name("train_onnx_from_csv.py")),
            "--input-csv",
            str(csv_path),
            "--output-filename",
            str(onnx_path),
            "--window",
            str(args.window),
            "--pip-unit",
            str(args.pip_unit),
            "--rsi-period",
            str(args.rsi_period),
            "--n-iter",
            str(args.n_iter),
            "--n-splits",
            str(args.n_splits),
            "--n-jobs",
            str(args.n_jobs),
        ]
        subprocess.run(cmd, check=True)

        X = build_windows(df, args.window, args.pip_unit, args.rsi_period)
        if len(X) == 0:
            raise RuntimeError("No windows were generated for smoke test.")

        session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        input_name = session.get_inputs()[0].name
        outputs = session.run(None, {input_name: X})
        labels = np.array(outputs[0]).reshape(-1)
        unique, counts = np.unique(labels, return_counts=True)
        dist = {int(k): int(v) for k, v in zip(unique, counts)}

        print(f"Prediction distribution: {dist}")
        if len(dist) < 2:
            raise RuntimeError(f"Smoke test failed: model predicts only one class. distribution={dist}")

        print("Smoke test passed: model predicts both classes.")


if __name__ == "__main__":
    main()
