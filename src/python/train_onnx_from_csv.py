import argparse
import os
from pathlib import Path

import numpy as np
import onnx
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
from ta.momentum import RSIIndicator
from ta.volatility import AverageTrueRange


def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--input", required=True, help="CSV file path")
    parser.add_argument(
        "--output",
        default=None,
        help="ONNX output path (auto-generated from input when omitted)",
    )
    parser.add_argument("--window", type=int, default=20, help="Window size")
    parser.add_argument(
        "--forward", type=int, default=1, help="Forward bars for binary target"
    )
    parser.add_argument("--rsi_period", type=int, default=14, help="RSI period")
    parser.add_argument("--atr_period", type=int, default=14, help="ATR period")
    parser.add_argument(
        "--n_iter", type=int, default=5, help="RandomizedSearchCV iterations"
    )
    parser.add_argument(
        "--cv_splits", type=int, default=2, help="TimeSeriesSplit folds"
    )
    parser.add_argument("--jobs", type=int, default=-1, help="Parallel jobs")
    parser.add_argument("--random_state", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--opset", type=int, default=12, help="ONNX target opset for export"
    )
    parser.add_argument(
        "--date_col",
        default=None,
        help="Column name for date (if separate from time)",
    )
    parser.add_argument(
        "--time_col",
        default="time",
        help="Column name for time (or datetime if combined)",
    )
    parser.add_argument("--open_col", default="open", help="Column name for open")
    parser.add_argument("--high_col", default="high", help="Column name for high")
    parser.add_argument("--low_col", default="low", help="Column name for low")
    parser.add_argument("--close_col", default="close", help="Column name for close")
    parser.add_argument(
        "--volume_col", default="tick_volume", help="Column name for volume"
    )
    args = parser.parse_args()

    if not os.path.exists(args.input):
        raise FileNotFoundError(f"File not found: {args.input}")
    if args.window < 1:
        raise ValueError("--window must be >= 1")
    if args.forward < 1:
        raise ValueError("--forward must be >= 1")
    if args.rsi_period < 1 or args.atr_period < 1:
        raise ValueError("--rsi_period and --atr_period must be >= 1")
    if args.cv_splits < 2:
        raise ValueError("--cv_splits must be >= 2")
    if args.n_iter < 1:
        raise ValueError("--n_iter must be >= 1")

    return args


def _normalize_colname(col: str) -> str:
    return (
        str(col)
        .strip()
        .lower()
        .replace("<", "")
        .replace(">", "")
        .replace("_", "")
        .replace(" ", "")
    )


def _pick_column(df: pd.DataFrame, preferred: str | None, aliases: list[str]) -> str | None:
    normalized_cols = {_normalize_colname(c): c for c in df.columns}

    if preferred:
        preferred_norm = _normalize_colname(preferred)
        if preferred in df.columns:
            return preferred
        if preferred_norm in normalized_cols:
            return normalized_cols[preferred_norm]

    for alias in aliases:
        alias_norm = _normalize_colname(alias)
        if alias_norm in normalized_cols:
            return normalized_cols[alias_norm]

    for alias in aliases:
        alias_norm = _normalize_colname(alias)
        for col in df.columns:
            if alias_norm and alias_norm in _normalize_colname(col):
                return col

    return None


def load_market_csv(
    path: str,
    date_col: str | None,
    time_col: str,
    open_col: str,
    high_col: str,
    low_col: str,
    close_col: str,
    volume_col: str,
) -> pd.DataFrame:
    try:
        df_raw = pd.read_csv(path, sep=None, engine="python")
    except Exception:
        df_raw = pd.read_csv(path)

    date_col = _pick_column(df_raw, date_col, ["date", "<date>"])
    time_col = _pick_column(df_raw, time_col, ["time", "<time>", "datetime", "timestamp"])
    open_col = _pick_column(df_raw, open_col, ["open", "<open>"])
    high_col = _pick_column(df_raw, high_col, ["high", "<high>"])
    low_col = _pick_column(df_raw, low_col, ["low", "<low>"])
    close_col = _pick_column(df_raw, close_col, ["close", "<close>"])
    volume_col = _pick_column(
        df_raw,
        volume_col,
        ["volume", "vol", "tickvol", "tick_volume", "tickvolume", "<tickvol>"],
    )

    missing_cols = [
        name
        for name, value in {
            "open": open_col,
            "high": high_col,
            "low": low_col,
            "close": close_col,
        }.items()
        if value is None
    ]
    if missing_cols:
        raise ValueError(
            f"Missing required OHLC columns: {missing_cols}. Available columns: {list(df_raw.columns)}"
        )

    rename_map = {
        open_col: "open",
        high_col: "high",
        low_col: "low",
        close_col: "close",
    }
    if volume_col:
        rename_map[volume_col] = "tick_volume"

    df = df_raw.rename(columns=rename_map).copy()

    if date_col and time_col:
        df["time"] = pd.to_datetime(
            df_raw[date_col].astype(str).str.strip()
            + " "
            + df_raw[time_col].astype(str).str.strip(),
            errors="coerce",
        )
    elif time_col:
        df["time"] = pd.to_datetime(df_raw[time_col], errors="coerce")
    elif date_col:
        df["time"] = pd.to_datetime(df_raw[date_col], errors="coerce")

    numeric_cols = ["open", "high", "low", "close"]
    if "tick_volume" in df.columns:
        numeric_cols.append("tick_volume")
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    drop_subset = ["open", "high", "low", "close"]
    if "time" in df.columns:
        drop_subset.append("time")
    df = df.dropna(subset=drop_subset)

    if "time" in df.columns:
        df = df.sort_values("time").reset_index(drop=True)
    else:
        df = df.reset_index(drop=True)

    return df


def build_training_frame(df: pd.DataFrame, atr_period: int, rsi_period: int, forward: int):
    atr = AverageTrueRange(
        high=df["high"], low=df["low"], close=df["close"], window=atr_period, fillna=False
    ).average_true_range()
    atr = atr.replace(0.0, np.nan)

    df["feat_body"] = (df["close"] - df["open"]) / atr
    df["feat_range"] = (df["high"] - df["low"]) / atr
    df["feat_rsi"] = RSIIndicator(close=df["close"], window=rsi_period, fillna=False).rsi() / 100.0

    future_close = df["close"].shift(-forward)
    df["target"] = np.where(future_close > df["close"], 1.0, 0.0)
    df.loc[future_close.isna(), "target"] = np.nan

    features = ["feat_body", "feat_range", "feat_rsi"]
    df = df.dropna(subset=features + ["target"]).copy()
    df["target"] = df["target"].astype(np.int64)
    return df, features


def make_windows(df: pd.DataFrame, features: list[str], window: int):
    if len(df) <= window:
        raise ValueError(
            f"Not enough rows after feature generation: rows={len(df)}, window={window}"
        )

    x, y = [], []
    for i in range(window, len(df)):
        x.append(df[features].iloc[i - window : i].values.flatten())
        y.append(df["target"].iloc[i])
    return np.array(x, dtype=np.float32), np.array(y)


def main():
    args = parse_args()
    output_filename = args.output or (Path(args.input).stem + ".onnx")

    print("--- QUICK TRAINING ---")
    print(f"Loading rates from: {args.input}")
    print(f"Output ONNX: {output_filename}")

    df = load_market_csv(
        path=args.input,
        date_col=args.date_col,
        time_col=args.time_col,
        open_col=args.open_col,
        high_col=args.high_col,
        low_col=args.low_col,
        close_col=args.close_col,
        volume_col=args.volume_col,
    )
    print(f"Loaded rows: {len(df)}")

    df, features = build_training_frame(
        df=df,
        atr_period=args.atr_period,
        rsi_period=args.rsi_period,
        forward=args.forward,
    )

    x, y = make_windows(df=df, features=features, window=args.window)

    print("Searching efficient model parameters (Random Search)...")
    param_dist = {
        "n_estimators": [100, 150, 200],
        "max_depth": [5, 8, 12],
        "min_samples_leaf": [1, 5],
    }
    tscv = TimeSeriesSplit(n_splits=args.cv_splits)
    search = RandomizedSearchCV(
        RandomForestClassifier(random_state=args.random_state),
        param_distributions=param_dist,
        n_iter=args.n_iter,
        cv=tscv,
        scoring="accuracy",
        n_jobs=args.jobs,
        random_state=args.random_state,
    )
    search.fit(x, y)
    model = search.best_estimator_
    print(f"Best params: {search.best_params_}")

    input_size = args.window * len(features)
    initial_type = [("float_input", FloatTensorType([None, input_size]))]
    onx = convert_sklearn(
        model,
        initial_types=initial_type,
        target_opset=args.opset,
        options={type(model): {"zipmap": False}},
    )
    onnx.checker.check_model(onx)

    output_path = Path(output_filename)
    if output_path.parent != Path("."):
        output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        f.write(onx.SerializeToString())

    print(f"Model saved to: {output_path}")
    print(f"Opset version: {args.opset}")
    print("--- PROCESS COMPLETED ---")


if __name__ == "__main__":
    main()
