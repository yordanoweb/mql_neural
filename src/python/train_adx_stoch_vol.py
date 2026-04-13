"""
Train MLP or RandomForest on ADX + Stochastic + Volume features and export to ONNX.

Usage:
    python train_adx_stoch_vol.py --input csv/ndx100_rates_m5.csv \
        --symbol ndx100 --timeframe M5 --model mlp \
        --window 20 --forward 1 --min_pct 0.0
"""

import argparse
import os

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from utils.features import (
    add_adx_features, add_stoch_features, add_volume_features,
    add_label, make_windows,
)
from utils.onnx_export import export

FEATURE_COLS = [
    'adx_norm', 'dip_norm', 'din_norm',
    'stoch_k', 'stoch_d', 'stoch_diff', 'stoch_signal',
    'vol_norm', 'vol_change', 'vol_ma_ratio', 'obv_norm', 'vol_spike',
]


def load(path: str) -> pd.DataFrame:
    df = pd.read_parquet(path) if path.endswith('.parquet') else pd.read_csv(path)
    df['time'] = pd.to_datetime(df['time'])
    return df.sort_values('time').reset_index(drop=True)


def build_model(model_type: str):
    estimator = (
        MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=300, random_state=42)
        if model_type == 'mlp'
        else RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=42)
    )
    return Pipeline([('scaler', StandardScaler()), ('clf', estimator)])


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--input',      required=True,  help='CSV or Parquet input path')
    parser.add_argument('--symbol',     required=True,  help='Symbol name for output filename')
    parser.add_argument('--timeframe',  required=True,  help='Timeframe: M1 M5 M15 M30 H1 H4 D1')
    parser.add_argument('--model',      default='mlp',  choices=['mlp', 'rf'], help='Model type')
    parser.add_argument('--window',     type=int,   default=20,  help='Window size')
    parser.add_argument('--forward',    type=int,   default=1,   help='Forward bars for label')
    parser.add_argument('--min_pct',    type=float, default=0.0, help='Min price move %% to count as signal')
    parser.add_argument('--adx_period', type=int,   default=14,  help='ADX indicator period')
    parser.add_argument('--stoch_k',    type=int,   default=10,  help='Stochastic K period')
    parser.add_argument('--stoch_d',    type=int,   default=3,   help='Stochastic D period')
    parser.add_argument('--vol_window', type=int,   default=10,  help='Volume rolling window')
    parser.add_argument('--output',     default=None,            help='ONNX output path (auto-generated if omitted)')
    args = parser.parse_args()

    df = load(args.input)
    df = add_adx_features(df, args.adx_period)
    df = add_stoch_features(df, args.stoch_k, args.stoch_d)
    df = add_volume_features(df, args.vol_window)
    df = add_label(df, args.forward, args.min_pct)
    df.dropna(inplace=True)

    print("\nClass distribution:")
    print(df['label'].value_counts(normalize=True).to_string())

    X, y = make_windows(df, FEATURE_COLS, args.window)

    split = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    model = build_model(args.model)
    model.fit(X_train, y_train)
    print("\n" + classification_report(y_test, model.predict(X_test)))

    n_feat   = len(FEATURE_COLS)
    out_path = args.output or os.path.join(
        'onnx', f"{args.symbol}_{args.timeframe.lower()}_{n_feat}_feat_adx_stoch_vol.onnx"
    )
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    export(model, FEATURE_COLS, args.window, out_path)


if __name__ == '__main__':
    main()
