"""
Train MLP or RandomForest on ADX + Stochastic + Volume features and export to ONNX.

Label: ATR-based binary — 1 (buy) if max upside >= min_profit_atr AND upside > downside
       over the next forward_bars candles. 0 otherwise.

Hyperparameter search uses RandomizedSearchCV with TimeSeriesSplit (no data leakage).

Usage:
    python train_adx_stoch_vol.py --input csv/ndx100_rates_m5.csv \
        --symbol ndx100 --timeframe M5 --model rf \
        --window 20 --forward 10 --min_profit_atr 1.5
"""

import argparse
import os
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report

from utils.colors import Colors, colorize as c
from utils.features import (
    add_price_features, add_adx_features, add_stoch_features,
    add_volume_features, add_label, make_windows,
)
from utils.onnx_export import export

FEATURE_COLS = [
    'feat_body', 'feat_range',
    'adx_strength', 'adx_di_signal', 'adx_di_sep', 'adx_momentum', 'adx_regime',
    'stoch_momentum', 'stoch_position', 'stoch_velocity', 'stoch_divergence',
    'vol_ratio', 'vol_momentum', 'vol_price_div', 'vol_percentile', 'vol_zscore',
]


def load(path: str) -> pd.DataFrame:
    df = pd.read_parquet(path) if path.endswith('.parquet') else pd.read_csv(path)
    df['time'] = pd.to_datetime(df['time'])
    return df.sort_values('time').reset_index(drop=True)


def build_model(model_type: str, n_iter: int, jobs: int):
    tscv = TimeSeriesSplit(n_splits=3)
    if model_type == 'rf':
        param_dist = {
            'n_estimators':      [100, 150, 200, 250],
            'max_depth':         [5, 8, 12, 15],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf':  [1, 2, 4],
            'max_features':      ['sqrt', 'log2', None],
        }
        estimator = RandomForestClassifier(random_state=42, class_weight='balanced')
        return RandomizedSearchCV(estimator, param_dist, n_iter=n_iter,
                                  cv=tscv, scoring='balanced_accuracy',
                                  n_jobs=jobs, verbose=1)
    else:
        # MLP: wrap in Pipeline with scaler, no hyperparam search
        return Pipeline([
            ('scaler', StandardScaler()),
            ('clf', MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=500,
                                  random_state=42, class_weight='balanced' if hasattr(MLPClassifier, 'class_weight') else None)),
        ])


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--input',          required=True,              help='CSV or Parquet input path')
    parser.add_argument('--symbol',         required=True,              help='Symbol name for output filename')
    parser.add_argument('--timeframe',      required=True,              help='Timeframe: M1 M5 M15 M30 H1 H4 D1')
    parser.add_argument('--model',          default='rf', choices=['mlp', 'rf'], help='Model type')
    parser.add_argument('--window',         type=int,   default=20,     help='Window size')
    parser.add_argument('--forward',        type=int,   default=10,     help='Forward bars for label')
    parser.add_argument('--min_profit_atr', type=float, default=1.5,    help='Min upside in ATR units to label as buy')
    parser.add_argument('--atr_period',     type=int,   default=14,     help='ATR period')
    parser.add_argument('--adx_period',     type=int,   default=8,      help='ADX period')
    parser.add_argument('--adx_min',        type=float, default=20.0,   help='ADX minimum threshold')
    parser.add_argument('--stoch_k',        type=int,   default=14,     help='Stochastic K period')
    parser.add_argument('--stoch_d',        type=int,   default=3,      help='Stochastic D period')
    parser.add_argument('--vol_window',     type=int,   default=20,     help='Volume rolling window')
    parser.add_argument('--n_iter',         type=int,   default=10,     help='RandomizedSearchCV iterations (rf only)')
    parser.add_argument('--jobs',           type=int,   default=3,      help='Parallel jobs (rf only)')
    parser.add_argument('--output',         default=None,               help='ONNX output path (auto-generated if omitted)')
    args = parser.parse_args()

    t0 = time.time()
    print(c("=" * 60, Colors.CYAN))
    print(c(f"Training {args.model.upper()} — ADX + Stoch + Volume (16 features)", Colors.CYAN))
    print(c("=" * 60, Colors.CYAN))

    df = load(args.input)
    df = add_price_features(df, args.atr_period)
    df = add_adx_features(df, args.adx_period, args.adx_min)
    df = add_stoch_features(df, args.stoch_k, args.stoch_d)
    df = add_volume_features(df, args.vol_window)
    df.dropna(inplace=True)
    df = add_label(df, args.forward, args.min_profit_atr)

    n_classes = df['label'].nunique()
    print(c("\nClass distribution:", Colors.WHITE))
    dist = df['label'].value_counts(normalize=True)
    for cls, pct in dist.items():
        color = Colors.GREEN if cls == 1 else Colors.YELLOW
        print(c(f"  {cls}: {pct:.2%}", color))

    if n_classes < 2:
        print(c("\n✗ Only 1 class — adjust --min_profit_atr or --forward and retry.", Colors.RED))
        return

    X, y = make_windows(df, FEATURE_COLS, args.window)
    assert np.isfinite(X).all(), "X contains non-finite values after cleaning"

    split    = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    print(c(f"\nTrain: {len(X_train)}  Test: {len(X_test)}  Features: {len(FEATURE_COLS)}  Input shape: {X.shape}", Colors.WHITE))

    model = build_model(args.model, args.n_iter, args.jobs)
    model.fit(X_train, y_train)

    if hasattr(model, 'best_params_'):
        print(c(f"\nBest params: {model.best_params_}", Colors.YELLOW))
        print(c(f"Best CV score: {model.best_score_:.4f}", Colors.YELLOW))

    print(c("\nTest set report:", Colors.CYAN))
    print(classification_report(y_test, model.predict(X_test)))

    n_feat   = len(FEATURE_COLS)
    out_path = args.output or os.path.join(
        'onnx', f"{args.symbol}_{args.timeframe.lower()}_{n_feat}_feat_adx_stoch_vol.onnx"
    )
    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
    # export the fitted estimator, not the search wrapper
    fitted = model.best_estimator_ if hasattr(model, 'best_estimator_') else model
    export(fitted, FEATURE_COLS, args.window, out_path)

    elapsed = int(time.time() - t0)
    print(c(f"\n✓ Done in {elapsed // 60}m {elapsed % 60}s", Colors.GREEN))


if __name__ == '__main__':
    main()
