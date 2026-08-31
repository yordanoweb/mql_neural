#!/usr/bin/env python3
"""
train_ensemble_buy_only.py
=================
Genera un ensemble de 5 modelos ONNX buy-only con validación OOS cronológica.
Basado en train_buy_only.txt, extendido con múltiples perspectivas y métricas OOS.

Uso básico (genera los 5 modelos de una vez):
    python train_ensemble.py --input-csv EURUSD_M5.csv --output-dir ./models --forward 10

Uso para un solo modelo:
    python train_ensemble.py --input-csv EURUSD_M5.csv --output-dir ./models \
        --model-id B --window 15 --feature-set standard --forward 10

El script produce:
    - model_A_impulse_SYMBOL.onnx
    - model_B_swing_SYMBOL.onnx
    - model_C_trend_SYMBOL.onnx
    - model_D_structure_SYMBOL.onnx
    - model_E_volatility_SYMBOL.onnx
    - ensemble_report_SYMBOL.json
    - ensemble_threshold_report.json (sweep OOS 0.50..0.80)
    - ensemble_thresholds.csv (mismo sweep en formato tabular)
"""

import pandas as pd
import numpy as np
import argparse
import os
import re
import json
from collections import Counter
from datetime import datetime, timezone
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.metrics import (
    balanced_accuracy_score, precision_score, recall_score,
    roc_auc_score, confusion_matrix
)
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
import onnx

# ---------------------------------------------------------------------------
# Fallback para el módulo indicators (si no está disponible, usa implementación
# interna simple de RSI)
# ---------------------------------------------------------------------------
try:
    from indicators import calculate_rsi as _rsi_external
    def _rsi_impl(prices, period):
        return _rsi_external(prices, period)
except ImportError:
    def _rsi_impl(prices, period):
        """RSI simple compatible con lista de floats."""
        s = pd.Series(prices, dtype=float)
        delta = s.diff()
        gain = delta.clip(lower=0)
        loss = (-delta).clip(lower=0)
        avg_gain = gain.ewm(alpha=1.0/period, min_periods=period, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1.0/period, min_periods=period, adjust=False).mean()
        rs = avg_gain / (avg_loss + 1e-12)
        rsi = 100.0 - (100.0 / (1.0 + rs))
        return rsi.fillna(50.0).tolist()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train buy-only RandomForest ensemble with OOS validation and export ONNX.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # --- Inputs / Outputs ---
    parser.add_argument("--input-csv", required=True, help="Input OHLC CSV file path")
    parser.add_argument("--output-dir", required=True, help="Directory to save ONNX models and report")
    parser.add_argument("--output-prefix", default="model", help="Prefix for ONNX filenames")

    # --- Modelo individual vs batch ---
    parser.add_argument("--model-id", default="all",
                        choices=["all", "A", "B", "C", "D", "E"],
                        help="Which model to train: 'all' runs the full ensemble")
    parser.add_argument("--window", type=int, default=None,
                        help="Override window size (only used with --model-id != all)")
    parser.add_argument("--feature-set", default=None,
                        choices=["standard", "structure", "volatility"],
                        help="Override feature set (only used with --model-id != all)")

    # --- Parámetros de entrenamiento ---
    parser.add_argument("--forward", type=int, default=10,
                        help="Forward bars for buy label: close[t+forward] > close[t]")
    parser.add_argument("--rsi-period", type=int, default=14, help="RSI period")
    parser.add_argument("--test-size", type=float, default=0.30,
                        help="Fraction of most recent data reserved for OOS test")
    parser.add_argument("--n-iter", type=int, default=5,
                        help="RandomizedSearchCV iterations")
    parser.add_argument("--n-splits", type=int, default=3,
                        help="TimeSeriesSplit folds for cross-validation")
    parser.add_argument("--n-jobs", type=int, default=-1,
                        help="Parallel jobs for search")

    # --- Columnas CSV ---
    parser.add_argument("--symbol", default="auto", help="Trading symbol for metadata")
    parser.add_argument("--timeframe", default="auto", help="Timeframe for metadata")
    parser.add_argument("--date-col", default="date", help="Date column name")
    parser.add_argument("--time-col", default="time", help="Time column name")
    parser.add_argument("--open-col", default="open", help="Open price column")
    parser.add_argument("--high-col", default="high", help="High price column")
    parser.add_argument("--low-col", default="low", help="Low price column")
    parser.add_argument("--close-col", default="close", help="Close price column")
    parser.add_argument("--volume-col", default="tick_volume", help="Volume column")

    return parser.parse_args()


# =============================================================================
# CONFIGURACIÓN DE LOS 5 MODELOS DEL ENSEMBLE
# =============================================================================

ENSEMBLE_CONFIG = {
    "A": {
        "alias": "impulse",
        "window": 5,
        "feature_set": "standard",
        "perspective": "microstructure",
        "description": "Movimiento inmediato. Captura microestructura y momentum instantaneo."
    },
    "B": {
        "alias": "swing",
        "window": 15,
        "feature_set": "standard",
        "perspective": "short_term",
        "description": "Contexto corto. Sweet spot intradia. Equilibrio ruido/senal."
    },
    "C": {
        "alias": "trend",
        "window": 30,
        "feature_set": "standard",
        "perspective": "medium_term",
        "description": "Contexto medio. Identifica direccion dominante en tendencias claras."
    },
    "D": {
        "alias": "structure",
        "window": 20,
        "feature_set": "structure",
        "perspective": "scale_invariant",
        "description": "Patron independiente de escala. Usa ratios body/range y ATR-normalizados."
    },
    "E": {
        "alias": "volatility",
        "window": 20,
        "feature_set": "volatility",
        "perspective": "volatility_regime",
        "description": "Regimen de volatilidad. Detecta expansion/contraccion para timing de entrada."
    },
}


# =============================================================================
# FUNCIONES AUXILIARES
# =============================================================================

def infer_symbol_timeframe_from_filename(path):
    name = os.path.splitext(os.path.basename(path))[0]
    parts = name.split("_")
    tf_regex = re.compile(
        r"^(M1|M2|M3|M4|M5|M6|M10|M12|M15|M20|M30|H1|H2|H3|H4|H6|H8|H12|D1|W1|MN1)$"
    )
    inferred_symbol = parts[0] if len(parts) > 0 else "unknown"
    inferred_timeframe = "unknown"
    for part in parts[1:]:
        if tf_regex.match(part):
            inferred_timeframe = part
            break
    return inferred_symbol, inferred_timeframe


def resolve_column(df, preferred, fallback):
    if preferred in df.columns:
        return preferred
    if fallback in df.columns:
        return fallback
    return None


def calculate_rsi(series, period=14):
    rsi_list = _rsi_impl(series.values.tolist(), period)
    return pd.Series(rsi_list, index=series.index)


def set_onnx_metadata(model, metadata):
    for key, value in metadata.items():
        value_str = json.dumps(value, sort_keys=True) if isinstance(value, (dict, list)) else str(value)
        updated = False
        for prop in model.metadata_props:
            if prop.key == key:
                prop.value = value_str
                updated = True
                break
        if not updated:
            prop = model.metadata_props.add()
            prop.key = key
            prop.value = value_str


def build_features(df, feature_set, open_col, high_col, low_col, close_col, rsi_period):
    """
    Genera el DataFrame de features segun el feature_set seleccionado.
    Devuelve (df_features, feature_names).
    """
    df = df.copy()

    if feature_set == "standard":
        df['feat_body'] = df[close_col] - df[open_col]
        df['feat_range'] = df[high_col] - df[low_col]
        df['feat_rsi'] = calculate_rsi(df[close_col], rsi_period) / 100.0
        feature_names = ['feat_body', 'feat_range', 'feat_rsi']

    elif feature_set == "structure":
        df['feat_body_ratio'] = (df[close_col] - df[open_col]) / (df[high_col] - df[low_col] + 1e-9)
        # ATR simple (media movil del rango)
        df['atr_20'] = (df[high_col] - df[low_col]).rolling(window=20, min_periods=1).mean()
        df['feat_range_norm'] = (df[high_col] - df[low_col]) / (df['atr_20'] + 1e-9)
        df['feat_rsi'] = calculate_rsi(df[close_col], rsi_period) / 100.0
        feature_names = ['feat_body_ratio', 'feat_range_norm', 'feat_rsi']

    elif feature_set == "volatility":
        df['feat_range'] = df[high_col] - df[low_col]
        df['feat_range_shift'] = df['feat_range'].shift(1)
        df['feat_range_expansion'] = df['feat_range'] / (df['feat_range_shift'] + 1e-9)
        df['feat_range_of_range'] = df['feat_range'].rolling(window=5, min_periods=1).max() - \
                                       df['feat_range'].rolling(window=5, min_periods=1).min()
        df['feat_rsi'] = calculate_rsi(df[close_col], rsi_period) / 100.0
        feature_names = ['feat_range', 'feat_range_expansion', 'feat_rsi']

    else:
        raise ValueError(f"Unknown feature_set: {feature_set}")

    return df, feature_names


def create_windows(df, features, window, forward, close_col):
    """
    Crea ventanas deslizantes y target.
    Devuelve (X, y, indices).
    """
    df = df.copy()
    df['target'] = (df[close_col].shift(-forward) > df[close_col]).astype(int)
    df.dropna(subset=['target'] + features, inplace=True)

    X, y, indices = [], [], []
    for i in range(window, len(df)):
        # Asegurar que tenemos suficientes datos hacia adelante para el target
        if i + forward >= len(df) + window:
            break
        window_data = df[features].iloc[i - window:i].values.flatten()
        X.append(window_data)
        y.append(df['target'].iloc[i])
        indices.append(df.index[i])

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int64)
    return X, y, indices


def evaluate_model(model, X, y):
    """
    Evalua un modelo entrenado sobre un conjunto de datos.
    Devuelve dict con metricas.
    """
    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)

    # Manejar caso de una sola clase en y
    try:
        ba = balanced_accuracy_score(y, y_pred)
    except Exception:
        ba = None

    try:
        prec = precision_score(y, y_pred, pos_label=1, zero_division=0)
    except Exception:
        prec = None

    try:
        rec = recall_score(y, y_pred, pos_label=1, zero_division=0)
    except Exception:
        rec = None

    try:
        roc = roc_auc_score(y, y_proba[:, 1])
    except Exception:
        roc = None

    cm = confusion_matrix(y, y_pred, labels=[0, 1]).tolist()

    pred_counts = Counter(y_pred.tolist())
    signal_count = int(pred_counts.get(1, 0))
    total = len(y_pred)

    # Confidence stats for BUY predictions
    buy_mask = y_pred == 1
    avg_confidence_buy = float(y_proba[buy_mask, 1].mean()) if buy_mask.any() else 0.0
    max_confidence_buy = float(y_proba[buy_mask, 1].max()) if buy_mask.any() else 0.0
    min_confidence_buy = float(y_proba[buy_mask, 1].min()) if buy_mask.any() else 0.0

    return {
        "balanced_accuracy": float(ba) if ba is not None else None,
        "precision_buy": float(prec) if prec is not None else None,
        "recall_buy": float(rec) if rec is not None else None,
        "roc_auc": float(roc) if roc is not None else None,
        "confusion_matrix": cm,
        "signals_buy_count": signal_count,
        "signals_buy_pct": round(100.0 * signal_count / total, 2) if total > 0 else 0.0,
        "avg_confidence_buy": round(avg_confidence_buy, 6),
        "max_confidence_buy": round(max_confidence_buy, 6),
        "min_confidence_buy": round(min_confidence_buy, 6),
        "total_samples": total,
        "class_distribution": dict(Counter(y.tolist())),
        "prediction_distribution": dict(pred_counts),
    }



def analyze_ensemble_thresholds(results, start=0.50, stop=0.80, step=0.01):
    """
    Analiza el ensemble sobre la intersección temporal OOS de todos los modelos.

    La probabilidad BUY del ensemble es la media de las probabilidades BUY
    producidas por los modelos que tienen predicción en cada timestamp común.
    Evalúa thresholds desde start hasta stop (inclusive).
    """
    valid = [
        r for r in results
        if "error" not in r
        and r.get("oos_indices")
        and r.get("oos_y") is not None
        and r.get("oos_proba_buy") is not None
    ]

    if len(valid) < 2:
        return {
            "status": "FAILED",
            "error": "Need at least 2 successfully trained models with OOS predictions."
        }

    # Mapear cada modelo por timestamp OOS.
    maps = []
    for r in valid:
        maps.append({
            int(idx): (int(y), float(p))
            for idx, y, p in zip(
                r["oos_indices"], r["oos_y"], r["oos_proba_buy"]
            )
        })

    common_indices = set(maps[0].keys())
    for m in maps[1:]:
        common_indices &= set(m.keys())

    common_indices = sorted(common_indices)

    if not common_indices:
        return {
            "status": "FAILED",
            "error": "No common OOS timestamps found between ensemble models."
        }

    # El target es el mismo para todos; tomamos el del primer modelo.
    y = np.array([maps[0][idx][0] for idx in common_indices], dtype=np.int64)

    # Media simple de probabilidades BUY.
    ensemble_proba = np.mean(
        np.array([[m[idx][1] for m in maps] for idx in common_indices], dtype=np.float64),
        axis=1
    )

    thresholds = []
    t = float(start)
    while t <= float(stop) + 1e-9:
        pred = (ensemble_proba >= t).astype(int)
        signal_count = int(pred.sum())
        total = len(pred)

        precision = precision_score(y, pred, pos_label=1, zero_division=0)
        recall = recall_score(y, pred, pos_label=1, zero_division=0)
        ba = balanced_accuracy_score(y, pred)

        thresholds.append({
            "threshold": round(t, 2),
            "precision_buy": round(float(precision), 6),
            "recall_buy": round(float(recall), 6),
            "balanced_accuracy": round(float(ba), 6),
            "signals_buy_count": signal_count,
            "signals_buy_pct": round(100.0 * signal_count / total, 2),
            "avg_confidence_buy": round(
                float(ensemble_proba[pred == 1].mean()), 6
            ) if signal_count > 0 else 0.0,
        })
        t += float(step)

    # Referencia matemática: threshold con mejor balanced accuracy.
    # No se usa para modificar automáticamente el EA; sólo se reporta.
    best_ba = max(thresholds, key=lambda x: x["balanced_accuracy"])

    return {
        "status": "OK",
        "aggregation": "simple_mean_probability_buy",
        "models_used": [r["model_id"] for r in valid],
        "common_oos_samples": len(common_indices),
        "threshold_range": {
            "start": start,
            "stop": stop,
            "step": step
        },
        "class_distribution": dict(Counter(y.tolist())),
        "thresholds": thresholds,
        "best_balanced_accuracy_threshold": best_ba["threshold"],
        "best_balanced_accuracy": best_ba["balanced_accuracy"],
    }


def train_single_model(
    csv_file,
    output_path,
    model_id,
    config,
    forward,
    rsi_period,
    test_size,
    n_iter,
    n_splits,
    n_jobs,
    symbol,
    timeframe,
    date_col,
    time_col,
    open_col_pref,
    high_col_pref,
    low_col_pref,
    close_col_pref,
    volume_col_pref,
):
    """
    Entrena un unico modelo del ensemble y exporta ONNX.
    Devuelve dict con metricas y metadatos.
    """
    print(f"\n{'='*60}")
    print(f"  TRAINING MODEL {model_id} — {config['alias'].upper()}")
    print(f"  Perspective: {config['perspective']}")
    print(f"  Window: {config['window']} | Features: {config['feature_set']}")
    print(f"{'='*60}")

    if not os.path.exists(csv_file):
        raise FileNotFoundError(f"Input CSV not found: {csv_file}")

    # Validaciones
    if config['window'] <= 0:
        raise ValueError("window must be > 0")
    if forward <= 0:
        raise ValueError("forward must be > 0")
    if rsi_period <= 0:
        raise ValueError("rsi-period must be > 0")
    if n_iter <= 0:
        raise ValueError("n-iter must be > 0")
    if n_splits < 2:
        raise ValueError("n-splits must be >= 2")
    if not (0.0 < test_size < 1.0):
        raise ValueError("test-size must be between 0 and 1")

    # Cargar datos
    df = pd.read_csv(csv_file)
    records_loaded = len(df)
    print(f"Records loaded: {records_loaded}")

    # Resolver columnas
    open_col = resolve_column(df, open_col_pref, "open")
    high_col = resolve_column(df, high_col_pref, "high")
    low_col = resolve_column(df, low_col_pref, "low")
    close_col = resolve_column(df, close_col_pref, "close")

    required = [open_col, high_col, low_col, close_col]
    missing = [c for c in required if c is None or c not in df.columns]
    if missing:
        raise ValueError(f"Missing OHLC columns: {missing}. Available: {list(df.columns)}")

    # Feature engineering
    df, features = build_features(
        df, config['feature_set'], open_col, high_col, low_col, close_col, rsi_period
    )
    print(f"Features: {features}")

    # Crear ventanas
    X, y, indices = create_windows(df, features, config['window'], forward, close_col)
    print(f"Total samples (windows): {len(X)}")

    if len(X) == 0:
        raise ValueError("No samples generated. Check window/forward vs data length.")

    class_counts = Counter(y.tolist())
    print(f"Class distribution: {dict(class_counts)}")
    if len(class_counts) < 2:
        raise ValueError(f"Only one class present: {dict(class_counts)}")

    # Split cronologico OOS
    split_idx = int(len(X) * (1.0 - test_size))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    print(f"Train samples: {len(X_train)} | Test samples: {len(X_test)}")
    print(f"Train class dist: {dict(Counter(y_train.tolist()))}")
    print(f"Test class dist:  {dict(Counter(y_test.tolist()))}")

    # Entrenamiento con RandomizedSearchCV + TimeSeriesSplit
    print("Running RandomizedSearchCV (TimeSeriesSplit)...")
    param_dist = {
        'n_estimators': [100, 150, 200, 250],
        'max_depth': [5, 8, 12, 16, None],
        'min_samples_leaf': [1, 2, 5, 10],
        'class_weight': [None, 'balanced', 'balanced_subsample']
    }

    tscv = TimeSeriesSplit(n_splits=n_splits)
    search = RandomizedSearchCV(
        RandomForestClassifier(random_state=42),
        param_distributions=param_dist,
        n_iter=n_iter,
        cv=tscv,
        scoring='balanced_accuracy',
        n_jobs=n_jobs,
        random_state=42,
        refit=True
    )
    search.fit(X_train, y_train)
    model = search.best_estimator_
    print(f"Best params: {search.best_params_}")
    print(f"Best CV balanced_accuracy: {search.best_score_:.4f}")

    # Evaluacion TRAIN (fit, solo referencia)
    train_metrics = evaluate_model(model, X_train, y_train)
    print(f"TRAIN  — BA: {train_metrics['balanced_accuracy']:.4f} | "
          f"Precision: {train_metrics['precision_buy']:.4f} | "
          f"Recall: {train_metrics['recall_buy']:.4f}")

    # Evaluacion OOS TEST (la importante)
    test_metrics = evaluate_model(model, X_test, y_test)
    print(f"TEST   — BA: {test_metrics['balanced_accuracy']:.4f} | "
          f"Precision: {test_metrics['precision_buy']:.4f} | "
          f"Recall: {test_metrics['recall_buy']:.4f} | "
          f"ROC-AUC: {test_metrics['roc_auc']:.4f}")
    print(f"TEST signals BUY: {test_metrics['signals_buy_count']} "
          f"({test_metrics['signals_buy_pct']}%)")

    # Exportar ONNX
    input_size = config['window'] * len(features)
    initial_type = [('float_input', FloatTensorType([None, input_size]))]
    onx = convert_sklearn(
        model,
        initial_types=initial_type,
        target_opset=12,
        options={type(model): {'zipmap': False}}
    )
    onnx.checker.check_model(onx)

    # Verificar output de probabilidades
    prob_output = next((o for o in onx.graph.output if o.name == 'probabilities'), None)
    if prob_output is None:
        raise RuntimeError("ONNX export missing 'probabilities' output")
    prob_shape = [d.dim_value if d.dim_value else d.dim_param
                  for d in prob_output.type.tensor_type.shape.dim]
    if len(prob_shape) != 2 or prob_shape[1] != 2:
        raise RuntimeError(f"Expected probabilities shape [*, 2], got {prob_shape}")
    print(f"ONNX probabilities shape: {prob_shape}")

    # Metadata enriquecida
    metadata = {
        "ensemble.id": model_id,
        "ensemble.alias": config['alias'],
        "ensemble.perspective": config['perspective'],
        "ensemble.description": config['description'],
        "ensemble.window": config['window'],
        "ensemble.feature_set": config['feature_set'],
        "ensemble.features": features,
        "ensemble.feature_count": len(features),
        "ensemble.input_size": input_size,
        "training.created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "training.input_csv_path": os.path.abspath(csv_file),
        "training.input_csv_name": os.path.basename(csv_file),
        "training.output_filename": os.path.basename(output_path),
        "training.symbol": symbol,
        "training.timeframe": timeframe,
        "training.forward": forward,
        "training.rsi_period": rsi_period,
        "training.test_size": test_size,
        "training.n_iter": n_iter,
        "training.n_splits": n_splits,
        "training.n_jobs": n_jobs,
        "training.records_loaded": int(records_loaded),
        "training.samples_total": int(len(X)),
        "training.samples_train": int(len(X_train)),
        "training.samples_test": int(len(X_test)),
        "training.target": f"close[t+{forward}] > close[t]",
        "training.target_class_distribution": dict(class_counts),
        "training.model_type": "RandomForestClassifier",
        "training.random_state": 42,
        "training.best_params": search.best_params_,
        "training.best_cv_score": float(search.best_score_),
        "training.scoring": "balanced_accuracy",
        "validation.train_balanced_accuracy": train_metrics['balanced_accuracy'],
        "validation.train_precision_buy": train_metrics['precision_buy'],
        "validation.train_recall_buy": train_metrics['recall_buy'],
        "validation.test_balanced_accuracy": test_metrics['balanced_accuracy'],
        "validation.test_precision_buy": test_metrics['precision_buy'],
        "validation.test_recall_buy": test_metrics['recall_buy'],
        "validation.test_roc_auc": test_metrics['roc_auc'],
        "validation.test_signals_buy_count": test_metrics['signals_buy_count'],
        "validation.test_signals_buy_pct": test_metrics['signals_buy_pct'],
        "validation.test_avg_confidence_buy": test_metrics['avg_confidence_buy'],
        "validation.test_confusion_matrix": test_metrics['confusion_matrix'],
        "validation.test_class_distribution": test_metrics['class_distribution'],
    }
    set_onnx_metadata(onx, metadata)

    # Guardar ONNX
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "wb") as f:
        f.write(onx.SerializeToString())
    print(f"ONNX saved: {output_path}")

    return {
        "model_id": model_id,
        "alias": config['alias'],
        "perspective": config['perspective'],
        "window": config['window'],
        "feature_set": config['feature_set'],
        "features": features,
        "input_size": input_size,
        "output_path": output_path,
        "best_params": search.best_params_,
        "best_cv_score": float(search.best_score_),
        "train_metrics": train_metrics,
        "test_metrics": test_metrics,

        # Datos OOS internos para el análisis posterior del ensemble.
        # Se mantienen fuera del JSON principal para no inflarlo.
        "oos_indices": [int(i) for i in indices[split_idx:]],
        "oos_y": y_test.tolist(),
        "oos_proba_buy": model.predict_proba(X_test)[:, 1].astype(float).tolist(),
    }


def main():
    args = parse_args()

    # Inferir simbolo/timeframe
    inferred_symbol, inferred_timeframe = infer_symbol_timeframe_from_filename(args.input_csv)
    resolved_symbol = inferred_symbol if args.symbol == "auto" else args.symbol
    resolved_timeframe = inferred_timeframe if args.timeframe == "auto" else args.timeframe

    print("=" * 70)
    print("  ENSEMBLE BUY-ONLY TRAINING")
    print(f"  CSV: {args.input_csv}")
    print(f"  Output dir: {args.output_dir}")
    print(f"  Symbol: {resolved_symbol} | Timeframe: {resolved_timeframe}")
    print(f"  Forward: {args.forward} | Test size: {args.test_size}")
    print("=" * 70)

    os.makedirs(args.output_dir, exist_ok=True)

    results = []

    if args.model_id == "all":
        model_ids = list(ENSEMBLE_CONFIG.keys())
    else:
        model_ids = [args.model_id]

    for mid in model_ids:
        config = ENSEMBLE_CONFIG[mid].copy()

        # Overrides para modelo individual
        if args.model_id != "all":
            if args.window is not None:
                config['window'] = args.window
            if args.feature_set is not None:
                config['feature_set'] = args.feature_set

        output_filename = f"{args.output_prefix}_{mid}_{config['alias']}_buy_{resolved_symbol}.onnx"
        output_path = os.path.join(args.output_dir, output_filename)

        try:
            result = train_single_model(
                csv_file=args.input_csv,
                output_path=output_path,
                model_id=mid,
                config=config,
                forward=args.forward,
                rsi_period=args.rsi_period,
                test_size=args.test_size,
                n_iter=args.n_iter,
                n_splits=args.n_splits,
                n_jobs=args.n_jobs,
                symbol=resolved_symbol,
                timeframe=resolved_timeframe,
                date_col=args.date_col,
                time_col=args.time_col,
                open_col_pref=args.open_col,
                high_col_pref=args.high_col,
                low_col_pref=args.low_col,
                close_col_pref=args.close_col,
                volume_col_pref=args.volume_col,
            )
            results.append(result)
        except Exception as e:
            print(f"\n*** ERROR training model {mid}: {e} ***")
            results.append({
                "model_id": mid,
                "alias": config['alias'],
                "error": str(e),
            })

    # Generar reporte comparativo
    if len(results) > 1 or (len(results) == 1 and "error" not in results[0]):
        report = {
            "report_generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "input_csv": os.path.abspath(args.input_csv),
            "symbol": resolved_symbol,
            "timeframe": resolved_timeframe,
            "forward": args.forward,
            "test_size": args.test_size,
            "models": []
        }

        for r in results:
            if "error" in r:
                report["models"].append({
                    "model_id": r["model_id"],
                    "alias": r["alias"],
                    "status": "FAILED",
                    "error": r["error"],
                })
            else:
                report["models"].append({
                    "model_id": r["model_id"],
                    "alias": r["alias"],
                    "perspective": r["perspective"],
                    "window": r["window"],
                    "feature_set": r["feature_set"],
                    "features": r["features"],
                    "input_size": r["input_size"],
                    "output_path": r["output_path"],
                    "status": "OK",
                    "best_cv_score": r["best_cv_score"],
                    "test_balanced_accuracy": r["test_metrics"]["balanced_accuracy"],
                    "test_precision_buy": r["test_metrics"]["precision_buy"],
                    "test_recall_buy": r["test_metrics"]["recall_buy"],
                    "test_roc_auc": r["test_metrics"]["roc_auc"],
                    "test_signals_buy_count": r["test_metrics"]["signals_buy_count"],
                    "test_signals_buy_pct": r["test_metrics"]["signals_buy_pct"],
                    "test_avg_confidence_buy": r["test_metrics"]["avg_confidence_buy"],
                })

        report_path = os.path.join(args.output_dir, "ensemble_report_buy_only.json")
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2, default=str)
        print(f"\n{'='*70}")
        print(f"  ENSEMBLE REPORT saved: {report_path}")
        print(f"{'='*70}")

        # -------------------------------------------------------------------
        # ANALISIS ADICIONAL DE THRESHOLD DEL ENSEMBLE
        # -------------------------------------------------------------------
        # Se analiza únicamente la intersección temporal OOS de los modelos
        # entrenados correctamente. La probabilidad ensemble es la media simple
        # de las probabilidades BUY de los modelos.
        ensemble_analysis = analyze_ensemble_thresholds(results)

        threshold_report_path = os.path.join(
            args.output_dir, "ensemble_threshold_report_buy_only.json"
        )
        with open(threshold_report_path, "w") as f:
            json.dump(ensemble_analysis, f, indent=2, default=str)

        print(f"  ENSEMBLE THRESHOLD REPORT saved: {threshold_report_path}")

        if ensemble_analysis.get("status") == "OK":
            threshold_rows = ensemble_analysis["thresholds"]

            # CSV adicional, cómodo para Excel/pandas.
            threshold_csv_path = os.path.join(
                args.output_dir, "ensemble_thresholds_buy_only.csv"
            )
            pd.DataFrame(threshold_rows).to_csv(
                threshold_csv_path, index=False
            )

            print(f"  ENSEMBLE THRESHOLD CSV saved: {threshold_csv_path}")
            print(
                f"  Common OOS samples: "
                f"{ensemble_analysis['common_oos_samples']}"
            )
            print(
                f"  Best BA threshold (reference only): "
                f"{ensemble_analysis['best_balanced_accuracy_threshold']:.2f} "
                f"(BA={ensemble_analysis['best_balanced_accuracy']:.4f})"
            )

            print("\n  ENSEMBLE THRESHOLD SWEEP (OOS)")
            print(
                f"  {'Thr':<6} {'Prec':<8} {'Recall':<8} "
                f"{'BA':<8} {'Signals':<9} {'%':<7} {'AvgConf':<9}"
            )
            print("  " + "-" * 62)

            for row in threshold_rows:
                print(
                    f"  {row['threshold']:<6.2f} "
                    f"{row['precision_buy']:<8.3f} "
                    f"{row['recall_buy']:<8.3f} "
                    f"{row['balanced_accuracy']:<8.3f} "
                    f"{row['signals_buy_count']:<9} "
                    f"{row['signals_buy_pct']:<7.2f} "
                    f"{row['avg_confidence_buy']:<9.3f}"
                )
        else:
            print(
                f"  ENSEMBLE THRESHOLD ANALYSIS FAILED: "
                f"{ensemble_analysis.get('error', 'unknown error')}"
            )

        # Tabla resumen en consola
        print("\n  SUMMARY TABLE (OOS Test Metrics)")
        print(f"  {'ID':<4} {'Alias':<12} {'Window':<7} {'FeatureSet':<12} {'BA':<7} {'Prec':<7} {'Rec':<7} {'ROC':<7} {'Signals':<10}")
        print("  " + "-" * 78)
        for m in report["models"]:
            if m.get("status") == "OK":
                print(f"  {m['model_id']:<4} {m['alias']:<12} {m['window']:<7} "
                      f"{m['feature_set']:<12} "
                      f"{m['test_balanced_accuracy']:.3f}   "
                      f"{m['test_precision_buy']:.3f}   "
                      f"{m['test_recall_buy']:.3f}   "
                      f"{m['test_roc_auc']:.3f}   "
                      f"{m['test_signals_buy_count']}")
            else:
                print(f"  {m['model_id']:<4} {m['alias']:<12} {'—':<7} {'—':<12} FAILED")

    print("\n--- PROCESS COMPLETED ---")


if __name__ == "__main__":
    main()
