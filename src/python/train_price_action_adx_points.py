import pandas as pd
import numpy as np
import argparse
import sys
import json
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
from ta.trend import ADXIndicator
from ta.momentum import StochasticOscillator
import warnings

warnings.filterwarnings('ignore')

def main():
    parser = argparse.ArgumentParser(description='Entrenamiento ONNX: Perfil Selectivo con ADX + Stochastic')
    parser.add_argument('--csv',               type=str,   required=True)
    parser.add_argument('--output',            type=str,   default='./onnx')
    parser.add_argument('--window',            type=int,   default=20)
    parser.add_argument('--adx_thresh',        type=float, default=24.0)
    parser.add_argument('--move_points',       type=float, default=50.0)
    parser.add_argument('--future',            type=int,   default=10)
    parser.add_argument('--n_iter',            type=int,   default=10)
    # Stochastic parameters
    parser.add_argument('--stoch_k',           type=int,   default=5,    help='Stochastic %K period')
    parser.add_argument('--stoch_d',           type=int,   default=3,    help='Stochastic %D smoothing period')
    parser.add_argument('--stoch_slowing',     type=int,   default=3,    help='Stochastic slowing period')
    parser.add_argument('--stoch_oversold',    type=float, default=30.0, help='Oversold threshold for BUY label')
    parser.add_argument('--stoch_overbought',  type=float, default=70.0, help='Overbought threshold for SELL label')

    args = parser.parse_args()
    csv_path   = Path(args.csv)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n--- Analizando Datos: {csv_path.name} ---")
    df = pd.read_csv(csv_path)

    # ------------------------------------------------------------------
    # 1. Features — ADX (original 5) + Stochastic %K and %D (2 new)
    # ------------------------------------------------------------------
    df['feat_body']  = df['close'] - df['open']
    df['feat_range'] = df['high']  - df['low']

    adx_inst        = ADXIndicator(high=df['high'], low=df['low'], close=df['close'], window=14)
    df['feat_adx']  = adx_inst.adx()
    df['feat_pdi']  = adx_inst.adx_pos()
    df['feat_mdi']  = adx_inst.adx_neg()

    stoch_inst      = StochasticOscillator(
                          high=df['high'], low=df['low'], close=df['close'],
                          window=args.stoch_k,
                          smooth_window=args.stoch_d
                      )
    df['feat_stoch_k'] = stoch_inst.stoch()
    df['feat_stoch_d'] = stoch_inst.stoch_signal()

    features_list = [
        'feat_body', 'feat_range',
        'feat_adx',  'feat_pdi', 'feat_mdi',
        'feat_stoch_k', 'feat_stoch_d'
    ]

    df = df.dropna(subset=features_list).reset_index(drop=True)

    # ------------------------------------------------------------------
    # 2. Labeling — ADX trend filter + Stochastic crossover from zone
    # ------------------------------------------------------------------
    future_max = df['close'].shift(-1).rolling(window=args.future).max()
    future_min = df['close'].shift(-1).rolling(window=args.future).min()

    # Stochastic crossover detection
    # BUY  : %K crosses above %D (prev K <= D, curr K > D) while prev %K was in oversold zone
    k     = df['feat_stoch_k']
    d     = df['feat_stoch_d']
    k_prev = k.shift(1)
    d_prev = d.shift(1)

    stoch_cross_up   = (k_prev <= d_prev) & (k > d) & (k_prev < args.stoch_oversold)
    stoch_cross_down = (k_prev >= d_prev) & (k < d) & (k_prev > args.stoch_overbought)

    labels   = np.zeros(len(df))
    buy_cond = (
        (df['feat_adx'] > args.adx_thresh) &
        (df['feat_pdi'] > df['feat_mdi']) &
        (future_max > df['close'] + args.move_points) &
        stoch_cross_up
    )
    sell_cond = (
        (df['feat_adx'] > args.adx_thresh) &
        (df['feat_mdi'] > df['feat_pdi']) &
        (future_min < df['close'] - args.move_points) &
        stoch_cross_down
    )

    labels[buy_cond]  = 1
    labels[sell_cond] = 2

    count_buy  = int(np.sum(labels == 1))
    count_sell = int(np.sum(labels == 2))

    print(f"SAMPLES TOTALES:  {len(df)}")
    print(f"SEÑALES ENCONTRADAS -> BUY: {count_buy} | SELL: {count_sell}")
    print(f"Stochastic ({args.stoch_k},{args.stoch_d},{args.stoch_slowing}) "
          f"| Oversold<{args.stoch_oversold} | Overbought>{args.stoch_overbought}")

    if count_buy < 10 or count_sell < 10:
        print("\n¡ERROR! Muy pocas señales encontradas.")
        print("Sugerencias: baja --move_points, sube --future, "
              "sube --stoch_oversold / baja --stoch_overbought, o usa más datos.")
        sys.exit(1)

    # ------------------------------------------------------------------
    # 3. Preparación de ventanas
    # ------------------------------------------------------------------
    X_vals = df[features_list].values
    X, y = [], []
    for i in range(args.window, len(df)):
        X.append(X_vals[i - args.window:i].flatten())
        y.append(labels[i])

    X = np.array(X, dtype=np.float32)
    y = np.array(y)

    # ------------------------------------------------------------------
    # 4. Entrenamiento
    # ------------------------------------------------------------------
    print(f"\nEntrenando modelo (n_iter={args.n_iter}, features_por_barra={len(features_list)})...")
    tscv = TimeSeriesSplit(n_splits=3)
    model_search = RandomizedSearchCV(
        RandomForestClassifier(random_state=42, class_weight='balanced'),
        param_distributions={
            'n_estimators':      [100, 200, 300, 500],
            'max_depth':         [10, 20, 30, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf':  [1, 2, 4],
        },
        n_iter=args.n_iter, cv=tscv, scoring='balanced_accuracy', n_jobs=-1
    )
    model_search.fit(X, y)

    # ------------------------------------------------------------------
    # 5. Exportación ONNX
    # ------------------------------------------------------------------
    num_features_per_bar = len(features_list)   # 7
    num_inputs           = num_features_per_bar * args.window

    if X.shape[1] != num_inputs:
        raise ValueError(
            f"Mismatch interno: X tiene {X.shape[1]} features, pero "
            f"window({args.window})*features({num_features_per_bar})={num_inputs}"
        )

    initial_type = [('float_input', FloatTensorType([1, num_inputs]))]
    onx = convert_sklearn(
        model_search.best_estimator_,
        initial_types=initial_type,
        target_opset=12,
        options={type(model_search.best_estimator_): {'zipmap': False}}
    )

    output_path = output_dir / f"{csv_path.stem}_Selectivo.onnx"
    with open(output_path, "wb") as f:
        f.write(onx.SerializeToString())

    # Sidecar de metadatos — incluye parámetros Stochastic para configurar el EA
    meta = {
        "model_file":          output_path.name,
        "window_size":         args.window,
        "features_per_bar":    num_features_per_bar,
        "num_inputs":          num_inputs,
        "feature_order":       features_list,
        "future_window":       args.future,
        "adx_threshold":       args.adx_thresh,
        "move_points":         args.move_points,
        "stoch_k":             args.stoch_k,
        "stoch_d":             args.stoch_d,
        "stoch_slowing":       args.stoch_slowing,
        "stoch_oversold":      args.stoch_oversold,
        "stoch_overbought":    args.stoch_overbought,
    }
    meta_path = output_dir / f"{csv_path.stem}_Selectivo.meta.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"\n✓ ÉXITO. Modelo guardado en: {output_path}")
    print(f"  Accuracy Balanceado : {model_search.best_score_:.4f}")
    print(f"  Input ONNX shape    : [1, {num_inputs}] "
          f"(window={args.window} x {num_features_per_bar} features)")
    print(f"  Features            : {features_list}")
    print(f"  Metadata            : {meta_path}")

if __name__ == "__main__":
    main()
