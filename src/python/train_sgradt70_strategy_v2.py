"""
SGRADT 7.0 v2 - Training Script
Estrategia simplificada con EMA 9 como pivote
7 features: stoch_k, stoch_d, adx, pdi, mdi, ema_gate, volume_gate

La red neuronal toma TODAS las decisiones basándose en:
- 5 features técnicas (stoch_k, stoch_d, adx, pdi, mdi)
- 2 features de contexto (ema_gate, volume_gate)

No hay lógica manual de entry/exit - el modelo aprende los patrones completos.
"""

import pandas as pd
import numpy as np
import argparse
import json
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
from ta.trend import ADXIndicator, EMAIndicator
from ta.momentum import StochasticOscillator
import warnings

warnings.filterwarnings('ignore')


def calculate_features_and_labels(df, args):
    """
    Calcula las 7 features y labels basados en ganancia futura.
    
    Features:
    1. stoch_k (Stochastic %K)
    2. stoch_d (Stochastic %D signal)
    3. adx (Average Directional Index)
    4. pdi (Positive Directional Indicator)
    5. mdi (Minus Directional Indicator)
    6. ema_gate (1 si Open > EMA9, -1 si Open < EMA9, 0 si igual)
    7. volume_gate (ratio del volumen actual vs promedio de 10 barras)
    
    Labels:
    - Se calculan mirando hacia adelante (args.future barras)
    - BUY (1): Si precio sube >= min_profit_points
    - SELL (2): Si precio baja >= min_profit_points
    - HOLD (0): En otro caso
    """
    
    # Calcular EMA 9
    print("Calculando EMA 9...")
    ema_inst = EMAIndicator(close=df['close'], window=args.ema_period)
    df['ema9'] = ema_inst.ema_indicator()
    
    # Calcular ADX
    print("Calculando ADX...")
    adx_inst = ADXIndicator(
        high=df['high'], 
        low=df['low'], 
        close=df['close'], 
        window=args.adx_period
    )
    df['adx'] = adx_inst.adx()
    df['pdi'] = adx_inst.adx_pos()
    df['mdi'] = adx_inst.adx_neg()
    
    # Calcular Stochastic
    print("Calculando Stochastic...")
    stoch = StochasticOscillator(
        high=df['high'], 
        low=df['low'], 
        close=df['close'],
        window=args.stoch_k,
        smooth_window=args.stoch_d
    )
    df['stoch_k'] = stoch.stoch()
    df['stoch_d'] = stoch.stoch_signal()
    
    # Calcular Volume Gate (ratio vs promedio de 10 barras)
    print("Calculando Volume Gate...")
    df['volume_avg_10'] = df['tick_volume'].rolling(window=10).mean()
    df['volume_gate'] = df['tick_volume'] / df['volume_avg_10']
    df['volume_gate'] = df['volume_gate'].fillna(1.0)  # Default to 1.0 if NaN
    
    # Calcular EMA Gate (posición relativa de Open vs EMA)
    print("Calculando EMA Gate...")
    df['ema_gate'] = 0.0
    df.loc[df['open'] > df['ema9'], 'ema_gate'] = 1.0   # Above EMA
    df.loc[df['open'] < df['ema9'], 'ema_gate'] = -1.0  # Below EMA
    # When equal, stays 0.0
    
    # Crear features finales
    df['feat_stoch_main'] = df['stoch_k']
    df['feat_stoch_signal'] = df['stoch_d']
    df['feat_adx'] = df['adx']
    df['feat_pdi'] = df['pdi']
    df['feat_mdi'] = df['mdi']
    df['feat_ema_gate'] = df['ema_gate']
    df['feat_volume_gate'] = df['volume_gate']
    
    # Lista de features
    features_list = [
        'feat_stoch_main',
        'feat_stoch_signal',
        'feat_adx',
        'feat_pdi',
        'feat_mdi',
        'feat_ema_gate',
        'feat_volume_gate'
    ]
    
    # Dropna
    df = df.dropna(subset=features_list).reset_index(drop=True)
    
    print(f"Datos válidos después de NaN: {len(df)} barras")
    
    # ========== CALCULAR LABELS (FORWARD-LOOKING) ==========
    
    labels = np.zeros(len(df))
    
    for i in range(len(df) - args.future):
        
        entry_price = df['close'].iloc[i]
        
        # Buscar el mejor profit/loss en la ventana futura
        max_profit = 0
        max_loss = 0
        
        for j in range(i+1, min(i+args.future+1, len(df))):
            future_price = df['close'].iloc[j]
            
            # Calcular profit en puntos
            profit_points = (future_price - entry_price) / entry_price * 10000
            loss_points = (entry_price - future_price) / entry_price * 10000
            
            if profit_points > max_profit:
                max_profit = profit_points
            if loss_points > max_loss:
                max_loss = loss_points
        
        # Asignar label basado en el mejor movimiento
        if max_profit >= args.min_profit_points:
            labels[i] = 1  # BUY signal
        elif max_loss >= args.min_profit_points:
            labels[i] = 2  # SELL signal
        # else: stays 0 (HOLD)
    
    return df, labels, features_list


def train_and_export(csv_path, df, labels, features_list, args, output_dir):
    """
    Trains a Random Forest on the given dataset and exports it as ONNX.
    Called once per CSV file.
    """
    count_buy  = int(np.sum(labels == 1))
    count_sell = int(np.sum(labels == 2))
    count_hold = int(np.sum(labels == 0))

    print(f"\n{'='*70}")
    print(f"SEÑALES DETECTADAS: {csv_path.name}")
    print(f"{'='*70}")
    print(f"  BUY  (1): {count_buy:6d} señales ({count_buy/len(df)*100:5.2f}%)")
    print(f"  SELL (2): {count_sell:6d} señales ({count_sell/len(df)*100:5.2f}%)")
    print(f"  HOLD (0): {count_hold:6d} señales ({count_hold/len(df)*100:5.2f}%)")
    print(f"{'='*70}\n")

    if count_buy < 10 or count_sell < 10:
        print(f"  [SKIPPED] Muy pocas señales en {csv_path.name} — omitiendo entrenamiento.")
        print(f"  Sugerencias:")
        print(f"    - Reduce --min_profit_points (actual: {args.min_profit_points})")
        print(f"    - Aumenta --future (actual: {args.future})")
        return

    # Preparar ventanas
    print(f"Preparando ventanas de {args.window} barras...")
    X_vals = df[features_list].values
    X, y = [], []
    for i in range(args.window, len(df)):
        X.append(X_vals[i - args.window:i].flatten())
        y.append(labels[i])

    X = np.array(X, dtype=np.float32)
    y = np.array(y)
    print(f"Dataset: X shape = {X.shape}, y shape = {y.shape}")

    # Entrenamiento
    print(f"\n{'='*70}")
    print(f"ENTRENANDO RANDOM FOREST — {csv_path.name}")
    print(f"{'='*70}")
    print(f"Iteraciones: {args.n_iter} | Validacion: TimeSeriesSplit (3 splits)")
    print(f"{'='*70}\n")

    tscv = TimeSeriesSplit(n_splits=3)
    model_search = RandomizedSearchCV(
        RandomForestClassifier(random_state=42, class_weight='balanced'),
        param_distributions={
            'n_estimators': [100, 200, 300, 500],
            'max_depth': [10, 20, 30, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
        },
        n_iter=args.n_iter,
        cv=tscv,
        scoring='balanced_accuracy',
        n_jobs=-1,
        verbose=1
    )

    model_search.fit(X, y)

    print(f"\nEntrenamiento completado")
    print(f"  Mejor Balanced Accuracy: {model_search.best_score_:.4f}")
    print(f"  Mejores parametros: {model_search.best_params_}")

    # Exportacion ONNX
    print(f"\n{'='*70}")
    print(f"EXPORTANDO MODELO ONNX — {csv_path.name}")
    print(f"{'='*70}\n")

    num_features_per_bar = len(features_list)
    num_inputs = num_features_per_bar * args.window

    initial_type = [('float_input', FloatTensorType([1, num_inputs]))]
    onx = convert_sklearn(
        model_search.best_estimator_,
        initial_types=initial_type,
        target_opset=12,
        options={
            type(model_search.best_estimator_): {
                'zipmap': False,
                'nocl': False
            }
        }
    )

    output_path = output_dir / f"{csv_path.stem}_SGRADT70_v2.onnx"
    with open(output_path, "wb") as f:
        f.write(onx.SerializeToString())

    # Metadata
    meta = {
        "model_file": output_path.name,
        "source_file": csv_path.name,
        "version": "SGRADT 7.0 v2",
        "strategy": "Neural Network Driven (7 Features)",
        "timestamp": pd.Timestamp.now().isoformat(),

        "window_size": args.window,
        "features_per_bar": num_features_per_bar,
        "num_inputs": num_inputs,
        "feature_order": features_list,
        
        "feature_descriptions": {
            "feat_stoch_main": "Stochastic %K",
            "feat_stoch_signal": "Stochastic %D",
            "feat_adx": "Average Directional Index",
            "feat_pdi": "Positive Directional Indicator",
            "feat_mdi": "Minus Directional Indicator",
            "feat_ema_gate": "Position vs EMA9 (1=above, -1=below, 0=equal)",
            "feat_volume_gate": "Volume ratio vs 10-bar average"
        },

        "validation": {
            "min_profit_points": args.min_profit_points,
            "future_window": args.future,
        },

        "ema_period": args.ema_period,

        "stoch_k_period": args.stoch_k,
        "stoch_d_period": args.stoch_d,

        "adx_period": args.adx_period,

        "balanced_accuracy": float(model_search.best_score_),
        "best_params": model_search.best_params_,
        "signal_counts": {
            "buy": count_buy,
            "sell": count_sell,
            "hold": count_hold
        },

        "classes": {
            "0": "HOLD",
            "1": "BUY",
            "2": "SELL"
        },
        
        "notes": [
            "NN-driven strategy: model makes ALL decisions",
            "Gates are now features, not filters",
            "ATR-based SL/TP in MT5 code"
        ]
    }

    meta_path = output_dir / f"{csv_path.stem}_SGRADT70_v2.meta.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"Modelo guardado:   {output_path}")
    print(f"Metadata guardado: {meta_path}")
    print(f"\n{'='*70}")
    print(f"RESUMEN — {csv_path.name}")
    print(f"{'='*70}")
    print(f"  Input shape: [1, {num_inputs}]")
    print(f"  Features: {num_features_per_bar} x {args.window} barras")
    print("  Output: 3 clases (HOLD=0, BUY=1, SELL=2)")
    print(f"  Accuracy: {model_search.best_score_:.4f}")
    print("\n  Estrategia:")
    print("    - NN-driven: modelo decide basándose en 7 features")
    print("    - Features: stoch_k, stoch_d, adx, pdi, mdi, ema_gate, volume_gate")
    print("    - Exit: ATR-based SL/TP (configurado en MT5)")
    print(f"{'='*70}\n")


def main():
    parser = argparse.ArgumentParser(
        description='SGRADT 7.0 v2 - NN-driven Strategy with 7 features'
    )
    parser.add_argument('--csv', type=str, nargs='+', required=True,
                       help='Uno o más archivos CSV con datos OHLC (debe incluir columna tick_volume)')
    parser.add_argument('--output', type=str, default='./onnx',
                       help='Directorio de salida (default: ./onnx)')
    parser.add_argument('--window', type=int, default=20,
                       help='Ventana de lookback para features (default: 20)')

    # Parametros de validacion
    parser.add_argument('--min_profit_points', type=float, default=20.0,
                       help='Puntos minimos de ganancia para validar señal (default: 20)')
    parser.add_argument('--future', type=int, default=50,
                       help='Barras futuras maximas para buscar profit (default: 50)')

    # EMA parameters
    parser.add_argument('--ema_period', type=int, default=9,
                       help='EMA period (default: 9)')

    # Stochastic parameters
    parser.add_argument('--stoch_k', type=int, default=7,
                       help='Stochastic K period (default: 7)')
    parser.add_argument('--stoch_d', type=int, default=3,
                       help='Stochastic D period (default: 3)')

    # ADX parameters
    parser.add_argument('--adx_period', type=int, default=8,
                       help='ADX period (default: 8)')

    # Training parameters
    parser.add_argument('--n_iter', type=int, default=7,
                       help='Iteraciones para RandomizedSearchCV (default: 7)')

    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_paths = [Path(p) for p in args.csv]

    print(f"\n{'='*70}")
    print("SGRADT 7.0 v2 - NN-Driven Strategy (7 Features)")
    print(f"{'='*70}")
    print(f"Archivos a procesar: {len(csv_paths)}")
    for p in csv_paths:
        print(f"  - {p.name}")
    print(f"{'='*70}\n")
    print("IMPORTANTE: Los CSV deben incluir la columna 'tick_volume'")
    print(f"{'='*70}\n")

    skipped, completed = [], []

    for csv_path in csv_paths:
        if not csv_path.exists():
            print(f"[WARNING] Archivo no encontrado, omitiendo: {csv_path}")
            skipped.append(csv_path.name)
            continue

        print(f"\n{'='*70}")
        print(f"PROCESANDO: {csv_path.name}")
        print(f"{'='*70}")

        df = pd.read_csv(csv_path)
        print(f"Datos cargados: {len(df)} barras")
        
        # Verificar que existe la columna tick_volume
        if 'tick_volume' not in df.columns:
            print(f"[ERROR] Columna 'tick_volume' no encontrada en {csv_path.name}")
            print(f"        Columnas disponibles: {list(df.columns)}")
            skipped.append(csv_path.name)
            continue

        df, labels, features_list = calculate_features_and_labels(df, args)

        count_buy  = int(np.sum(labels == 1))
        count_sell = int(np.sum(labels == 2))

        if count_buy < 10 or count_sell < 10:
            skipped.append(csv_path.name)
        else:
            completed.append(csv_path.name)

        train_and_export(csv_path, df, labels, features_list, args, output_dir)

    # Resumen global
    print(f"\n{'='*70}")
    print("PROCESO COMPLETADO")
    print(f"{'='*70}")
    print(f"  Completados ({len(completed)}): {', '.join(completed) if completed else 'ninguno'}")
    print(f"  Omitidos    ({len(skipped)}):   {', '.join(skipped) if skipped else 'ninguno'}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
