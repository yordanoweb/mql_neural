"""
SGRADT 7.0 - Training Script
Estrategia simplificada con EMA 9 como pivote
5 features: body, range, stoch_k, stoch_d, adx

Entrada: Vela abre por encima/debajo de EMA 9 + confirmación ADX/Stoch
Salida: Vela abre cruzando EMA 9 en dirección opuesta
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


def calculate_signals_and_labels(df, args):
    """
    Calcula señales de entrada y labels basados en EMA 9 cross.
    
    Entry BUY:
    - Open[0] > EMA9[0] (vela abre por encima)
    - ADX > adx_limit (tendencia fuerte)
    - Stochastic oversold crossover O strong momentum
    
    Entry SELL:
    - Open[0] < EMA9[0] (vela abre por debajo)
    - ADX > adx_limit (tendencia fuerte)
    - Stochastic overbought crossover O strong momentum
    
    Exit:
    - BUY: Cuando open cruza por debajo de EMA 9
    - SELL: Cuando open cruza por encima de EMA 9
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
    
    # Crear features
    df['feat_body'] = df['close'] - df['open']
    df['feat_range'] = df['high'] - df['low']
    df['feat_stoch_main'] = df['stoch_k']
    df['feat_stoch_signal'] = df['stoch_d']
    df['feat_adx'] = df['adx']
    
    # Dropna
    features_list = ['feat_body', 'feat_range', 'feat_stoch_main', 'feat_stoch_signal', 'feat_adx']
    df = df.dropna(subset=features_list + ['ema9']).reset_index(drop=True)
    
    print(f"Datos válidos después de NaN: {len(df)} barras")
    
    # ========== CALCULAR LABELS ==========
    
    labels = np.zeros(len(df))
    
    for i in range(1, len(df) - args.future):
        
        # Referencias
        open_0 = df['open'].iloc[i]
        ema9_0 = df['ema9'].iloc[i]
        adx_0 = df['adx'].iloc[i]
        stoch_k_0 = df['stoch_k'].iloc[i]
        stoch_k_1 = df['stoch_k'].iloc[i-1] if i > 0 else 0
        stoch_d_0 = df['stoch_d'].iloc[i]
        stoch_d_1 = df['stoch_d'].iloc[i-1] if i > 0 else 0
        
        # ========== BUY SIGNAL ==========
        
        # Condición: Vela abre por encima de EMA 9
        above_ema = open_0 > ema9_0
        
        # ADX confirma tendencia
        adx_strong = adx_0 > args.adx_limit
        
        # Stochastic oversold crossover O strong momentum
        stoch_oversold_cross = (stoch_k_1 < stoch_d_1) and (stoch_k_0 > stoch_d_0) and (stoch_k_0 <= args.stoch_oversold)
        stoch_momentum_up = (stoch_k_0 > stoch_k_1 + 7)
        
        stoch_buy = stoch_oversold_cross or stoch_momentum_up
        
        buy_signal = above_ema and adx_strong and stoch_buy
        
        # ========== SELL SIGNAL ==========
        
        # Condición: Vela abre por debajo de EMA 9
        below_ema = open_0 < ema9_0
        
        # Stochastic overbought crossover O strong momentum
        stoch_overbought_cross = (stoch_k_1 > stoch_d_1) and (stoch_k_0 < stoch_d_0) and (stoch_k_0 >= args.stoch_overbought)
        stoch_momentum_down = (stoch_k_0 < stoch_k_1 - 7)
        
        stoch_sell = stoch_overbought_cross or stoch_momentum_down
        
        sell_signal = below_ema and adx_strong and stoch_sell
        
        # ========== VALIDAR EXIT EN EL FUTURO ==========
        
        if buy_signal:
            # Buscar exit: cuando open cruza por debajo de EMA 9
            for j in range(i+1, min(i+args.future+1, len(df))):
                if df['open'].iloc[j] < df['ema9'].iloc[j]:
                    # Calcular ganancia
                    entry_price = df['open'].iloc[i]
                    exit_price = df['open'].iloc[j]
                    profit_points = (exit_price - entry_price) / df['close'].iloc[i] * 10000  # En puntos
                    
                    if profit_points >= args.min_profit_points:
                        labels[i] = 1  # BUY
                    break
        
        elif sell_signal:
            # Buscar exit: cuando open cruza por encima de EMA 9
            for j in range(i+1, min(i+args.future+1, len(df))):
                if df['open'].iloc[j] > df['ema9'].iloc[j]:
                    # Calcular ganancia
                    entry_price = df['open'].iloc[i]
                    exit_price = df['open'].iloc[j]
                    profit_points = (entry_price - exit_price) / df['close'].iloc[i] * 10000  # En puntos
                    
                    if profit_points >= args.min_profit_points:
                        labels[i] = 2  # SELL
                    break
    
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
        print(f"    - Reduce --adx_limit (actual: {args.adx_limit})")
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

    output_path = output_dir / f"{csv_path.stem}_SGRADT70_ema9.onnx"
    with open(output_path, "wb") as f:
        f.write(onx.SerializeToString()) # pyright: ignore

    # Metadata
    meta = {
        "model_file": output_path.name,
        "source_file": csv_path.name,
        "version": "SGRADT 7.0",
        "strategy": "EMA 9 Cross",
        "timestamp": pd.Timestamp.now().isoformat(),

        "window_size": args.window,
        "features_per_bar": num_features_per_bar,
        "num_inputs": num_inputs,
        "feature_order": features_list,

        "validation": {
            "min_profit_points": args.min_profit_points,
            "future_window": args.future,
        },

        "ema_period": args.ema_period,

        "stoch_k_period": args.stoch_k,
        "stoch_d_period": args.stoch_d,
        "stoch_oversold": args.stoch_oversold,
        "stoch_overbought": args.stoch_overbought,

        "adx_period": args.adx_period,
        "adx_limit": args.adx_limit,

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
        }
    }

    meta_path = output_dir / f"{csv_path.stem}_SGRADT70_ema9.meta.json"
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
    print("    - Entry: Open vs EMA 9 + ADX + Stochastic")
    print("    - Exit: Open cruza EMA 9 en direccion opuesta")
    print("    - Features: 5 (stoch_k, stoch_d, adx, di_plus, di_minus)")
    print(f"{'='*70}\n")


def main():
    parser = argparse.ArgumentParser(
        description='SGRADT 7.0 - Estrategia EMA 9 con 5 features'
    )
    parser.add_argument('--csv', type=str, nargs='+', required=True,
                       help='Uno o más archivos CSV con datos OHLC')
    parser.add_argument('--output', type=str, default='./onnx',
                       help='Directorio de salida (default: ./onnx)')
    parser.add_argument('--window', type=int, default=20,
                       help='Ventana de lookback para features (default: 20)')

    # Parametros de validacion
    parser.add_argument('--min_profit_points', type=float, default=20.0,
                       help='Puntos minimos de ganancia para validar señal (default: 20)')
    parser.add_argument('--future', type=int, default=50,
                       help='Barras futuras maximas para buscar exit (default: 50)')

    # EMA parameters
    parser.add_argument('--ema_period', type=int, default=9,
                       help='EMA period (default: 9)')

    # Stochastic parameters
    parser.add_argument('--stoch_k', type=int, default=7,
                       help='Stochastic K period (default: 7)')
    parser.add_argument('--stoch_d', type=int, default=3,
                       help='Stochastic D period (default: 3)')
    parser.add_argument('--stoch_oversold', type=float, default=20.0,
                       help='Nivel de sobreventa (default: 20)')
    parser.add_argument('--stoch_overbought', type=float, default=80.0,
                       help='Nivel de sobrecompra (default: 80)')

    # ADX parameters
    parser.add_argument('--adx_period', type=int, default=8,
                       help='ADX period (default: 8)')
    parser.add_argument('--adx_limit', type=float, default=25.0,
                       help='ADX threshold para confirmar tendencia (default: 25)')

    # Training parameters
    parser.add_argument('--n_iter', type=int, default=7,
                       help='Iteraciones para RandomizedSearchCV (default: 7)')

    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_paths = [Path(p) for p in args.csv]

    print(f"\n{'='*70}")
    print("SGRADT 7.0 - EMA 9 Strategy (5 Features)")
    print(f"{'='*70}")
    print(f"Archivos a procesar: {len(csv_paths)}")
    for p in csv_paths:
        print(f"  - {p.name}")
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

        df, labels, features_list = calculate_signals_and_labels(df, args)

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
