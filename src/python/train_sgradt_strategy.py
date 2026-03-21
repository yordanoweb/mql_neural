"""
SGRADT 5.0 - Training Script
Implementa las condiciones exactas de Stochastic y ADX según el documento de estrategia.
"""

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


def calculate_stochastic_signals(df, k_period=7, d_period=3, slowing=3, 
                                  oversold=20.0, overbought=80.0):
    """
    Calcula las señales de Stochastic según la estrategia SGRADT 5.0.
    
    Señales BUY:
    1. Oversold Crossover: Main cruza arriba de Signal en zona oversold
    2. Strong Upward Momentum: Main aumenta +7 en dos barras consecutivas
    
    Señales SELL:
    1. Overbought Crossover: Main cruza abajo de Signal en zona overbought
    2. Strong Downward Momentum: Main disminuye -7 en dos barras consecutivas
    """
    stoch = StochasticOscillator(
        high=df['high'], 
        low=df['low'], 
        close=df['close'],
        window=k_period,
        smooth_window=d_period
    )
    
    main = stoch.stoch()      # %K - Main line
    signal = stoch.stoch_signal()  # %D - Signal line
    
    # Referencias a barras anteriores
    main_0 = main
    main_1 = main.shift(1)
    main_2 = main.shift(2)
    main_3 = main.shift(3)
    
    signal_0 = signal
    signal_1 = signal.shift(1)
    signal_2 = signal.shift(2)
    signal_3 = signal.shift(3)
    
    # ========== BUY SIGNALS ==========
    
    # Condición 1: Oversold Crossover (lookback 1-2)
    buy_oversold_1 = (
        (main_2 < signal_2) & 
        (main_1 > signal_1) & 
        (main_1 <= oversold)
    )
    
    # Condición 1 alternativa: Oversold Crossover (lookback 2-3)
    buy_oversold_2 = (
        (main_3 < signal_3) & 
        (main_2 > signal_2) & 
        (main_2 <= oversold)
    )
    
    # Condición 2: Strong Upward Momentum
    buy_momentum = (
        (main_0 > main_1 + 7) & 
        (main_1 > main_2 + 7)
    )
    
    stoch_buy = buy_oversold_1 | buy_oversold_2 | buy_momentum
    
    # ========== SELL SIGNALS ==========
    
    # Condición 1: Overbought Crossover (lookback 1-2)
    sell_overbought_1 = (
        (main_2 > signal_2) & 
        (main_1 < signal_1) & 
        (main_1 >= overbought)
    )
    
    # Condición 1 alternativa: Overbought Crossover (lookback 2-3)
    sell_overbought_2 = (
        (main_3 > signal_3) & 
        (main_2 < signal_2) & 
        (main_2 >= overbought)
    )
    
    # Condición 2: Strong Downward Momentum
    sell_momentum = (
        (main_0 < main_1 - 7) & 
        (main_1 < main_2 - 7)
    )
    
    stoch_sell = sell_overbought_1 | sell_overbought_2 | sell_momentum
    
    return {
        'stoch_main': main,
        'stoch_signal': signal,
        'stoch_buy': stoch_buy,
        'stoch_sell': stoch_sell
    }


def calculate_adx_signals(df, adx_period=8, adx_limit=32):
    """
    Calcula las señales de ADX según la estrategia SGRADT 5.0.
    
    Pre-condición: 
    - ADX > limit (actual o anterior)
    - O ADX muestra movimiento fuerte hacia arriba (+5 en una barra)
    
    Señales BUY:
    1. DI+ trending up, DI- trending down
    2. DI- reversal con DI+ comenzando a subir
    
    Señales SELL:
    1. DI- trending up, DI+ trending down
    2. DI+ reversal con DI- comenzando a subir
    """
    adx_inst = ADXIndicator(
        high=df['high'], 
        low=df['low'], 
        close=df['close'], 
        window=adx_period
    )
    
    adx = adx_inst.adx()
    pdi = adx_inst.adx_pos()  # +DI
    mdi = adx_inst.adx_neg()  # -DI
    
    # Referencias a barras anteriores
    adx_0 = adx
    adx_1 = adx.shift(1)
    adx_2 = adx.shift(2)
    
    pdi_0 = pdi
    pdi_1 = pdi.shift(1)
    pdi_2 = pdi.shift(2)
    pdi_3 = pdi.shift(3)
    
    mdi_0 = mdi
    mdi_1 = mdi.shift(1)
    mdi_2 = mdi.shift(2)
    mdi_3 = mdi.shift(3)
    
    # ========== PRE-CONDICIÓN: Mercado en tendencia ==========
    adx_strong = (
        (adx_0 > adx_limit) | 
        (adx_1 > adx_limit) |
        (adx_1 - adx_2 > 5) |
        (adx_0 - adx_1 > 5)
    )
    
    # ========== BUY SIGNALS (solo si pre-condición se cumple) ==========
    
    # Condición 1: DI+ trending up, DI- trending down
    buy_trend = (
        (pdi_0 > pdi_2) & 
        (pdi_1 > pdi_2) & 
        (pdi_0 > pdi_1) &
        (mdi_0 < mdi_1) & 
        (mdi_1 < mdi_2)
    )
    
    # Condición 2: -DI reversal
    buy_reversal = (
        (mdi_2 < mdi_3) & 
        (mdi_1 < mdi_2) & 
        (mdi_0 < mdi_1) &
        (pdi_0 > pdi_2)
    )
    
    adx_buy = adx_strong & (buy_trend | buy_reversal)
    
    # ========== SELL SIGNALS (solo si pre-condición se cumple) ==========
    
    # Condición 1: DI- trending up, DI+ trending down
    sell_trend = (
        (mdi_0 > mdi_2) & 
        (mdi_1 > mdi_2) & 
        (mdi_0 > mdi_1) &
        (pdi_0 < pdi_1) & 
        (pdi_1 < pdi_2)
    )
    
    # Condición 2: +DI reversal
    sell_reversal = (
        (pdi_2 < pdi_3) & 
        (pdi_1 < pdi_2) & 
        (pdi_0 < pdi_1) &
        (mdi_0 > mdi_2)
    )
    
    adx_sell = adx_strong & (sell_trend | sell_reversal)
    
    return {
        'adx': adx,
        'pdi': pdi,
        'mdi': mdi,
        'adx_buy': adx_buy,
        'adx_sell': adx_sell
    }


def main():
    parser = argparse.ArgumentParser(
        description='SGRADT 5.0 - Entrenamiento con estrategia Stochastic + ADX'
    )
    parser.add_argument('--csv', type=str, required=True, 
                       help='Archivo CSV con datos OHLC')
    parser.add_argument('--output', type=str, default='./onnx',
                       help='Directorio de salida')
    parser.add_argument('--window', type=int, default=20,
                       help='Ventana de lookback para features')
    
    # Parámetros de validación de señales
    parser.add_argument('--move_points', type=float, default=50.0,
                       help='Puntos mínimos de movimiento para validar señal')
    parser.add_argument('--future', type=int, default=10,
                       help='Barras futuras para validar movimiento')
    
    # Stochastic parameters (SGRADT 5.0 defaults)
    parser.add_argument('--stoch_k', type=int, default=7,
                       help='Stochastic K period')
    parser.add_argument('--stoch_d', type=int, default=3,
                       help='Stochastic D period')
    parser.add_argument('--stoch_slowing', type=int, default=3,
                       help='Stochastic slowing')
    parser.add_argument('--stoch_oversold', type=float, default=20.0,
                       help='Nivel de sobreventa')
    parser.add_argument('--stoch_overbought', type=float, default=80.0,
                       help='Nivel de sobrecompra')
    
    # ADX parameters (SGRADT 5.0 defaults)
    parser.add_argument('--adx_period', type=int, default=8,
                       help='ADX period')
    parser.add_argument('--adx_limit', type=float, default=32.0,
                       help='ADX threshold para confirmar tendencia')
    
    # Training parameters
    parser.add_argument('--n_iter', type=int, default=20,
                       help='Iteraciones para RandomizedSearchCV')
    parser.add_argument('--strategy', type=str, 
                       choices=['stoch', 'adx', 'combined'], 
                       default='combined',
                       help='Estrategia: stoch, adx o combined')
    
    args = parser.parse_args()
    csv_path = Path(args.csv)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*70}")
    print("SGRADT 5.0 - Training Strategy Model")
    print(f"{'='*70}")
    print(f"Archivo: {csv_path.name}")
    print(f"Estrategia: {args.strategy.upper()}")
    print(f"{'='*70}\n")
    
    # Cargar datos
    df = pd.read_csv(csv_path)
    print(f"Datos cargados: {len(df)} barras")
    
    # ========== CALCULAR INDICADORES Y SEÑALES ==========
    
    print("\nCalculando Stochastic...")
    stoch_results = calculate_stochastic_signals(
        df, 
        k_period=args.stoch_k,
        d_period=args.stoch_d,
        slowing=args.stoch_slowing,
        oversold=args.stoch_oversold,
        overbought=args.stoch_overbought
    )
    
    print("Calculando ADX...")
    adx_results = calculate_adx_signals(
        df,
        adx_period=args.adx_period,
        adx_limit=args.adx_limit
    )
    
    # Agregar features al DataFrame
    df['feat_body'] = df['close'] - df['open']
    df['feat_range'] = df['high'] - df['low']
    df['feat_stoch_main'] = stoch_results['stoch_main']
    df['feat_stoch_signal'] = stoch_results['stoch_signal']
    df['feat_adx'] = adx_results['adx']
    df['feat_pdi'] = adx_results['pdi']
    df['feat_mdi'] = adx_results['mdi']
    
    features_list = [
        'feat_body', 'feat_range',
        'feat_stoch_main', 'feat_stoch_signal',
        'feat_adx', 'feat_pdi', 'feat_mdi'
    ]
    
    # ========== CALCULAR SEÑALES ANTES DE DROPNA ==========
    
    # Guardar señales con el índice original del DataFrame
    df['signal_stoch_buy'] = stoch_results['stoch_buy']
    df['signal_stoch_sell'] = stoch_results['stoch_sell']
    df['signal_adx_buy'] = adx_results['adx_buy']
    df['signal_adx_sell'] = adx_results['adx_sell']
    
    # Ahora hacer dropna
    df = df.dropna(subset=features_list).reset_index(drop=True)
    print(f"Datos válidos después de NaN: {len(df)} barras")
    
    # ========== VALIDACIÓN DE MOVIMIENTO FUTURO ==========
    
    print(f"\nValidando movimientos futuros (>{args.move_points} puntos en {args.future} barras)...")
    
    future_max = df['close'].shift(-1).rolling(window=args.future).max()
    future_min = df['close'].shift(-1).rolling(window=args.future).min()
    
    # Validar que el movimiento sea suficiente
    valid_buy_move = (future_max > df['close'] + args.move_points)
    valid_sell_move = (future_min < df['close'] - args.move_points)
    
    # ========== COMBINAR SEÑALES SEGÚN ESTRATEGIA ==========
    
    labels = np.zeros(len(df))
    
    if args.strategy == 'stoch':
        # Solo señales de Stochastic
        buy_signals = df['signal_stoch_buy'] & valid_buy_move
        sell_signals = df['signal_stoch_sell'] & valid_sell_move
        
    elif args.strategy == 'adx':
        # Solo señales de ADX
        buy_signals = df['signal_adx_buy'] & valid_buy_move
        sell_signals = df['signal_adx_sell'] & valid_sell_move
        
    else:  # combined
        # Ambos indicadores deben confirmar
        buy_signals = (
            df['signal_stoch_buy'] & 
            df['signal_adx_buy'] & 
            valid_buy_move
        )
        sell_signals = (
            df['signal_stoch_sell'] & 
            df['signal_adx_sell'] & 
            valid_sell_move
        )
    
    labels[buy_signals] = 1
    labels[sell_signals] = 2
    
    count_buy = int(np.sum(labels == 1))
    count_sell = int(np.sum(labels == 2))
    count_hold = int(np.sum(labels == 0))
    
    print(f"\n{'='*70}")
    print(f"SEÑALES DETECTADAS:")
    print(f"{'='*70}")
    print(f"  BUY  (1): {count_buy:6d} señales ({count_buy/len(df)*100:5.2f}%)")
    print(f"  SELL (2): {count_sell:6d} señales ({count_sell/len(df)*100:5.2f}%)")
    print(f"  HOLD (0): {count_hold:6d} señales ({count_hold/len(df)*100:5.2f}%)")
    print(f"{'='*70}\n")
    
    if count_buy < 10 or count_sell < 10:
        print("❌ ERROR: Muy pocas señales encontradas.")
        print("\nSugerencias:")
        print("  • Reduce --move_points (actual: {})".format(args.move_points))
        print("  • Aumenta --future (actual: {})".format(args.future))
        print("  • Ajusta umbrales de Stochastic (oversold/overbought)")
        print("  • Reduce --adx_limit (actual: {})".format(args.adx_limit))
        print("  • Usa más datos históricos")
        sys.exit(1)
    
    # ========== PREPARAR VENTANAS PARA ENTRENAMIENTO ==========
    
    print(f"Preparando ventanas de {args.window} barras...")
    X_vals = df[features_list].values
    X, y = [], []
    
    for i in range(args.window, len(df)):
        X.append(X_vals[i - args.window:i].flatten())
        y.append(labels[i])
    
    X = np.array(X, dtype=np.float32)
    y = np.array(y)
    
    print(f"Dataset final: X shape = {X.shape}, y shape = {y.shape}")
    
    # ========== ENTRENAMIENTO ==========
    
    print(f"\n{'='*70}")
    print(f"ENTRENANDO RANDOM FOREST")
    print(f"{'='*70}")
    print(f"Iteraciones: {args.n_iter}")
    print(f"Validación cruzada: TimeSeriesSplit (3 splits)")
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
    
    print(f"\n✓ Entrenamiento completado")
    print(f"  Mejor Balanced Accuracy: {model_search.best_score_:.4f}")
    print(f"  Mejores parámetros: {model_search.best_params_}")
    
    # ========== EXPORTACIÓN ONNX ==========
    
    print(f"\n{'='*70}")
    print(f"EXPORTANDO MODELO ONNX")
    print(f"{'='*70}\n")
    
    num_features_per_bar = len(features_list)
    num_inputs = num_features_per_bar * args.window
    
    initial_type = [('float_input', FloatTensorType([1, num_inputs]))]
    onx = convert_sklearn(
        model_search.best_estimator_,
        initial_types=initial_type,
        target_opset=12,
        options={type(model_search.best_estimator_): {'zipmap': False}}
    )
    
    output_path = output_dir / f"{csv_path.stem}_SGRADT50_{args.strategy}.onnx"
    with open(output_path, "wb") as f:
        f.write(onx.SerializeToString())
    
    # ========== METADATA ==========
    
    meta = {
        "model_file": output_path.name,
        "strategy": args.strategy,
        "timestamp": pd.Timestamp.now().isoformat(),
        
        # Window config
        "window_size": args.window,
        "features_per_bar": num_features_per_bar,
        "num_inputs": num_inputs,
        "feature_order": features_list,
        
        # Validation config
        "future_window": args.future,
        "move_points": args.move_points,
        
        # Stochastic config
        "stoch_k_period": args.stoch_k,
        "stoch_d_period": args.stoch_d,
        "stoch_slowing": args.stoch_slowing,
        "stoch_oversold": args.stoch_oversold,
        "stoch_overbought": args.stoch_overbought,
        
        # ADX config
        "adx_period": args.adx_period,
        "adx_limit": args.adx_limit,
        
        # Training results
        "balanced_accuracy": float(model_search.best_score_),
        "best_params": model_search.best_params_,
        "signal_counts": {
            "buy": count_buy,
            "sell": count_sell,
            "hold": count_hold
        },
        
        # Model classes
        "classes": {
            "0": "HOLD",
            "1": "BUY",
            "2": "SELL"
        }
    }
    
    meta_path = output_dir / f"{csv_path.stem}_SGRADT50_{args.strategy}.meta.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    
    print(f"✓ Modelo guardado: {output_path}")
    print(f"✓ Metadata guardado: {meta_path}")
    print(f"\n{'='*70}")
    print("RESUMEN FINAL")
    print(f"{'='*70}")
    print(f"  Input shape: [1, {num_inputs}]")
    print(f"  Features: {num_features_per_bar} x {args.window} barras")
    print("  Output: 3 clases (HOLD=0, BUY=1, SELL=2)")
    print(f"  Accuracy: {model_search.best_score_:.4f}")
    print(f"{'='*70}\n")
    print("✅ PROCESO COMPLETADO CON ÉXITO\n")


if __name__ == "__main__":
    main()
