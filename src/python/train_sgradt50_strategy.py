"""
SGRADT 5.0 - Training Script (5 Features Version)
Entrenamiento basado únicamente en: Stochastic %K, %D, ADX, +DI, -DI
"""

import pandas as pd
import numpy as np
import argparse
import sys
import json
import time
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
from ta.trend import ADXIndicator
from ta.momentum import StochasticOscillator
import warnings

warnings.filterwarnings('ignore')


def calculate_indicators(df, k_period=7, d_period=3, slowing=3, adx_period=8):
    """
    Calcula los 5 indicadores principales del modelo.
    
    Returns:
        dict con: stoch_main, stoch_signal, adx, pdi, mdi
    """
    # Stochastic
    stoch = StochasticOscillator(
        high=df['high'], 
        low=df['low'], 
        close=df['close'],
        window=k_period,
        smooth_window=d_period
    )
    
    stoch_main = stoch.stoch()        # %K
    stoch_signal = stoch.stoch_signal()  # %D
    
    # ADX
    adx_inst = ADXIndicator(
        high=df['high'], 
        low=df['low'], 
        close=df['close'], 
        window=adx_period
    )
    
    adx = adx_inst.adx()
    pdi = adx_inst.adx_pos()  # +DI
    mdi = adx_inst.adx_neg()  # -DI
    
    return {
        'stoch_main': stoch_main,
        'stoch_signal': stoch_signal,
        'adx': adx,
        'pdi': pdi,
        'mdi': mdi
    }


def validate_signals(df, move_points=50.0, future=10):
    """
    Valida movimientos futuros para clasificar señales.

    Cuando ambas condiciones (sube Y baja move_points) se cumplen en la misma
    ventana, se desempata usando el movimiento neto futuro: si el cierre neto
    sube -> BUY, si baja -> SELL. Esto evita que SELL sobreescriba BUY en
    barras donde el precio hace las dos cosas (mercados volátiles con tendencia).

    Returns:
        labels: array con 0=HOLD, 1=BUY, 2=SELL
    """
    future_max   = df['close'].shift(-1).rolling(window=future).max()
    future_min   = df['close'].shift(-1).rolling(window=future).min()
    future_close = df['close'].shift(-future)

    valid_buy_move  = (future_max > df['close'] + move_points)
    valid_sell_move = (future_min < df['close'] - move_points)
    both            = valid_buy_move & valid_sell_move

    labels = np.zeros(len(df))

    # Casos no ambiguos
    labels[valid_buy_move  & ~both] = 1
    labels[valid_sell_move & ~both] = 2

    # Desempate por movimiento neto cuando ambas condiciones son True
    labels[both & (future_close >  df['close'])] = 1
    labels[both & (future_close <= df['close'])] = 2

    return labels


def format_time(seconds):
    """Formatea segundos en formato legible"""
    if seconds < 60:
        return f"{seconds:.2f}s"
    elif seconds < 3600:
        mins = int(seconds // 60)
        secs = seconds % 60
        return f"{mins}m {secs:.2f}s"
    else:
        hours = int(seconds // 3600)
        mins = int((seconds % 3600) // 60)
        secs = seconds % 60
        return f"{hours}h {mins}m {secs:.2f}s"


def main():
    parser = argparse.ArgumentParser(
        description='SGRADT 5.0 - Entrenamiento con 5 features'
    )
    parser.add_argument('--csv', type=str, nargs='+', required=True, 
                       help='Archivo(s) CSV con datos OHLC (puede ser múltiples)')
    parser.add_argument('--output', type=str, default='./onnx',
                       help='Directorio de salida')
    parser.add_argument('--window', type=int, default=20,
                       help='Ventana de lookback para features')
    
    # Parámetros de validación de señales
    parser.add_argument('--move_points', type=float, default=50.0,
                       help='Puntos mínimos de movimiento para validar señal')
    parser.add_argument('--future', type=int, default=10,
                       help='Barras futuras para validar movimiento')
    
    # Parámetros de indicadores (SGRADT 5.0 defaults)
    parser.add_argument('--stoch_k', type=int, default=7,
                       help='Stochastic K period')
    parser.add_argument('--stoch_d', type=int, default=3,
                       help='Stochastic D period')
    parser.add_argument('--stoch_slowing', type=int, default=3,
                       help='Stochastic slowing')
    parser.add_argument('--adx_period', type=int, default=8,
                       help='ADX period')
    
    # Training parameters
    parser.add_argument('--n_iter', type=int, default=20,
                       help='Iteraciones para RandomizedSearchCV')
    
    args = parser.parse_args()
    
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Obtener lista de archivos CSV
    csv_files = [Path(csv) for csv in args.csv]
    total_files = len(csv_files)
    
    print(f"\n{'='*70}")
    print("SGRADT 5.0 - Training Model (5 Features)")
    print(f"{'='*70}")
    print(f"Total de archivos a procesar: {total_files}")
    print(f"Features: 5 (Stochastic K, D + ADX, +DI, -DI)")
    print(f"{'='*70}\n")
    
    # Iniciar temporizador total
    total_start_time = time.time()
    
    # Lista para rastrear tiempos y resultados
    results = []
    
    # Procesar cada archivo CSV
    for file_idx, csv_path in enumerate(csv_files, 1):
        print(f"\n{'#'*70}")
        print(f"# PROCESANDO ARCHIVO {file_idx}/{total_files}")
        print(f"# {csv_path.name}")
        print(f"{'#'*70}\n")
        
        # Iniciar temporizador para este archivo
        file_start_time = time.time()
        
        try:
            # Cargar datos
            df = pd.read_csv(csv_path)
            print(f"Datos cargados: {len(df)} barras")
            
            # ========== CALCULAR INDICADORES ==========
            
            print("\nCalculando indicadores...")
            indicators = calculate_indicators(
                df, 
                k_period=args.stoch_k,
                d_period=args.stoch_d,
                slowing=args.stoch_slowing,
                adx_period=args.adx_period
            )
            
            # Agregar features al DataFrame
            df['feat_stoch_main'] = indicators['stoch_main']
            df['feat_stoch_signal'] = indicators['stoch_signal']
            df['feat_adx'] = indicators['adx']
            df['feat_pdi'] = indicators['pdi']
            df['feat_mdi'] = indicators['mdi']
            
            features_list = [
                'feat_stoch_main', 
                'feat_stoch_signal',
                'feat_adx', 
                'feat_pdi', 
                'feat_mdi'
            ]
            
            # Eliminar NaN
            df = df.dropna(subset=features_list).reset_index(drop=True)
            print(f"Datos válidos después de NaN: {len(df)} barras")
            
            # ========== VALIDACIÓN DE MOVIMIENTO FUTURO ==========
            
            print(f"\nValidando movimientos futuros (>{args.move_points} puntos en {args.future} barras)...")
            
            labels = validate_signals(df, args.move_points, args.future)
            
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
                print("ERROR: Muy pocas señales encontradas en este archivo.")
                print("Saltando al siguiente archivo...\n")
                
                file_elapsed = time.time() - file_start_time
                results.append({
                    'file': csv_path.name,
                    'status': 'FAILED',
                    'reason': 'Pocas señales',
                    'time': file_elapsed
                })
                continue
            
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
            
            print(f"\nEntrenamiento completado")
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
            
            output_path = output_dir / f"{csv_path.stem}_SGRADT50.onnx"
            with open(output_path, "wb") as f:
                f.write(onx.SerializeToString())
            
            # ========== METADATA ==========
            
            meta = {
                "model_file": output_path.name,
                "version": "5.0",
                "timestamp": pd.Timestamp.now().isoformat(),
                
                # Window config
                "window_size": args.window,
                "features_per_bar": num_features_per_bar,
                "num_inputs": num_inputs,
                "feature_order": features_list,
                
                # Validation config
                "future_window": args.future,
                "move_points": args.move_points,
                
                # Indicator config
                "stoch_k_period": args.stoch_k,
                "stoch_d_period": args.stoch_d,
                "stoch_slowing": args.stoch_slowing,
                "adx_period": args.adx_period,
                
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
            
            meta_path = output_dir / f"{csv_path.stem}_SGRADT50.meta.json"
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(meta, f, ensure_ascii=False, indent=2)
            
            # Calcular tiempo de este archivo
            file_elapsed = time.time() - file_start_time
            
            print(f"\nModelo guardado: {output_path}")
            print(f"Metadata guardado: {meta_path}")
            print(f"\n{'='*70}")
            print("RESUMEN ARCHIVO")
            print(f"{'='*70}")
            print(f"  Input shape: [1, {num_inputs}]")
            print(f"  Features: {num_features_per_bar} x {args.window} barras")
            print(f"    1. feat_stoch_main   (Stochastic %K)")
            print(f"    2. feat_stoch_signal (Stochastic %D)")
            print(f"    3. feat_adx          (ADX)")
            print(f"    4. feat_pdi          (+DI)")
            print(f"    5. feat_mdi          (-DI)")
            print("  Output: 3 clases (HOLD=0, BUY=1, SELL=2)")
            print(f"  Accuracy: {model_search.best_score_:.4f}")
            print(f"  TIEMPO: {format_time(file_elapsed)}")
            print(f"{'='*70}\n")
            
            # Guardar resultado exitoso
            results.append({
                'file': csv_path.name,
                'status': 'SUCCESS',
                'accuracy': model_search.best_score_,
                'time': file_elapsed
            })
            
        except Exception as e:
            file_elapsed = time.time() - file_start_time
            print(f"\nERROR procesando {csv_path.name}: {str(e)}")
            print(f"TIEMPO: {format_time(file_elapsed)}\n")
            
            results.append({
                'file': csv_path.name,
                'status': 'ERROR',
                'reason': str(e),
                'time': file_elapsed
            })
    
    # Calcular tiempo total
    total_elapsed = time.time() - total_start_time
    
    # ========== RESUMEN FINAL ==========
    
    print(f"\n{'#'*70}")
    print(f"# RESUMEN FINAL DE PROCESAMIENTO")
    print(f"{'#'*70}\n")
    
    successful = [r for r in results if r['status'] == 'SUCCESS']
    failed = [r for r in results if r['status'] != 'SUCCESS']
    
    print(f"Total de archivos procesados: {total_files}")
    print(f"  Exitosos: {len(successful)}")
    print(f"  Fallidos: {len(failed)}")
    print(f"\nTiempo total: {format_time(total_elapsed)}")
    
    if successful:
        print(f"\n{'='*70}")
        print("MODELOS GENERADOS EXITOSAMENTE:")
        print(f"{'='*70}")
        for r in successful:
            print(f"  {r['file']:40s} | Acc: {r['accuracy']:.4f} | Tiempo: {format_time(r['time'])}")
    
    if failed:
        print(f"\n{'='*70}")
        print("ARCHIVOS FALLIDOS:")
        print(f"{'='*70}")
        for r in failed:
            reason = r.get('reason', 'Error desconocido')
            print(f"  {r['file']:40s} | {reason:30s} | Tiempo: {format_time(r['time'])}")
    
    print(f"\n{'#'*70}")
    print(f"# PROCESO COMPLETADO")
    print(f"# Tiempo total: {format_time(total_elapsed)}")
    print(f"{'#'*70}\n")


if __name__ == "__main__":
    main()
