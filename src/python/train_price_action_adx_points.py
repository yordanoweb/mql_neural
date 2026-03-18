import pandas as pd
import numpy as np
import argparse
import sys
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
from ta.trend import ADXIndicator
import warnings

warnings.filterwarnings('ignore')

def main():
    parser = argparse.ArgumentParser(description='Entrenamiento ONNX: Perfil Selectivo con Sensores')
    parser.add_argument('--csv', type=str, required=True)
    parser.add_argument('--output', type=str, default='./models')
    parser.add_argument('--window', type=int, default=20)
    parser.add_argument('--adx_thresh', type=float, default=24.0)
    parser.add_argument('--move_points', type=float, default=50.0) # Ajustado a algo más realista para M5
    parser.add_argument('--future', type=int, default=10)
    parser.add_argument('--n_iter', type=int, default=20)
    
    args = parser.parse_args()
    csv_path = Path(args.csv)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n--- Analizando Datos: {csv_path.name} ---")
    df = pd.read_csv(csv_path)
    
    # 1. Features
    df['feat_body'] = df['close'] - df['open']
    df['feat_range'] = df['high'] - df['low']
    adx_inst = ADXIndicator(high=df['high'], low=df['low'], close=df['close'], window=14)
    df['feat_adx'] = adx_inst.adx()
    df['feat_pdi'] = adx_inst.adx_pos()
    df['feat_mdi'] = adx_inst.adx_neg()
    
    features_list = ['feat_body', 'feat_range', 'feat_adx', 'feat_pdi', 'feat_mdi']
    df = df.dropna(subset=features_list).reset_index(drop=True)

    # 2. Etiquetado con Contador (AQUÍ ESTÁ LA CLAVE)
    future_max = df['close'].shift(-1).rolling(window=args.future).max()
    future_min = df['close'].shift(-1).rolling(window=args.future).min()

    labels = np.zeros(len(df))
    buy_cond = (df['feat_adx'] > args.adx_thresh) & (df['feat_pdi'] > df['feat_mdi']) & (future_max > df['close'] + args.move_points)
    sell_cond = (df['feat_adx'] > args.adx_thresh) & (df['feat_mdi'] > df['feat_pdi']) & (future_min < df['close'] - args.move_points)
    
    labels[buy_cond] = 1
    labels[sell_cond] = 2
    
    count_buy = np.sum(labels == 1)
    count_sell = np.sum(labels == 2)
    
    print(f"SAMPLES TOTALES: {len(df)}")
    print(f"SEÑALES ENCONTRADAS -> BUY: {count_buy} | SELL: {count_sell}")

    if count_buy < 10 or count_sell < 10:
        print("\n¡ERROR! Muy pocas señales encontradas. Baja el valor de --move_points o sube --future.")
        sys.exit(1)

    # 3. Preparación
    X_vals = df[features_list].values
    X, y = [], []
    for i in range(args.window, len(df)):
        X.append(X_vals[i-args.window:i].flatten())
        y.append(labels[i])
        
    X = np.array(X).astype(np.float32)
    y = np.array(y)

    # 4. Entrenamiento con log de progreso
    print(f"Entrenando modelo (n_iter={args.n_iter})...")
    tscv = TimeSeriesSplit(n_splits=3)
    model_search = RandomizedSearchCV(
        RandomForestClassifier(random_state=42, class_weight='balanced'),
        param_distributions={'n_estimators': [300, 500], 'max_depth': [20, 30]},
        n_iter=args.n_iter, cv=tscv, scoring='balanced_accuracy', n_jobs=-1
    )
    model_search.fit(X, y)

    # 5. Exportación
    # Modifica esta parte en train_price_action_adx_points.py
    num_inputs = len(features_list) * args.window # Esto da 100
    # Forzamos [1, 100] en lugar de [None, 100] para evitar ambigüedad en MQL5
    # En lugar de [None, num_inputs], usamos una lista fija [1, 100]
    initial_type = [('float_input', FloatTensorType([1, 100]))] # 100 es window(20) * features(5) 

    onx = convert_sklearn(
        model_search.best_estimator_, 
        initial_types=initial_type, 
        target_opset=12, # MetaTrader soporta bien el opset 12 [cite: 21]
        options={type(model_search.best_estimator_): {'zipmap': False}}
    )    
    output_path = output_dir / f"{csv_path.stem}_Selectivo.onnx"
    with open(output_path, "wb") as f:
        f.write(onx.SerializeToString())
    
    print(f"✓ ÉXITO. Modelo guardado. Accuracy Balanceado: {model_search.best_score_:.4f}")

if __name__ == "__main__":
    main()
