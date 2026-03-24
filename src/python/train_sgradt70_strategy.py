"""
SGRADT 7.0 - Training Script (Individual File Mode)
EMA COMPLETELY REMOVED.
Entry: ADX + Stochastic.
Exit/Labeling: Fixed bar horizon (Target reached within 'future' bars).
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
from ta.trend import ADXIndicator # EMAIndicator removed
from ta.momentum import StochasticOscillator
import warnings

warnings.filterwarnings('ignore')

def calculate_signals_and_labels(df, args):
    """
    Calcula señales sin rastro de EMA. 
    El éxito de la señal se mide por alcanzar el profit en 'future' barras.
    """
    # Indicadores (ADX y Stochastic solamente)
    adx_inst = ADXIndicator(high=df['high'], low=df['low'], close=df['close'], window=args.adx_period)
    df['adx'] = adx_inst.adx()
    
    stoch = StochasticOscillator(high=df['high'], low=df['low'], close=df['close'], window=args.stoch_k, smooth_window=args.stoch_d)
    df['stoch_k'] = stoch.stoch()
    df['stoch_d'] = stoch.stoch_signal()
    
    # Features (5)
    df['feat_body'] = df['close'] - df['open']
    df['feat_range'] = df['high'] - df['low']
    df['feat_stoch_main'] = df['stoch_k']
    df['feat_stoch_signal'] = df['stoch_d']
    df['feat_adx'] = df['adx']
    
    features_list = ['feat_body', 'feat_range', 'feat_stoch_main', 'feat_stoch_signal', 'feat_adx']
    df = df.dropna(subset=features_list).reset_index(drop=True)
    
    labels = np.zeros(len(df))
    
    for i in range(1, len(df) - args.future):
        # Entry check (Stochastic + ADX only)
        adx_strong = df['adx'].iloc[i] > args.adx_limit
        stoch_k_0, stoch_k_1 = df['stoch_k'].iloc[i], df['stoch_k'].iloc[i-1]
        stoch_d_0, stoch_d_1 = df['stoch_d'].iloc[i], df['stoch_d'].iloc[i-1]
        
        # BUY Logic
        stoch_buy = ((stoch_k_1 < stoch_d_1) and (stoch_k_0 > stoch_d_0) and (stoch_k_0 <= args.stoch_oversold)) or (stoch_k_0 > stoch_k_1 + 7)
        buy_signal = adx_strong and stoch_buy
        
        # SELL Logic
        stoch_sell = ((stoch_k_1 > stoch_d_1) and (stoch_k_0 < stoch_d_0) and (stoch_k_0 >= args.stoch_overbought)) or (stoch_k_0 < stoch_k_1 - 7)
        sell_signal = adx_strong and stoch_sell

        entry_price = df['open'].iloc[i+1] # Entramos al abrir la siguiente vela

        if buy_signal:
            # EXIT: ¿Llega al profit objetivo en las próximas 'future' velas?
            for j in range(i+1, i + args.future + 1):
                max_reach = df['high'].iloc[j]
                profit = (max_reach - entry_price) / df['close'].iloc[i] * 10000
                if profit >= args.min_profit_points:
                    labels[i] = 1
                    break
        elif sell_signal:
            # EXIT: ¿Llega al profit objetivo en las próximas 'future' velas?
            for j in range(i+1, i + args.future + 1):
                min_reach = df['low'].iloc[j]
                profit = (entry_price - min_reach) / df['close'].iloc[i] * 10000
                if profit >= args.min_profit_points:
                    labels[i] = 2
                    break
                    
    return df, labels, features_list

def main():
    parser = argparse.ArgumentParser(description='SGRADT 7.0 - NO EMA VERSION')
    parser.add_argument('--csv', type=str, nargs='+', required=True)
    parser.add_argument('--output', type=str, default='./onnx')
    parser.add_argument('--window', type=int, default=20)
    parser.add_argument('--min_profit_points', type=float, default=20.0)
    parser.add_argument('--future', type=int, default=50)
    # EMA Parameter removed
    parser.add_argument('--stoch_k', type=int, default=7)
    parser.add_argument('--stoch_d', type=int, default=3)
    parser.add_argument('--stoch_oversold', type=float, default=20.0)
    parser.add_argument('--stoch_overbought', type=float, default=80.0)
    parser.add_argument('--adx_period', type=int, default=8)
    parser.add_argument('--adx_limit', type=float, default=25.0)
    parser.add_argument('--n_iter', type=int, default=7)
    
    args = parser.parse_args()
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    for csv_path in [Path(p) for p in args.csv]:
        if not csv_path.exists(): continue

        print(f"\nProcessing: {csv_path.name}")
        df = pd.read_csv(csv_path)
        df, labels, features_list = calculate_signals_and_labels(df, args)

        count_buy = int(np.sum(labels == 1))
        count_sell = int(np.sum(labels == 2))

        if count_buy < 5 or count_sell < 5:
            print(f"Skipping {csv_path.name}: Insufficient signals (B:{count_buy}, S:{count_sell})")
            continue

        X_list, y_list = [], []
        X_vals = df[features_list].values
        for i in range(args.window, len(df)):
            X_list.append(X_vals[i - args.window:i].flatten())
            y_list.append(labels[i])

        X = np.array(X_list, dtype=np.float32)
        y = np.array(y_current := y_list)

        # Training
        model = RandomizedSearchCV(
            RandomForestClassifier(random_state=42, class_weight='balanced'),
            param_distributions={'n_estimators': [100, 200], 'max_depth': [10, 20, None]},
            n_iter=args.n_iter, cv=TimeSeriesSplit(n_splits=3), scoring='balanced_accuracy', n_jobs=-1
        )
        model.fit(X, y)

        # Export ONNX
        num_inputs = len(features_list) * args.window
        onx = convert_sklearn(model.best_estimator_, initial_types=[('float_input', FloatTensorType([1, num_inputs]))], target_opset=12)
        
        out_name = output_dir / f"{csv_path.stem}_SGRADT70.onnx"
        with open(out_name, "wb") as f: f.write(onx.SerializeToString())
        
        print(f"Model saved: {out_name.name} | Accuracy: {model.best_score_:.4f}")

if __name__ == "__main__":
    main()
