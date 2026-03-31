import pandas as pd
import numpy as np
import sys
import os
from pathlib import Path
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from onnxmltools import convert_xgboost
from onnxmltools.convert.common.data_types import FloatTensorType
import ta
import argparse

class Colors:
    RESET   = '\033[0m'
    RED     = '\033[91m'
    GREEN   = '\033[92m'
    YELLOW  = '\033[93m'
    BLUE    = '\033[94m'
    MAGENTA = '\033[95m'
    CYAN    = '\033[96m'
    WHITE   = '\033[97m'

def colorize(text, color):
    return f"{color}{text}{Colors.RESET}"

parser = argparse.ArgumentParser(description="EMA Cross Success Predictor")
parser.add_argument("--input_csv", type=str, required=True)
parser.add_argument("--output_dir", type=str, default=".")
parser.add_argument("--ema_period", type=int, default=20)
parser.add_argument("--future", type=int, default=5)
parser.add_argument("--profit_atr", type=float, default=1.5)
parser.add_argument("--stop_atr", type=float, default=1.0)
args = parser.parse_args()

csv_file = args.input_csv
output_dir = args.output_dir
ema_period = args.ema_period
future = args.future
profit_atr = args.profit_atr
stop_atr = args.stop_atr

os.makedirs(output_dir, exist_ok=True)
output_filename = os.path.join(output_dir, Path(csv_file).stem + f"_ema{ema_period}_f{future}_cls.onnx").replace("_rates", "")

print(colorize("--- EMA CROSS BINARY CLASSIFIER ---", Colors.CYAN))
print(f"EMA: {ema_period} | Future: {future} | Target: {profit_atr}ATR | Stop: {stop_atr}ATR")

df = pd.read_csv(csv_file)
print(f"Rows: {len(df)}")

df['ema'] = ta.trend.ema_indicator(df['close'], window=ema_period)
df['atr'] = ta.volatility.AverageTrueRange(df['high'], df['low'], df['close'], window=14).average_true_range()

# FEATURE ÚNICA: Distancia en ATRs (más robusta que %)
df['ema_dist'] = (df['close'] - df['ema']) / df['atr']

# 🔥 FIX: Limpiar infinitos y valores extremos
df['ema_dist'] = df['ema_dist'].replace([np.inf, -np.inf], np.nan)
df['ema_dist'] = df['ema_dist'].clip(-10.0, 10.0)  # Limitar a ±10 ATRs

# Cruces
df['above'] = df['close'] > df['ema']
df['cross_up'] = (~df['above'].shift(1).fillna(False)) & df['above']
df['cross_down'] = (df['above'].shift(1).fillna(False)) & (~df['above'])

# LABELING: ¿Tendrá éxito el cruce?
labels = np.zeros(len(df))
directions = np.zeros(len(df))

for i in range(len(df) - future - 1):
    # 🔥 FIX: Skip si ATR es inválido o muy pequeño
    if pd.isna(df['atr'].iloc[i]) or df['atr'].iloc[i] < 0.00001:
        continue
    if not (df['cross_up'].iloc[i] or df['cross_down'].iloc[i]):
        continue
        
    entry = df['close'].iloc[i]
    atr = df['atr'].iloc[i]
    if pd.isna(atr) or atr == 0:
        continue
    
    highs = df['high'].iloc[i+1:i+future+1]
    lows = df['low'].iloc[i+1:i+future+1]
    
    if df['cross_up'].iloc[i]:
        profit_lvl = entry + (profit_atr * atr)
        stop_lvl = entry - (stop_atr * atr)
        hit_profit = (highs >= profit_lvl).any()
        hit_stop = (lows <= stop_lvl).any()
        
        if hit_profit and (not hit_stop or highs[highs >= profit_lvl].index[0] <= lows[lows <= stop_lvl].index[0]):
            labels[i] = 1
        directions[i] = 1
        
    elif df['cross_down'].iloc[i]:
        profit_lvl = entry - (profit_atr * atr)
        stop_lvl = entry + (stop_atr * atr)
        hit_profit = (lows <= profit_lvl).any()
        hit_stop = (highs >= stop_lvl).any()
        
        if hit_profit and (not hit_stop or lows[lows <= profit_lvl].index[0] <= highs[highs >= stop_lvl].index[0]):
            labels[i] = 1
        directions[i] = -1

# --- LÍNEAS 95-100 (DATASET) ---
# Dataset
mask = (df['cross_up'] | df['cross_down']).values
X = df.loc[mask, ['ema_dist']].values.astype(np.float32)
y = labels[mask]
dirs = directions[mask]

# 🔥 FIX: Doble chequeo de valores inválidos
valid = ~np.isnan(X).any(axis=1) & ~np.isinf(X).any(axis=1) & ~np.isnan(y)
X = X[valid]
y = y[valid].astype(int)
dirs = dirs[valid]

# 🔥 FIX: Asegurar que no hay infinitos escondidos
X = np.nan_to_num(X, nan=0.0, posinf=10.0, neginf=-10.0)

print(f"Samples: {len(X)} | Base success rate: {y.mean():.1%}")
print(f"Feature stats - Min: {X.min():.2f}, Max: {X.max():.2f}, Mean: {X.mean():.2f}")

if len(X) < 100:
    sys.exit("Insufficient data")

# Split temporal
split = int(len(X) * 0.8)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]
dirs_test = dirs[split:]

# Clasificador
model = XGBClassifier(
    n_estimators=100,
    max_depth=3,
    learning_rate=0.1,
    scale_pos_weight=(len(y_train)-y_train.sum())/max(y_train.sum(),1),
    random_state=42,
    eval_metric='logloss'
)
model.fit(X_train, y_train)

# Eval
y_proba = model.predict_proba(X_test)[:, 1]
y_pred = model.predict(X_test)

print(f"Accuracy: {accuracy_score(y_test, y_pred):.1%}")

# Simulación: solo operar si prob > 0.6
thresh = 0.6
high_conf = y_proba > thresh
if high_conf.sum() > 0:
    success_filtered = y_test[high_conf].mean()
    print(f"High-conf trades (>{thresh}): {high_conf.sum()} | Success: {success_filtered:.1%}")
    
    # Desglose
    long_mask = dirs_test == 1
    if (high_conf & long_mask).sum() > 0:
        print(f"  Long: {y_test[high_conf & long_mask].mean():.1%}")
    if (high_conf & (dirs_test==-1)).sum() > 0:
        print(f"  Short: {y_test[high_conf & (dirs_test==-1)].mean():.1%}")

# Export
initial_type = [('float_input', FloatTensorType([None, 1]))]
onx = convert_xgboost(model, initial_types=initial_type)
with open(output_filename, "wb") as f:
    f.write(onx.SerializeToString())
print(colorize(f"Saved: {output_filename}", Colors.GREEN))