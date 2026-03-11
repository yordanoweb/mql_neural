import pandas as pd
import numpy as np
import sys
import os
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
from indicators import calculate_stochastic, calculate_adx
import onnx

# --- CONFIGURATION ---
if len(sys.argv) < 3:
    print("Usage: python train_onnx_stochastic_adx.py <csv_file> <onnx_output_dir>")
    sys.exit(1)

csv_file = sys.argv[1]
onnx_output_dir = sys.argv[2]

if not os.path.exists(csv_file):
    print(f"Error: File '{csv_file}' not found")
    sys.exit(1)

if not os.path.exists(onnx_output_dir):
    os.makedirs(onnx_output_dir)

output_filename = Path(onnx_output_dir) / (Path(csv_file).stem + "_stochastic_adx.onnx")
print(f"--- STOCHASTIC & ADX TRAINING ---")
print(f"Loading rates from: {csv_file}")
print(f"Output ONNX will be: {output_filename}")

df = pd.read_csv(csv_file)
print(f"Records loaded: {len(df)}")

pip_unit = 0.0001

# --- STRATEGY PARAMETERS (from Markets.mqh) ---
STOCH_K_PERIOD = 7
STOCH_D_PERIOD = 3
STOCH_SLOWING = 3
STOCH_OVERBOUGHT = 80.0
STOCH_OVERSOLD = 20.0
STOCH_BYPASS = False

ADX_PERIOD = 8
ADX_LIMIT = 32.0
ADX_BYPASS = False
ADX_DI_OVER = False

# --- INDICATOR CALCULATION ---
print("Calculating Stochastic Oscillator...")
k_fast, d_fast, k_slow, d_slow = calculate_stochastic(
    df['high'].tolist(),
    df['low'].tolist(),
    df['close'].tolist(),
    k_period=STOCH_K_PERIOD,
    d_period=STOCH_D_PERIOD,
    slowing=STOCH_SLOWING
)

# Use slow %K as main line, slow %D as signal line
stoch_k = k_slow
stoch_d = d_slow

print("Calculating ADX...")
plus_di, minus_di, dx, adx = calculate_adx(
    df['high'].tolist(),
    df['low'].tolist(),
    df['close'].tolist(),
    period=ADX_PERIOD
)

# --- FEATURE ENGINEERING ---
window = 20

df['feat_body'] = (df['close'] - df['open']) / pip_unit
df['feat_range'] = (df['high'] - df['low']) / pip_unit
df['feat_stoch_k'] = pd.Series(stoch_k) / 100.0          # normalize to 0‑1
df['feat_stoch_d'] = pd.Series(stoch_d) / 100.0
df['feat_adx'] = pd.Series(adx)
df['feat_plus_di'] = pd.Series(plus_di)
df['feat_minus_di'] = pd.Series(minus_di)

# Drop rows where any indicator is missing (None -> NaN)
df = df.copy()
for col in ['feat_stoch_k', 'feat_stoch_d', 'feat_adx', 'feat_plus_di', 'feat_minus_di']:
    df = df[df[col].notna()]

print(f"Rows after dropping missing indicators: {len(df)}")

# --- LABEL GENERATION (signals from the MD document) ---
def adx_trending(i):
    """Check ADX pre‑condition for trend strength."""
    if i < 1:
        return False
    # ADX above limit on current or previous candle
    if adx[i] is not None and adx[i] > ADX_LIMIT:
        return True
    if adx[i-1] is not None and adx[i-1] > ADX_LIMIT:
        return True
    # Strong upward movement of ADX line
    if i >= 2:
        if adx[i-1] is not None and adx[i-2] is not None and (adx[i-1] - adx[i-2]) > 5:
            return True
        if adx[i] is not None and adx[i-1] is not None and (adx[i] - adx[i-1]) > 5:
            return True
    return False

def stoch_buy_signal(i):
    """Return True if a Stochastic buy signal occurs at candle i."""
    # Oversold crossover
    if i >= 2:
        if (stoch_k[i-2] is not None and stoch_d[i-2] is not None and
            stoch_k[i-2] < stoch_d[i-2] and
            stoch_k[i-1] is not None and stoch_d[i-1] is not None and
            stoch_k[i-1] > stoch_d[i-1] and
            stoch_k[i-1] <= STOCH_OVERSOLD):
            return True
        # Alternative lookback (3 candles ago)
        if i >= 3:
            if (stoch_k[i-3] is not None and stoch_d[i-3] is not None and
                stoch_k[i-3] < stoch_d[i-3] and
                stoch_k[i-2] is not None and stoch_d[i-2] is not None and
                stoch_k[i-2] > stoch_d[i-2] and
                stoch_k[i-2] <= STOCH_OVERSOLD):
                return True
    # Strong upward momentum
    if i >= 2:
        if (stoch_k[i] is not None and stoch_k[i-1] is not None and stoch_k[i-2] is not None and
            stoch_k[i] > stoch_k[i-1] + 7 and
            stoch_k[i-1] > stoch_k[i-2] + 7):
            return True
    return False

def stoch_sell_signal(i):
    """Return True if a Stochastic sell signal occurs at candle i."""
    # Overbought crossover
    if i >= 2:
        if (stoch_k[i-2] is not None and stoch_d[i-2] is not None and
            stoch_k[i-2] > stoch_d[i-2] and
            stoch_k[i-1] is not None and stoch_d[i-1] is not None and
            stoch_k[i-1] < stoch_d[i-1] and
            stoch_k[i-1] >= STOCH_OVERBOUGHT):
            return True
        if i >= 3:
            if (stoch_k[i-3] is not None and stoch_d[i-3] is not None and
                stoch_k[i-3] > stoch_d[i-3] and
                stoch_k[i-2] is not None and stoch_d[i-2] is not None and
                stoch_k[i-2] < stoch_d[i-2] and
                stoch_k[i-2] >= STOCH_OVERBOUGHT):
                return True
    # Strong downward momentum
    if i >= 2:
        if (stoch_k[i] is not None and stoch_k[i-1] is not None and stoch_k[i-2] is not None and
            stoch_k[i] < stoch_k[i-1] - 7 and
            stoch_k[i-1] < stoch_k[i-2] - 7):
            return True
    return False

def adx_buy_signal(i):
    """Return True if an ADX buy signal occurs at candle i."""
    if not adx_trending(i):
        return False
    # DI+ trending up, DI- trending down
    if i >= 2:
        if (plus_di[i] is not None and plus_di[i-1] is not None and plus_di[i-2] is not None and
            minus_di[i] is not None and minus_di[i-1] is not None and minus_di[i-2] is not None):
            if (plus_di[i] > plus_di[i-2] and
                plus_di[i-1] > plus_di[i-2] and
                plus_di[i] > plus_di[i-1] and
                minus_di[i] < minus_di[i-1] and
                minus_di[i-1] < minus_di[i-2]):
                return True
    # -DI reversal
    if i >= 3:
        if (minus_di[i-3] is not None and minus_di[i-2] is not None and
            minus_di[i-1] is not None and minus_di[i] is not None and
            plus_di[i] is not None and plus_di[i-2] is not None):
            if (minus_di[i-2] < minus_di[i-3] and
                minus_di[i-1] < minus_di[i-2] and
                minus_di[i] < minus_di[i-1] and
                plus_di[i] > plus_di[i-2]):
                return True
    return False

def adx_sell_signal(i):
    """Return True if an ADX sell signal occurs at candle i."""
    if not adx_trending(i):
        return False
    # DI- trending up, DI+ trending down
    if i >= 2:
        if (plus_di[i] is not None and plus_di[i-1] is not None and plus_di[i-2] is not None and
            minus_di[i] is not None and minus_di[i-1] is not None and minus_di[i-2] is not None):
            if (minus_di[i] > minus_di[i-2] and
                minus_di[i-1] > minus_di[i-2] and
                minus_di[i] > minus_di[i-1] and
                plus_di[i] < plus_di[i-1] and
                plus_di[i-1] < plus_di[i-2]):
                return True
    # +DI reversal
    if i >= 3:
        if (plus_di[i-3] is not None and plus_di[i-2] is not None and
            plus_di[i-1] is not None and plus_di[i] is not None and
            minus_di[i] is not None and minus_di[i-2] is not None):
            if (plus_di[i-2] < plus_di[i-3] and
                plus_di[i-1] < plus_di[i-2] and
                plus_di[i] < plus_di[i-1] and
                minus_di[i] > minus_di[i-2]):
                return True
    return False

# Generate labels: 0 = hold, 1 = buy, 2 = sell
labels = []
for i in range(len(df)):
    buy = False
    sell = False
    if not STOCH_BYPASS:
        if stoch_buy_signal(i):
            buy = True
        if stoch_sell_signal(i):
            sell = True
    if not ADX_BYPASS:
        if adx_buy_signal(i):
            buy = True
        if adx_sell_signal(i):
            sell = True
    if buy and not sell:
        labels.append(1)
    elif sell and not buy:
        labels.append(2)
    else:
        labels.append(0)

df['target'] = labels

print(f"Label distribution: buy={labels.count(1)}, sell={labels.count(2)}, hold={labels.count(0)}")

# --- PREPARE SLIDING WINDOW FEATURES ---
features = ['feat_body', 'feat_range', 'feat_stoch_k', 'feat_stoch_d',
            'feat_adx', 'feat_plus_di', 'feat_minus_di']

X = []
y = []
for i in range(window, len(df)):
    window_data = df[features].iloc[i-window:i].values.flatten()
    X.append(window_data)
    y.append(df['target'].iloc[i])

X = np.array(X).astype(np.float32)
y = np.array(y)

print(f"Training samples: {len(X)}")

# --- MODEL TRAINING ---
param_dist = {
    'n_estimators': [100, 150, 200],
    'max_depth': [5, 8, 12],
    'min_samples_leaf': [1, 5]
}
tscv = TimeSeriesSplit(n_splits=2)
search = RandomizedSearchCV(
    RandomForestClassifier(random_state=42),
    param_distributions=param_dist,
    n_iter=5,
    cv=tscv,
    scoring='accuracy',
    n_jobs=-1
)
search.fit(X, y)
model = search.best_estimator_
print(f"Best configuration: {search.best_params_}")

# --- EXPORT TO ONNX ---
initial_type = [('float_input', FloatTensorType([None, len(features)*window]))]
onx = convert_sklearn(model, initial_types=initial_type, target_opset=12,
                      options={type(model): {'zipmap': False}})
onnx.checker.check_model(onx)

with open(output_filename, "wb") as f:
    f.write(onx.SerializeToString())

print(f"Model saved to: {output_filename}")
print(f"Opset version: 12 (MT5 compatible)")
print(f"--- PROCESS COMPLETED ---")