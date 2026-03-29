import pandas as pd
import numpy as np
import sys
import os
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
import onnx
import argparse

# ---------- Color setup ----------
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

# ---------- CONFIGURATION ----------
parser = argparse.ArgumentParser(
    description="Train ONNX model from CSV data with 3 features (body/ATR, range/ATR, VSA ratio)"
)
parser.add_argument("--input_csv",          type=str,   required=True,  help="Path to the input CSV file")
parser.add_argument("--output_dir",         type=str,   default=".",    help="Directory to save the ONNX model (default: current directory)")
parser.add_argument("--atr_period",         type=int,   default=14,     help="Period for ATR calculation used for normalization (default: 14)")
parser.add_argument("--vsa_ma_period",      type=int,   default=20,     help="Period for the rolling mean used to normalize the VSA ratio (default: 20)")
parser.add_argument("--window",             type=int,   default=20,     help="Window size (number of bars) for feature sequences")
parser.add_argument("--future",             type=int,   default=5,      help="Number of bars to look into the future for target labeling")
parser.add_argument("--n_iter",             type=int,   default=5,      help="Number of iterations for RandomizedSearchCV")
parser.add_argument("--min_profit_atr",     type=float, default=0.5,    help="Minimum profit expressed in ATR multiples for a positive target (default: 0.5)")

args = parser.parse_args()

csv_file        = args.input_csv
output_dir      = args.output_dir
atr_period      = args.atr_period
vsa_ma_period   = args.vsa_ma_period
window          = args.window
future          = args.future
n_iter          = args.n_iter
min_profit_atr  = args.min_profit_atr

if not os.path.exists(csv_file):
    print(colorize(f"Error: File '{csv_file}' not found", Colors.RED))
    sys.exit(1)

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

output_filename = os.path.join(output_dir, Path(csv_file).stem + f"_w{window}_f{future}_vsa{vsa_ma_period}_atr{atr_period}_min{min_profit_atr}.onnx")

print(colorize("--- FAST TRAINING ---", Colors.CYAN))
print(f"Loading rates from: {colorize(csv_file, Colors.WHITE)}")
print(f"Output ONNX will be: {colorize(output_filename, Colors.YELLOW)}")

# ---------- 1. LOAD DATA ----------
df = pd.read_csv(csv_file)
print(f"Rows loaded: {colorize(str(len(df)), Colors.GREEN)}")

# Validate required columns
required_cols = {'open', 'high', 'low', 'close', 'tick_volume'}
missing = required_cols - set(df.columns.str.lower())
if missing:
    print(colorize(f"Error: CSV is missing columns: {missing}", Colors.RED))
    sys.exit(1)

df.columns = df.columns.str.lower()

# ---------- 2. FEATURE ENGINEERING ----------
# --- ATR (Wilder's smoothed method, manual implementation to avoid library dependency) ---
prev_close = df['close'].shift(1)
tr = pd.concat([
    df['high'] - df['low'],
    (df['high'] - prev_close).abs(),
    (df['low']  - prev_close).abs()
], axis=1).max(axis=1)

# Use EWM with adjust=False to replicate Wilder's smoothing: alpha = 1 / atr_period
atr = tr.ewm(alpha=1.0 / atr_period, adjust=False).mean()

# Guard against zero ATR (flat markets or missing data)
atr = atr.replace(0, np.nan)

# --- Feature 1: Body / ATR (signed: positive = bullish, negative = bearish) ---
# Dimensionless; comparable across all instruments and timeframes.
df['feat_body'] = (df['close'] - df['open']) / atr

# --- Feature 2: Range / ATR ---
# Values near 1.0 = average volatility bar; >1 = expansion; <1 = compression.
df['feat_range'] = (df['high'] - df['low']) / atr

# --- Feature 3: VSA Ratio (Volume Spread Analysis) ---
# Concept: volume relative to spread measures market effort vs. result.
#   - High volume + narrow range  → absorption / potential reversal
#   - Low volume  + wide range    → weak move, likely to fail
#   - Low volume  + narrow range  → consolidation
# We normalize by the rolling mean so the feature is stationary and
# comparable across instruments and different phases of market activity.
bar_spread = (df['high'] - df['low']).replace(0, np.nan)
raw_vsa    = df['tick_volume'] / bar_spread              # effort-per-point
vsa_mean   = raw_vsa.rolling(window=vsa_ma_period).mean().replace(0, np.nan)
df['feat_vsa'] = raw_vsa / vsa_mean                      # normalized: 1.0 = average effort

print(colorize(
    f"Features: feat_body (body/ATR), feat_range (range/ATR), feat_vsa (VSA ratio normalized by {vsa_ma_period}-bar mean)",
    Colors.BLUE
))

# ---------- 3. TARGET LABELING (ATR-based, instrument-agnostic) ----------
print(
    f"Generating target: future={colorize(str(future), Colors.MAGENTA)} bars, "
    f"min_profit={colorize(str(min_profit_atr), Colors.MAGENTA)} ATR multiples..."
)

labels = np.zeros(len(df))
for i in range(len(df) - future):
    entry_price   = df['close'].iloc[i]
    bar_atr       = atr.iloc[i]
    if np.isnan(bar_atr) or bar_atr == 0:
        continue
    future_highs  = df['high'].iloc[i+1 : i+future+1]
    profit_in_atr = (future_highs.max() - entry_price) / bar_atr
    if profit_in_atr >= min_profit_atr:
        labels[i] = 1

df['target'] = labels
df.dropna(inplace=True)

pos_rate = df['target'].mean() * 100
print(f"Positive label rate: {colorize(f'{pos_rate:.1f}%', Colors.YELLOW)}")

# ---------- 4. PREPARE WINDOWS ----------
features = ['feat_body', 'feat_range', 'feat_vsa']
X, y = [], []

for i in range(window, len(df) - future):
    window_data = df[features].iloc[i-window:i].values.flatten()
    X.append(window_data)
    y.append(df['target'].iloc[i])

X = np.array(X, dtype=np.float32)
y = np.array(y)

# Clip extreme VSA values that can arise from extremely thin spreads
X = np.clip(X, -20.0, 20.0)

print(f"Dataset shape: X={X.shape}, y={y.shape}")

# ---------- 5. HYPERPARAMETER SEARCH ----------
print(colorize("Searching for efficient configuration (Random Search)...", Colors.YELLOW))

param_dist = {
    'n_estimators':    [100, 150, 200],
    'max_depth':       [5, 8, 12],
    'min_samples_leaf':[1, 5]
}

tscv = TimeSeriesSplit(n_splits=2)

search = RandomizedSearchCV(
    RandomForestClassifier(random_state=42, class_weight='balanced'),
    param_distributions=param_dist,
    n_iter=n_iter,
    cv=tscv,
    scoring='balanced_accuracy',
    n_jobs=4,
    verbose=2
)

print(colorize(f"Starting training with {X.shape[0]} samples...", Colors.CYAN))
search.fit(X, y)
model = search.best_estimator_
print(colorize(f"Best configuration: {search.best_params_}", Colors.GREEN))

# ---------- 6. EXPORT TO ONNX ----------
# Input shape: [batch, window * 3]  →  e.g. [None, 60] for window=20
initial_type = [('float_input', FloatTensorType([None, window * len(features)]))]
onx = convert_sklearn(
    model,
    initial_types=initial_type,
    target_opset=12,                          # MT5-compatible opset
    options={type(model): {'zipmap': False}}  # return plain probability array
)

onnx.checker.check_model(onx)

with open(output_filename, "wb") as f:
    f.write(onx.SerializeToString())

print(colorize(f"Model saved at: {output_filename}", Colors.GREEN))
print(colorize("--- PROCESS COMPLETED ---", Colors.CYAN))
print()
print(colorize("Feature summary for MQL5 EA:", Colors.BLUE))
print(f"  feat_body  [idx 0] = (close - open) / ATR({atr_period})           → signed body in ATR units")
print(f"  feat_range [idx 1] = (high  - low ) / ATR({atr_period})           → range in ATR units")
print(f"  feat_vsa   [idx 2] = (tick_volume / spread) / rolling_mean({vsa_ma_period}) → normalized effort ratio")
