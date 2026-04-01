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
from datetime import datetime

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

# --- CONFIGURATION ---
parser = argparse.ArgumentParser(description="Train ONNX model to find entry opportunities in specific time window")
parser.add_argument("--input_csv", type=str, required=True, help="Path to the input CSV file")
parser.add_argument("--output_dir", type=str, default=".", help="Directory to save the ONNX model (default: current directory)")
parser.add_argument("--window", type=int, default=20, help="Window size (number of bars) for features")
parser.add_argument("--future", type=int, default=10, help="Number of bars to look into the future for target")
parser.add_argument("--n_iter", type=int, default=5, help="Number of iterations for RandomizedSearchCV")
parser.add_argument("--target_pct", type=float, default=0.5, help="Target percentage move (default: 0.5 for 0.5%%)")
parser.add_argument("--time_start", type=int, default=1, help="Start hour for trading window (0-23)")
parser.add_argument("--time_end", type=int, default=3, help="End hour for trading window (0-23)")

args = parser.parse_args()

csv_file = args.input_csv
output_dir = args.output_dir
window = args.window
future = args.future
n_iter = args.n_iter
target_pct = args.target_pct
time_start = args.time_start
time_end = args.time_end

if not os.path.exists(csv_file):
    print(colorize(f"Error: File '{csv_file}' not found", Colors.RED))
    sys.exit(1)

# Ensure output directory exists
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Generate output filename
output_filename = os.path.join(
    output_dir, 
    Path(csv_file).stem + f"_w{window}_f{future}_pct{target_pct}_h{time_start}-{time_end}.onnx"
)
output_filename = output_filename.replace("_rates", "")

print(colorize("--- TIMEFRAME ENTRY TRAINING ---", Colors.CYAN))
print(f"Loading rates from: {colorize(csv_file, Colors.WHITE)}")
print(f"Trading window: {colorize(f'{time_start}:00 - {time_end}:00', Colors.MAGENTA)}")
print(f"Target move: {colorize(f'±{target_pct}%', Colors.MAGENTA)}")
print(f"Output ONNX will be: {colorize(output_filename, Colors.YELLOW)}")

# 1. LOAD DATA FROM CSV
df = pd.read_csv(csv_file)
print(f"Rows loaded: {colorize(str(len(df)), Colors.GREEN)}")

# Parse datetime if exists
if 'time' in df.columns:
    df['datetime'] = pd.to_datetime(df['time'])
elif 'datetime' in df.columns:
    df['datetime'] = pd.to_datetime(df['datetime'])
else:
    print(colorize("Warning: No 'time' or 'datetime' column found. Using all data.", Colors.YELLOW))
    df['datetime'] = pd.NaT

# Calculate features (only body and range)
df['feat_body'] = df['close'] - df['open']
df['feat_range'] = df['high'] - df['low']

# 2. GENERATE TARGET (Multi-class: 0=no_trade, 1=long, 2=short)
print(f"Generating target with future={colorize(str(future), Colors.MAGENTA)} bars and target={colorize(str(target_pct), Colors.MAGENTA)}%...")

labels = np.zeros(len(df))  # 0 = no trade
target_threshold = target_pct / 100.0  # Convert percentage to decimal

for i in range(len(df) - future):
    # Only consider entries within the specified time window
    if pd.notna(df['datetime'].iloc[i]):
        hour = df['datetime'].iloc[i].hour
        # Check if hour is in range (handle wrap-around if needed)
        if time_start <= time_end:
            if not (time_start <= hour < time_end):
                continue
        else:  # Wrap around midnight (e.g., 22-2)
            if not (hour >= time_start or hour < time_end):
                continue
    
    entry_price = df['close'].iloc[i]
    
    # Look at future prices
    future_highs = df['high'].iloc[i+1 : i+future+1]
    future_lows = df['low'].iloc[i+1 : i+future+1]
    
    # Calculate percentage moves
    max_upward = (future_highs.max() - entry_price) / entry_price
    max_downward = (entry_price - future_lows.min()) / entry_price
    
    # Determine trade type
    if max_upward >= target_threshold:
        labels[i] = 1  # LONG entry
    elif max_downward >= target_threshold:
        labels[i] = 2  # SHORT entry
    # else: remains 0 (no trade)

df['target'] = labels

# Count labels
unique, counts = np.unique(labels, return_counts=True)
label_dist = dict(zip(unique, counts))
print(colorize(f"Label distribution:", Colors.CYAN))
print(f"  No trade (0): {colorize(str(label_dist.get(0, 0)), Colors.WHITE)}")
print(f"  Long (1): {colorize(str(label_dist.get(1, 0)), Colors.GREEN)}")
print(f"  Short (2): {colorize(str(label_dist.get(2, 0)), Colors.RED)}")

df.dropna(inplace=True)

# 3. PREPARE WINDOWS
X, y = [], []
features = ['feat_body', 'feat_range']

for i in range(window, len(df) - future):
    window_data = df[features].iloc[i-window:i].values.flatten()
    X.append(window_data)
    y.append(df['target'].iloc[i])

X = np.array(X).astype(np.float32)
y = np.array(y)

print(f"Training samples: {colorize(str(len(X)), Colors.GREEN)}")

# Filter out if we have too few samples
if len(X) < 100:
    print(colorize(f"Error: Not enough samples ({len(X)}). Need at least 100.", Colors.RED))
    sys.exit(1)

# 4. FAST OPTIMIZATION (Random Search)
print(colorize("Searching for efficient configuration (Random Search)...", Colors.YELLOW))
param_dist = {
    'n_estimators': [100, 150, 200],
    'max_depth': [5, 8, 12, 15],
    'min_samples_leaf': [1, 3, 5],
    'min_samples_split': [2, 5, 10]
}

# TimeSeriesSplit with 2 folds for speed
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
print(colorize(f"Best score: {search.best_score_:.4f}", Colors.GREEN))

# 5. EXPORT WITH CSV-BASED NAME
initial_type = [('float_input', FloatTensorType([None, window * 2]))]  # 2 features now
# Use target_opset=12 for MetaTrader 5 compatibility
onx = convert_sklearn(model, initial_types=initial_type, target_opset=12, options={type(model): {'zipmap': False}})

# Validate ONNX model before saving
onnx.checker.check_model(onx)

with open(output_filename, "wb") as f:
    f.write(onx.SerializeToString())

print(colorize(f"Model saved at: {output_filename}", Colors.GREEN))
print(colorize("--- PROCESS COMPLETED ---", Colors.CYAN))
print(colorize("Model output classes:", Colors.CYAN))
print(f"  0 = No trade")
print(f"  1 = LONG entry (expecting {target_pct}% upward move)")
print(f"  2 = SHORT entry (expecting {target_pct}% downward move)")
