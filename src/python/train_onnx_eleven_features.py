import pandas as pd
import numpy as np
import sys
import os
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
import ta
import onnx
import onnx.helper
import argparse
import time

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

def safe_series(s, fill=0.0, clip=10.0):
    """Replace inf/-inf with NaN, fill NaN with fill, then clip to [-clip, clip]."""
    return s.replace([np.inf, -np.inf], np.nan).fillna(fill).clip(-clip, clip)

# --- ENHANCED FEATURE FUNCTIONS ---

def calculate_stochastic_features(high, low, close, window=14, smooth_k=3, smooth_d=3):
    """
    Calculate enhanced stochastic features:
    1. Stochastic momentum (K - D)
    2. Stochastic position (normalized K value)
    3. Stochastic velocity (rate of change of K)
    4. Stochastic divergence zones (overbought/oversold pressure)
    """
    stoch = ta.momentum.StochasticOscillator(high, low, close, 
                                             window=window, 
                                             smooth_window=smooth_k)
    
    stoch_k = stoch.stoch()
    stoch_d = stoch.stoch_signal()
    
    # Feature 1: Momentum (K - D) normalized to [-1, 1]
    feat_stoch_momentum = safe_series((stoch_k - stoch_d) / 100.0)
    
    # Feature 2: Position - where is K in its range (centered around 50)
    feat_stoch_position = safe_series((stoch_k - 50.0) / 50.0)  # Range [-1, 1]
    
    # Feature 3: Velocity (rate of change of K)
    feat_stoch_velocity = safe_series(stoch_k.diff() / 100.0)
    
    # Feature 4: Divergence pressure zones
    overbought_pressure = np.where(stoch_k > 80, -(stoch_k - 80) / 20.0, 0)
    oversold_pressure   = np.where(stoch_k < 20, (20 - stoch_k) / 20.0, 0)
    feat_stoch_divergence = safe_series(
        pd.Series(overbought_pressure + oversold_pressure, index=stoch_k.index)
    )
    
    return {
        'feat_stoch_momentum':  feat_stoch_momentum,
        'feat_stoch_position':  feat_stoch_position,
        'feat_stoch_velocity':  feat_stoch_velocity,
        'feat_stoch_divergence': feat_stoch_divergence
    }

def calculate_volume_features(tick_volume, close, window=20):
    """
    Calculate enhanced volume features:
    1. Volume ratio (current vs MA)
    2. Volume momentum (trend strength)
    3. Volume-price divergence
    4. Volume percentile (relative strength in recent history)
    5. Volume z-score
    """
    vol_ma  = tick_volume.rolling(window=window).mean()
    vol_std = tick_volume.rolling(window=window).std()
    
    # Feature 1: Normalized volume ratio
    denom_ma = vol_ma.copy()
    denom_ma[denom_ma.abs() < 1e-10] = 1.0
    feat_vol_ratio = safe_series(tick_volume / denom_ma, clip=5.0)
    
    # Feature 2: Volume momentum
    vol_ema_fast = tick_volume.ewm(span=5,  adjust=False).mean()
    vol_ema_slow = tick_volume.ewm(span=20, adjust=False).mean()
    denom_slow = vol_ema_slow.copy()
    denom_slow[denom_slow.abs() < 1e-10] = 1.0
    feat_vol_momentum = safe_series((vol_ema_fast - vol_ema_slow) / denom_slow, clip=5.0)
    
    # Feature 3: Volume-Price divergence
    # pct_change can be inf when the previous value is 0 — clip aggressively
    price_change = safe_series(close.pct_change().abs(), clip=5.0)
    vol_change   = safe_series(tick_volume.pct_change().abs(), clip=5.0)
    feat_vol_price_div = safe_series(vol_change - price_change, clip=5.0)
    
    # Feature 4: Volume percentile scaled to [-1, 1]
    feat_vol_percentile = tick_volume.rolling(window=window).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] if len(x) > 0 else 0.5
    )
    feat_vol_percentile = safe_series((feat_vol_percentile - 0.5) * 2)
    
    # Feature 5: Z-score (statistical anomaly detection)
    denom_std = vol_std.copy()
    denom_std[denom_std.abs() < 1e-10] = 1.0
    feat_vol_zscore = safe_series(
        np.clip((tick_volume - vol_ma) / denom_std, -3, 3) / 3.0
    )
    
    return {
        'feat_vol_ratio':      feat_vol_ratio,
        'feat_vol_momentum':   feat_vol_momentum,
        'feat_vol_price_div':  feat_vol_price_div,
        'feat_vol_percentile': feat_vol_percentile,
        'feat_vol_zscore':     feat_vol_zscore
    }

# --- CONFIGURATION ---
parser = argparse.ArgumentParser(description="Train ONNX model with ENHANCED features")
parser.add_argument("--input_csv",       type=str,   required=True, help="Path to the input CSV file (required)")
parser.add_argument("--output_dir",      type=str,   default=".",   help="Directory to save the ONNX model (default: .)")
parser.add_argument("--atr_period",      type=int,   default=14,    help="Period for ATR calculation (default: 14)")
parser.add_argument("--window",          type=int,   default=20,    help="Window size (number of bars) for features (default: 20)")
parser.add_argument("--future",          type=int,   default=10,    help="Number of bars to look into the future (default: 10)")
parser.add_argument("--n_iter",          type=int,   default=10,    help="Number of iterations for RandomizedSearchCV (default: 10)")
parser.add_argument("--min_profit_atr",  type=float, default=1.5,   help="Minimum profit in ATR multiples (default: 1.5)")
parser.add_argument("--stoch_window",    type=int,   default=14,    help="Stochastic period (default: 14)")
parser.add_argument("--vol_window",      type=int,   default=20,    help="Volume analysis window (default: 20)")
parser.add_argument("--jobs",            type=int,   default=3,     help="Number of parallel jobs (default: 3)")

args = parser.parse_args()

csv_file       = args.input_csv
output_dir     = args.output_dir
atr_period     = args.atr_period
window         = args.window
future         = args.future
n_iter         = args.n_iter
min_profit_atr = args.min_profit_atr
stoch_window   = args.stoch_window
vol_window     = args.vol_window
jobs           = args.jobs

start_time = time.time()

if not os.path.exists(csv_file):
    print(colorize(f"Error: File '{csv_file}' not found", Colors.RED))
    sys.exit(1)

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

output_filename = os.path.join(
    output_dir,
    Path(csv_file).stem + f"_enh_w{window}_f{future}_atr{atr_period}_minp{min_profit_atr}.onnx"
)
output_filename = str(output_filename).replace("_rates", "")

print(colorize("=" * 70, Colors.CYAN))
print(colorize("ENHANCED TRAINING WITH ADVANCED STOCHASTIC & VOLUME FEATURES", Colors.CYAN))
print(colorize("=" * 70, Colors.CYAN))
print(f"Loading rates from: {colorize(csv_file, Colors.WHITE)}")
print(f"Output ONNX will be: {colorize(output_filename, Colors.YELLOW)}")

# 1. LOAD DATA
df = pd.read_csv(csv_file)
print(f"Rows loaded: {colorize(str(len(df)), Colors.GREEN)}")

# 2. CALCULATE ATR FOR NORMALIZATION
print(colorize("\n[1/6] Calculating ATR for normalization...", Colors.BLUE))
atr_indicator = ta.volatility.AverageTrueRange(
    high=df['high'], low=df['low'], close=df['close'], window=atr_period
)
df['atr'] = atr_indicator.average_true_range()

# 3. CALCULATE BASIC FEATURES (Body & Range)
print(colorize("[2/6] Calculating basic price features (body, range)...", Colors.BLUE))
atr_safe = df['atr'].replace(0, np.nan).fillna(method='ffill').fillna(1.0)
df['feat_body']  = safe_series((df['close'] - df['open']) / atr_safe)
df['feat_range'] = safe_series((df['high']  - df['low'])  / atr_safe)

# 4. CALCULATE ENHANCED STOCHASTIC FEATURES
print(colorize(f"[3/6] Calculating enhanced stochastic features (window={stoch_window})...", Colors.BLUE))
stoch_features = calculate_stochastic_features(df['high'], df['low'], df['close'], window=stoch_window)
for key, value in stoch_features.items():
    df[key] = value
    print(f"  ✓ {key}")

# 5. CALCULATE ENHANCED VOLUME FEATURES
print(colorize(f"[4/6] Calculating enhanced volume features (window={vol_window})...", Colors.BLUE))
volume_features = calculate_volume_features(df['tick_volume'], df['close'], window=vol_window)
for key, value in volume_features.items():
    df[key] = value
    print(f"  ✓ {key}")

# 6. GENERATE TARGET
print(colorize(f"[5/6] Generating target labels (future={future}, min_profit_atr={min_profit_atr})...", Colors.BLUE))
labels = np.zeros(len(df))
for i in range(len(df) - future):
    if pd.isna(df['atr'].iloc[i]) or df['atr'].iloc[i] == 0:
        continue
    entry_price   = df['close'].iloc[i]
    current_atr   = df['atr'].iloc[i]
    future_prices = df['high'].iloc[i+1 : i+future+1]
    profit        = (future_prices.max() - entry_price) / current_atr
    if profit >= min_profit_atr:
        labels[i] = 1

df['target'] = labels
df.dropna(inplace=True)

# 7. PREPARE TRAINING DATA
print(colorize("[6/6] Preparing training windows...", Colors.BLUE))

features = [
    'feat_body',              # 1
    'feat_range',             # 2
    'feat_stoch_momentum',    # 3
    'feat_stoch_position',    # 4
    'feat_stoch_velocity',    # 5
    'feat_stoch_divergence',  # 6
    'feat_vol_ratio',         # 7
    'feat_vol_momentum',      # 8
    'feat_vol_price_div',     # 9
    'feat_vol_percentile',    # 10
    'feat_vol_zscore'         # 11
]

print(f"\nTotal features: {colorize(str(len(features)), Colors.MAGENTA)}")
for i, feat in enumerate(features, 1):
    print(f"  {i}. {feat}")

X, y = [], []
for i in range(window, len(df) - future):
    window_data = df[features].iloc[i-window:i].values.flatten()
    X.append(window_data)
    y.append(df['target'].iloc[i])

X = np.array(X, dtype=np.float32)
y = np.array(y)

# Final safety net: replace any remaining inf/NaN in X
X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

# Sanity check before fitting
assert np.isfinite(X).all(), "X still contains non-finite values after cleaning!"

print(f"\n{colorize('Training samples:', Colors.WHITE)} {colorize(str(len(X)), Colors.GREEN)}")
print(f"{colorize('Positive samples:', Colors.WHITE)} {colorize(str(int(y.sum())), Colors.GREEN)} ({colorize(f'{(y.sum()/len(y)*100):.2f}%', Colors.YELLOW)})")
print(f"{colorize('Input shape:', Colors.WHITE)} {colorize(str(X.shape), Colors.GREEN)}")

# 8. HYPERPARAMETER OPTIMIZATION
print(colorize("\n" + "=" * 70, Colors.CYAN))
print(colorize("STARTING HYPERPARAMETER OPTIMIZATION", Colors.CYAN))
print(colorize("=" * 70, Colors.CYAN))

param_dist = {
    'n_estimators':      [100, 150, 200, 250],
    'max_depth':         [5, 8, 12, 15],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf':  [1, 2, 4],
    'max_features':      ['sqrt', 'log2', None]
}

tscv = TimeSeriesSplit(n_splits=3)

search = RandomizedSearchCV(
    RandomForestClassifier(random_state=42, class_weight='balanced'),
    param_distributions=param_dist,
    n_iter=n_iter,
    cv=tscv,
    scoring='balanced_accuracy',
    n_jobs=jobs,
    verbose=2
)

search.fit(X, y)
model = search.best_estimator_

print(colorize("\n" + "=" * 70, Colors.GREEN))
print(colorize("OPTIMIZATION RESULTS", Colors.GREEN))
print(colorize("=" * 70, Colors.GREEN))
print(f"Best parameters: {colorize(str(search.best_params_), Colors.YELLOW)}")
print(f"Best CV score:   {colorize(f'{search.best_score_:.4f}', Colors.YELLOW)}")

# Feature importance analysis
feature_importance      = model.feature_importances_
feature_names_expanded  = [f"{feat}_bar{i}" for i in range(window) for feat in features]

importance_df = pd.DataFrame({
    'feature':    feature_names_expanded,
    'importance': feature_importance
}).sort_values('importance', ascending=False)

print(colorize("\nTop 10 Most Important Features:", Colors.CYAN))
print(importance_df.head(10).to_string(index=False))

# 9. EXPORT TO ONNX
print(colorize("\n" + "=" * 70, Colors.CYAN))
print(colorize("EXPORTING TO ONNX", Colors.CYAN))
print(colorize("=" * 70, Colors.CYAN))

initial_type = [('float_input', FloatTensorType([None, window * len(features)]))]
onx = convert_sklearn(model, initial_types=initial_type, target_opset=12,
                      options={type(model): {'zipmap': False}})

onnx.checker.check_model(onx)

# Add feature names metadata
feature_names_str = ",".join(features)
entry = onx.metadata_props.add()
entry.key = "feature_names"
entry.value = feature_names_str
onnx.save(onx, output_filename)

print(colorize(f"\n✓ Model saved at:   {output_filename}", Colors.GREEN))
print(colorize(f"✓ Features count:   {len(features)}",    Colors.GREEN))
print(colorize(f"✓ Window size:      {window}",           Colors.GREEN))
print(colorize(f"✓ Input shape:      [1, {window * len(features)}]", Colors.GREEN))
print(colorize("\n" + "=" * 70, Colors.CYAN))
print(colorize("PROCESS COMPLETED SUCCESSFULLY!", Colors.CYAN))
print(colorize("=" * 70, Colors.CYAN))

end_time = time.time()
print(colorize(f"\nTotal execution time: {int(end_time - start_time) // 60} minutes "
               f"and {int(end_time - start_time) % 60} seconds\n\n", Colors.GREEN))
