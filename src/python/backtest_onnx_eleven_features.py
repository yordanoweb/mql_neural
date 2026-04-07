import pandas as pd
import numpy as np
import sys
import os
import argparse
import time
from pathlib import Path
import onnxruntime as ort
import ta
import vectorbt as vbt
from itertools import product

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

# --- ENHANCED FEATURE FUNCTIONS (same as training) ---

def calculate_stochastic_features(high, low, close, window=14, smooth_k=3, smooth_d=3):
    """Calculate enhanced stochastic features."""
    stoch = ta.momentum.StochasticOscillator(high, low, close, 
                                             window=window, 
                                             smooth_window=smooth_k)
    
    stoch_k = stoch.stoch()
    stoch_d = stoch.stoch_signal()
    
    feat_stoch_momentum = safe_series((stoch_k - stoch_d) / 100.0)
    feat_stoch_position = safe_series((stoch_k - 50.0) / 50.0)
    feat_stoch_velocity = safe_series(stoch_k.diff() / 100.0)
    
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
    """Calculate enhanced volume features."""
    vol_ma  = tick_volume.rolling(window=window).mean()
    vol_std = tick_volume.rolling(window=window).std()
    
    denom_ma = vol_ma.copy()
    denom_ma[denom_ma.abs() < 1e-10] = 1.0
    feat_vol_ratio = safe_series(tick_volume / denom_ma, clip=5.0)
    
    vol_ema_fast = tick_volume.ewm(span=5,  adjust=False).mean()
    vol_ema_slow = tick_volume.ewm(span=20, adjust=False).mean()
    denom_slow = vol_ema_slow.copy()
    denom_slow[denom_slow.abs() < 1e-10] = 1.0
    feat_vol_momentum = safe_series((vol_ema_fast - vol_ema_slow) / denom_slow, clip=5.0)
    
    price_change = safe_series(close.pct_change().abs(), clip=5.0)
    vol_change   = safe_series(tick_volume.pct_change().abs(), clip=5.0)
    feat_vol_price_div = safe_series(vol_change - price_change, clip=5.0)
    
    feat_vol_percentile = tick_volume.rolling(window=window).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] if len(x) > 0 else 0.5
    )
    feat_vol_percentile = safe_series((feat_vol_percentile - 0.5) * 2)
    
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

def prepare_features(df, window, atr_period, stoch_window, vol_window):
    """Prepare all features matching the training script."""
    
    # Calculate ATR
    atr_indicator = ta.volatility.AverageTrueRange(
        high=df['high'], low=df['low'], close=df['close'], window=atr_period
    )
    df['atr'] = atr_indicator.average_true_range()
    
    # Basic features
    atr_safe = df['atr'].replace(0, np.nan).fillna(method='ffill').fillna(1.0)
    df['feat_body']  = safe_series((df['close'] - df['open']) / atr_safe)
    df['feat_range'] = safe_series((df['high']  - df['low'])  / atr_safe)
    
    # Stochastic features
    stoch_features = calculate_stochastic_features(df['high'], df['low'], df['close'], 
                                                   window=stoch_window)
    for key, value in stoch_features.items():
        df[key] = value
    
    # Volume features
    volume_features = calculate_volume_features(df['tick_volume'], df['close'], 
                                               window=vol_window)
    for key, value in volume_features.items():
        df[key] = value
    
    return df

def generate_predictions(df, model_session, window, features):
    """Generate predictions using the ONNX model."""
    
    predictions = np.zeros(len(df))
    probabilities = np.zeros(len(df))
    
    for i in range(window, len(df)):
        window_data = df[features].iloc[i-window:i].values.flatten()
        window_data = np.nan_to_num(window_data, nan=0.0, posinf=0.0, neginf=0.0)
        
        X_input = window_data.astype(np.float32).reshape(1, -1)
        
        input_name = model_session.get_inputs()[0].name
        output_names = [out.name for out in model_session.get_outputs()]
        
        outputs = model_session.run(output_names, {input_name: X_input})
        
        pred_label = outputs[0][0]
        pred_proba = outputs[1][0]
        
        predictions[i] = pred_label
        probabilities[i] = pred_proba[1] if len(pred_proba) > 1 else pred_proba[0]
    
    return predictions, probabilities

def run_backtest_combination(df, predictions, probabilities, 
                            threshold, tp_atr, sl_atr, 
                            hold_bars, min_prob):
    """Run backtest for a specific parameter combination."""
    
    # Generate entry signals
    entries = (probabilities >= min_prob) & (predictions == 1)
    
    # Create price and position arrays
    close_prices = df['close'].values
    atr_values = df['atr'].values
    
    # Initialize position tracking
    position = np.zeros(len(df))
    pnl = np.zeros(len(df))
    entry_price = 0
    entry_bar = -1
    
    for i in range(len(df)):
        # Check for entry
        if entries[i] and position[i-1] == 0 if i > 0 else entries[i]:
            position[i] = 1
            entry_price = close_prices[i]
            entry_bar = i
        
        # Check for exit if in position
        elif position[i-1] == 1 if i > 0 else False:
            current_price = close_prices[i]
            current_atr = atr_values[entry_bar] if entry_bar >= 0 else atr_values[i]
            bars_held = i - entry_bar
            
            # Calculate profit/loss in ATR multiples
            profit_atr = (current_price - entry_price) / current_atr if current_atr != 0 else 0
            
            # Exit conditions
            exit_signal = False
            
            # Take profit
            if profit_atr >= tp_atr:
                exit_signal = True
                pnl[i] = profit_atr * current_atr
            
            # Stop loss
            elif profit_atr <= -sl_atr:
                exit_signal = True
                pnl[i] = -sl_atr * current_atr
            
            # Max holding period
            elif bars_held >= hold_bars:
                exit_signal = True
                pnl[i] = profit_atr * current_atr
            
            if exit_signal:
                position[i] = 0
            else:
                position[i] = 1
    
    return position, pnl

# --- CONFIGURATION ---
parser = argparse.ArgumentParser(description="Backtest ONNX model with vectorbt optimization")
parser.add_argument("--input_csv",      type=str, required=True, help="Path to the input CSV file")
parser.add_argument("--onnx_model",     type=str, required=True, help="Path to the ONNX model file")
parser.add_argument("--output_dir",     type=str, default=".",   help="Directory to save results")

# Fixed parameters (must match training)
parser.add_argument("--window",         type=int, required=True, help="Window size used in training")
parser.add_argument("--atr_period",     type=int, default=14,    help="ATR period used in training")
parser.add_argument("--stoch_window",   type=int, default=14,    help="Stochastic window used in training")
parser.add_argument("--vol_window",     type=int, default=20,    help="Volume window used in training")

# Variable parameters for optimization
parser.add_argument("--min_prob_range", type=str, default="0.5,0.6,0.7,0.8", 
                    help="Comma-separated minimum probability thresholds")
parser.add_argument("--tp_atr_range",   type=str, default="1.0,1.5,2.0,2.5,3.0", 
                    help="Comma-separated take profit in ATR multiples")
parser.add_argument("--sl_atr_range",   type=str, default="0.5,1.0,1.5,2.0", 
                    help="Comma-separated stop loss in ATR multiples")
parser.add_argument("--hold_bars_range", type=str, default="5,10,15,20,30", 
                    help="Comma-separated max holding periods in bars")

args = parser.parse_args()

start_time = time.time()

# Parse ranges
min_prob_values = [float(x) for x in args.min_prob_range.split(',')]
tp_atr_values = [float(x) for x in args.tp_atr_range.split(',')]
sl_atr_values = [float(x) for x in args.sl_atr_range.split(',')]
hold_bars_values = [int(x) for x in args.hold_bars_range.split(',')]

print(colorize("=" * 80, Colors.CYAN))
print(colorize("ONNX MODEL BACKTESTING WITH VECTORBT OPTIMIZATION", Colors.CYAN))
print(colorize("=" * 80, Colors.CYAN))
print(f"CSV file:     {colorize(args.input_csv, Colors.WHITE)}")
print(f"ONNX model:   {colorize(args.onnx_model, Colors.WHITE)}")
print(f"Window size:  {colorize(str(args.window), Colors.YELLOW)}")
print(f"ATR period:   {colorize(str(args.atr_period), Colors.YELLOW)}")

# Check files exist
if not os.path.exists(args.input_csv):
    print(colorize(f"Error: CSV file '{args.input_csv}' not found", Colors.RED))
    sys.exit(1)

if not os.path.exists(args.onnx_model):
    print(colorize(f"Error: ONNX model '{args.onnx_model}' not found", Colors.RED))
    sys.exit(1)

# Load ONNX model
print(colorize("\n[1/5] Loading ONNX model...", Colors.BLUE))
session = ort.InferenceSession(args.onnx_model)
print(f"  ✓ Model loaded: {colorize(args.onnx_model, Colors.GREEN)}")

# Load data
print(colorize("\n[2/5] Loading and preparing data...", Colors.BLUE))
df = pd.read_csv(args.input_csv)
print(f"  ✓ Rows loaded: {colorize(str(len(df)), Colors.GREEN)}")

# Prepare features
print(colorize("\n[3/5] Calculating features...", Colors.BLUE))
df = prepare_features(df, args.window, args.atr_period, args.stoch_window, args.vol_window)
df.dropna(inplace=True)
print(f"  ✓ Rows after cleaning: {colorize(str(len(df)), Colors.GREEN)}")

# Define feature list (must match training)
features = [
    'feat_body',
    'feat_range',
    'feat_stoch_momentum',
    'feat_stoch_position',
    'feat_stoch_velocity',
    'feat_stoch_divergence',
    'feat_vol_ratio',
    'feat_vol_momentum',
    'feat_vol_price_div',
    'feat_vol_percentile',
    'feat_vol_zscore'
]

# Generate predictions
print(colorize("\n[4/5] Generating predictions with ONNX model...", Colors.BLUE))
predictions, probabilities = generate_predictions(df, session, args.window, features)
df['prediction'] = predictions
df['probability'] = probabilities

positive_signals = (predictions == 1).sum()
print(f"  ✓ Total predictions: {colorize(str(len(predictions)), Colors.GREEN)}")
print(f"  ✓ Buy signals: {colorize(str(positive_signals), Colors.GREEN)} "
      f"({colorize('{:.2f}%'.format(positive_signals/len(predictions)*100), Colors.YELLOW)})")

# Optimization with vectorbt
print(colorize("\n[5/5] Running parameter optimization...", Colors.BLUE))
print(colorize("=" * 80, Colors.CYAN))
print(f"Parameter combinations to test:")
print(f"  • Min Probability: {min_prob_values}")
print(f"  • Take Profit (ATR): {tp_atr_values}")
print(f"  • Stop Loss (ATR): {sl_atr_values}")
print(f"  • Hold Bars: {hold_bars_values}")

total_combinations = (len(min_prob_values) * len(tp_atr_values) * 
                     len(sl_atr_values) * len(hold_bars_values))
print(f"\n{colorize('Total combinations:', Colors.WHITE)} {colorize(str(total_combinations), Colors.MAGENTA)}")

# Run all combinations
results = []
combination_count = 0

for min_prob, tp_atr, sl_atr, hold_bars in product(min_prob_values, tp_atr_values, 
                                                     sl_atr_values, hold_bars_values):
    combination_count += 1
    
    if combination_count % 10 == 0:
        print(f"  Testing combination {combination_count}/{total_combinations}...", end='\r')
    
    position, pnl = run_backtest_combination(df, predictions, probabilities,
                                             threshold=0.5, tp_atr=tp_atr, 
                                             sl_atr=sl_atr, hold_bars=hold_bars,
                                             min_prob=min_prob)
    
    # Calculate metrics
    total_trades = (np.diff(position) == 1).sum()
    if total_trades > 0:
        total_pnl = pnl.sum()
        winning_trades = (pnl > 0).sum()
        losing_trades = (pnl < 0).sum()
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        avg_win = pnl[pnl > 0].mean() if winning_trades > 0 else 0
        avg_loss = abs(pnl[pnl < 0].mean()) if losing_trades > 0 else 0
        profit_factor = (pnl[pnl > 0].sum() / abs(pnl[pnl < 0].sum())) if losing_trades > 0 else float('inf')
        
        # Calculate max drawdown
        cumulative_pnl = pnl.cumsum()
        running_max = np.maximum.accumulate(cumulative_pnl)
        drawdown = running_max - cumulative_pnl
        max_drawdown = drawdown.max() if len(drawdown) > 0 else 0
        
        results.append({
            'min_prob': min_prob,
            'tp_atr': tp_atr,
            'sl_atr': sl_atr,
            'hold_bars': hold_bars,
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate': win_rate,
            'total_pnl': total_pnl,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'max_drawdown': max_drawdown,
            'sharpe_ratio': total_pnl / max_drawdown if max_drawdown > 0 else 0
        })

print(f"\n  ✓ Completed {colorize(str(total_combinations), Colors.GREEN)} combinations")

# Convert to DataFrame and sort by total PnL
results_df = pd.DataFrame(results)
results_df = results_df.sort_values('total_pnl', ascending=False)

# Display top results
print(colorize("\n" + "=" * 80, Colors.GREEN))
print(colorize("TOP 10 PARAMETER COMBINATIONS BY TOTAL PNL", Colors.GREEN))
print(colorize("=" * 80, Colors.GREEN))

top_10 = results_df.head(10)
for idx, row in top_10.iterrows():
    print(f"\n{colorize('Rank #{}'.format(top_10.index.get_loc(idx) + 1), Colors.CYAN)}")
    print(f"  Min Prob: {row['min_prob']:.2f} | TP: {row['tp_atr']:.1f}x | "
          f"SL: {row['sl_atr']:.1f}x | Hold: {row['hold_bars']} bars")
    print(f"  Trades: {int(row['total_trades'])} | Win Rate: {row['win_rate']*100:.2f}% | "
          f"PnL: {colorize('{:.2f}'.format(row['total_pnl']), Colors.GREEN)}")
    print(f"  Profit Factor: {row['profit_factor']:.2f} | Max DD: {row['max_drawdown']:.2f}")

# Save results to CSV
output_file = os.path.join(args.output_dir, 
                          f"backtest_results_{Path(args.onnx_model).stem}.csv")
results_df.to_csv(output_file, index=False)

print(colorize(f"\n✓ Results saved to: {output_file}", Colors.GREEN))

# Best combination
best = results_df.iloc[0]
print(colorize("\n" + "=" * 80, Colors.MAGENTA))
print(colorize("BEST PARAMETER COMBINATION", Colors.MAGENTA))
print(colorize("=" * 80, Colors.MAGENTA))
print(f"""Min Probability:  {colorize('{:.2f}'.format(best['min_prob']), Colors.YELLOW)}""")
print(f"""Take Profit:      {colorize('{:.1f}x ATR'.format(best['tp_atr']), Colors.YELLOW)}""")
print(f"""Stop Loss:        {colorize('{:.1f}x ATR'.format(best['sl_atr']), Colors.YELLOW)}""")
print(f"""Max Hold Bars:    {colorize(str(int(best['hold_bars'])), Colors.YELLOW)}""")
print(f"""Total Trades:     {colorize(str(int(best['total_trades'])), Colors.GREEN)}""")
print(f"""Win Rate:         {colorize('{:.2f}%'.format(best['win_rate']*100), Colors.GREEN)}""")
print(f"""Total PnL:        {colorize('{:.2f}'.format(best['total_pnl']), Colors.GREEN)}""")
print(f"""Profit Factor:    {colorize('{:.2f}'.format(best['profit_factor']), Colors.GREEN)}""")
print(f"""Max Drawdown:     {colorize('{:.2f}'.format(best['max_drawdown']), Colors.YELLOW)}""")
        
print(colorize("\n" + "=" * 80, Colors.CYAN))
print(colorize("BACKTESTING COMPLETED SUCCESSFULLY!", Colors.CYAN))
print(colorize("=" * 80, Colors.CYAN))

end_time = time.time()
print(colorize(f"\nTotal execution time: {int(end_time - start_time) // 60} minutes "
               f"and {int(end_time - start_time) % 60} seconds\n", Colors.GREEN))
