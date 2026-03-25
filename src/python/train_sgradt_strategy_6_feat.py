"""
SGRADT 7.1 - Training Script (6 Features with DI+/DI- Indicators)
EMA COMPLETELY REMOVED.
Entry: ADX + Stochastic.
Exit/Labeling: Fixed bar horizon (Target reached within 'future' bars).
DI+/DI- extracted from ADXIndicator.adx_pos() and adx_neg()
"""

import pandas as pd
import numpy as np
import argparse
import time
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
from ta.trend import ADXIndicator # EMAIndicator removed
from ta.momentum import StochasticOscillator
import warnings

warnings.filterwarnings('ignore')

# ANSI color codes
COLOR_INFO = "\033[94m"      # Blue
COLOR_DEBUG = "\033[96m"     # Cyan
COLOR_WARNING = "\033[93m"   # Yellow
COLOR_ERROR = "\033[91m"     # Red
COLOR_NUMBER = "\033[92m"    # Green
COLOR_KEYWORD = "\033[95m"   # Magenta
COLOR_RESET = "\033[0m"      # Reset to default

def format_message_with_colors(message):
    """Apply dynamic colors to numbers and keywords in a message"""
    import re
    
    # Define keywords to highlight (case-insensitive)
    keywords = [
        'BUY', 'SELL', 'TOTAL', 'HOLD', 'Accuracy', 'Processing time', 'Parameters', 
        'Best score', 'Best parameters', 'Model saved', 'Writing ONNX model',
        'Prepared training data', 'Label distribution', 'Starting model training',
        'Training completed', 'Exporting model', 'Signal counts', 'Loaded',
        'Processing', 'Starting SGRADT', 'Arguments', 'Output directory',
        'Training session completed', 'Skipping', 'Insufficient signals'
    ]
    
    # Apply keyword coloring
    for keyword in keywords:
        # Use regex to match whole words only, case-insensitive
        pattern = r'\b' + re.escape(keyword) + r'\b'
        message = re.sub(pattern, f'{COLOR_KEYWORD}{keyword}{COLOR_RESET}', message, flags=re.IGNORECASE)
    
    # Apply number coloring (integers and floats)
    # Match numbers including decimals and negative numbers
    number_pattern = r'-?\b\d+\.?\d*\b'
    def replace_number(match):
        return f'{COLOR_NUMBER}{match.group()}{COLOR_RESET}'
    
    message = re.sub(number_pattern, replace_number, message)
    
    return message

def log_info(message):
    """Print formatted log message with timestamp"""
    formatted_message = format_message_with_colors(message)
    print(f"{COLOR_INFO}[INFO]{COLOR_RESET} {formatted_message}")

def log_debug(message):
    """Print debug log message"""
    formatted_message = format_message_with_colors(message)
    print(f"{COLOR_DEBUG}[DEBUG]{COLOR_RESET} {formatted_message}")

def log_warning(message):
    """Print warning log message"""
    formatted_message = format_message_with_colors(message)
    print(f"{COLOR_WARNING}[WARNING]{COLOR_RESET} {formatted_message}")

def log_error(message):
    """Print error log message"""
    formatted_message = format_message_with_colors(message)
    print(f"{COLOR_ERROR}[ERROR]{COLOR_RESET} {formatted_message}")

def calculate_signals_and_labels(df, args):
    """
    Calcula señales sin rastro de EMA. 
    El éxito de la señal se mide por alcanzar el profit en 'future' barras.
    """
    # Indicadores (ADX y Stochastic solamente)
    adx_inst = ADXIndicator(high=df['high'], low=df['low'], close=df['close'], window=args.adx_period)
    df['adx'] = adx_inst.adx()
    df['plus_di'] = adx_inst.adx_pos()  # +DI using built-in method
    df['minus_di'] = adx_inst.adx_neg()  # -DI using built-in method
    
    stoch = StochasticOscillator(high=df['high'], low=df['low'], close=df['close'], window=args.stoch_k, smooth_window=args.stoch_d)
    df['stoch_k'] = stoch.stoch()
    df['stoch_d'] = stoch.stoch_signal()
    
    # Features (6: body, range, stoch_k, stoch_d, adx, di_diff)
    df['feat_body'] = df['close'] - df['open']
    df['feat_range'] = df['high'] - df['low']
    df['feat_stoch_main'] = df['stoch_k']
    df['feat_stoch_signal'] = df['stoch_d']
    df['feat_adx'] = df['adx']
    df['feat_di_diff'] = df['plus_di'] - df['minus_di']  # Positive on BUY, Negative on SELL
    
    features_list = ['feat_body', 'feat_range', 'feat_stoch_main', 'feat_stoch_signal', 'feat_adx', 'feat_di_diff']
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
    parser = argparse.ArgumentParser(description='SGRADT 7.1 - 6 Features with DI+/DI-')
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
    
    log_info(f"Starting SGRADT 7.1 training (6 Features with DI+ and DI-)")
    log_info(f"Arguments: {vars(args)}")
    log_info(f"Output directory: {output_dir}")
    
    start_time = time.time()
    processed_count = 0
    skipped_count = 0
    
    for csv_path in [Path(p) for p in args.csv]:
        if not csv_path.exists():
            log_warning(f"File does not exist: {csv_path}")
            continue
        
        log_info(f"= Processing: {csv_path.name}")
        file_start_time = time.time()
        df = pd.read_csv(csv_path)
        log_info(f"Loaded {len(df)} rows from {csv_path.name}")
        
        df, labels, features_list = calculate_signals_and_labels(df, args)
        
        count_buy = int(np.sum(labels == 1))
        count_sell = int(np.sum(labels == 2))
        total_signals = count_buy + count_sell
        
        log_info(f"Signal counts - BUY: {count_buy}, SELL: {count_sell}, TOTAL: {total_signals}")
        
        if count_buy < 5 or count_sell < 5:
            log_warning(f"Skipping {csv_path.name}: Insufficient signals (B:{count_buy}, S:{count_sell})")
            skipped_count += 1
            continue
        
        log_info(f"Preparing features with window size: {args.window}")
        X_list, y_list = [], []
        X_vals = df[features_list].values
        for i in range(args.window, len(df)):
            X_list.append(X_vals[i - args.window:i].flatten())
            y_list.append(labels[i])
        
        X = np.array(X_list, dtype=np.float32)
        y = np.array(y_list)  # Fixed the walrus operator usage
        
        log_info(f"Prepared training data: X.shape={X.shape}, y.shape={y.shape}")
        log_info(f"Label distribution in training data: {np.bincount(y.astype(int))}")
        
        # Training
        log_info("Starting model training with RandomizedSearchCV")
        log_info(f"Parameters: n_iter={args.n_iter}, cv=TimeSeriesSplit(n_splits=3)")
        
        model = RandomizedSearchCV(
            RandomForestClassifier(random_state=42, class_weight='balanced'),
            param_distributions={'n_estimators': [100, 200], 'max_depth': [10, 20, None]},
            n_iter=args.n_iter, cv=TimeSeriesSplit(n_splits=3), scoring='balanced_accuracy', n_jobs=-1
        )
        model.fit(X, y)
        
        log_info(f"Training completed. Best score: {model.best_score_:.4f}")
        log_info(f"Best parameters: {model.best_params_}")
        
        # Export ONNX
        log_info("Exporting model to ONNX format")
        num_inputs = len(features_list) * args.window
        log_info(f"ONNX model input shape: [1, {num_inputs}] ({len(features_list)} features × {args.window} window)")
        
        onx = convert_sklearn(
            model.best_estimator_,
            initial_types=[('float_input', FloatTensorType([1, num_inputs]))],
            target_opset=12,
            options={type(model.best_estimator_): {'zipmap': False}}
        )
        
        # Handle the return value from convert_sklearn (it returns a tuple)
        if isinstance(onx, tuple):
            onx_model = onx[0]  # First element is the model proto
        else:
            onx_model = onx
            
        out_name = output_dir / f"{csv_path.stem}_SGRADT70.onnx"
        log_info(f"Writing ONNX model to {out_name}")
        with open(out_name, "wb") as f: 
            f.write(onx_model.SerializeToString())
        
        file_elapsed = time.time() - file_start_time
        log_info(f"* Model saved: {out_name.name} | Accuracy: {model.best_score_:.4f} | Processing time: {file_elapsed:.2f}s")
        processed_count += 1
    
    total_elapsed = time.time() - start_time
    log_info(f"Training session completed. Processed: {processed_count}, Skipped: {skipped_count}, Total time: {total_elapsed:.2f}s")

if __name__ == "__main__":
    main()
