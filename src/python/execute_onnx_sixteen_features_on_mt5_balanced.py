import time
import argparse
import numpy as np
import pandas as pd
import MetaTrader5 as mt5
import onnxruntime as ort
import ta
import os
import sys

# =========================
# COLORS
# =========================
from utils.colors import Fore, Style, c

# =========================
# ARGUMENTS
# =========================
parser = argparse.ArgumentParser()

parser.add_argument("--model", required=True)
parser.add_argument("--symbol", default="EURUSD")
parser.add_argument("--timeframe", default="M1")

parser.add_argument("--confidence", type=float, default=0.55)
parser.add_argument("--window", type=int, default=20)

parser.add_argument("--start_hour", type=int, default=9)
parser.add_argument("--end_hour", type=int, default=23)
parser.add_argument("--interval", type=int, default=60)

parser.add_argument("--force_consistency", action="store_true", default=True)
parser.add_argument("--consistency_bars", type=int, default=3)

parser.add_argument("--lot", type=float, default=1.0)
parser.add_argument("--magic", type=int, default=int(time.time()))

parser.add_argument("--atr_period", type=int, default=8)
parser.add_argument("--sl_mult", type=float, default=1.0)
parser.add_argument("--tp_mult", type=float, default=2.0)

parser.add_argument("--stoch_period", type=int, default=5)
parser.add_argument("--vol_window", type=int, default=10)

parser.add_argument("--adx_period", type=int, default=14)
parser.add_argument("--adx_min", type=float, default=20.0)

parser.add_argument("--h1_trend", action="store_true", default=False)
parser.add_argument("--log_file", default="trading_log.csv")
parser.add_argument("--cooldown", type=int, default=60)
parser.add_argument("--trailing", action="store_true", default=False)

args = parser.parse_args()

# =========================
# DISPLAY PARAMETERS
# =========================
def print_parameters():
    """Display all parameters with indication of defaults vs CLI overrides"""
    print(c("\n" + "="*60, Fore.CYAN))
    print(c("     3-CLASS TRADING PARAMETERS (HOLD/BUY/SELL)", Fore.CYAN))
    print(c("="*60, Fore.CYAN))
    
    # Track which arguments were explicitly provided via CLI
    # We need to check sys.argv for this
    cli_args = set()
    for arg in sys.argv[1:]:
        if arg.startswith('--'):
            cli_args.add(arg.split('=')[0].replace('--', ''))
    
    # Define all parameters with their descriptions
    params = [
        ("MODEL", args.model, "model", "ONNX model file path"),
        ("SYMBOL", args.symbol, "symbol", "Trading symbol"),
        ("TIMEFRAME", args.timeframe, "timeframe", "Chart timeframe"),
        ("", "", "", ""),  # Separator
        ("CONFIDENCE THRESHOLD", f"{args.confidence:.2f}", "confidence", "Min probability to trigger signal"),
        ("WINDOW SIZE", args.window, "window", "Feature calculation window"),
        ("", "", "", ""),  # Separator
        ("TRADING HOURS", f"{args.start_hour:02d}:00 - {args.end_hour:02d}:00", "start_hour/end_hour", "Active trading window"),
        ("CHECK INTERVAL", f"{args.interval}s", "interval", "Seconds between checks"),
        ("", "", "", ""),  # Separator
        ("FORCE CONSISTENCY", str(args.force_consistency), "force_consistency", "Require consecutive signals"),
        ("CONSISTENCY BARS", args.consistency_bars, "consistency_bars", "Consecutive bars for confirmation"),
        ("", "", "", ""),  # Separator
        ("LOT SIZE", args.lot, "lot", "Position volume"),
        ("MAGIC NUMBER", args.magic, "magic", "Order identifier"),
        ("", "", "", ""),  # Separator
        ("ATR PERIOD", args.atr_period, "atr_period", "ATR calculation lookback"),
        ("SL MULTIPLIER", args.sl_mult, "sl_mult", "Stop-loss = ATR * multiplier"),
        ("TP MULTIPLIER", args.tp_mult, "tp_mult", "Take-profit = ATR * multiplier"),
        ("", "", "", ""),  # Separator
        ("STOCH PERIOD", args.stoch_period, "stoch_period", "Stochastic oscillator period"),
        ("VOL WINDOW", args.vol_window, "vol_window", "Volume analysis window"),
        ("ADX PERIOD", args.adx_period, "adx_period", "ADX period"),
        ("ADX MIN", args.adx_min, "adx_min", "ADX minimum threshold"),
        ("", "", "", ""),  # Separator
        ("H1 TREND FILTER", str(args.h1_trend), "h1_trend", "Filter trades by H1 trend"),
        ("LOG FILE", args.log_file, "log_file", "Trade log filename"),
        ("COOLDOWN", f"{args.cooldown}s", "cooldown", "Seconds between trades"),
        ("", "", "", ""),  # Separator
        ("TRAILING STOP", str(args.trailing), "trailing", "Use trailing stop (no TP, close on opposite M1 candle)"),
    ]
    
    for label, value, arg_name, description in params:
        if label == "":  # Separator
            print(c("-" * 60, Fore.BLUE))
            continue
            
        # Determine if this was set via CLI or is default
        # Handle special cases for composite fields
        is_cli = False
        if arg_name in cli_args:
            is_cli = True
        elif '/' in arg_name:
            parts = arg_name.split('/')
            if any(p in cli_args for p in parts):
                is_cli = True
        
        source = c("[CLI]", Fore.GREEN) if is_cli else c("[DEFAULT]", Fore.YELLOW)
        
        # Format the line
        label_col = f"{label:<25}"
        value_col = f"{str(value):<15}"
        desc_col = f"{description:<25}"
        
        print(f"{source} {label_col} {value_col} {desc_col}")
    
    # Additional derived parameters
    print(c("-" * 60, Fore.BLUE))
    print(c(f"[DERIVED] {'TIMEFRAME CODE':<25} {str(TF_MAP.get(args.timeframe, 'N/A')):<15} {'MT5 internal code':<25}", Fore.MAGENTA))
    
    print(c("="*60, Fore.CYAN))
    
    # Summary box
    cli_count = len(cli_args)
    default_count = len([p for p in params if p[0] != "" and p[0] not in ["MODEL"]]) - cli_count
    
    print(f"\n{c('SUMMARY:', Fore.WHITE)} {cli_count} parameters from CLI | Defaults active for remaining")
    
    if cli_count == 0:
        print(c("⚠️  WARNING: Using ALL default values. No command-line arguments detected.", Fore.RED))
    else:
        print(f"{c('✓', Fore.GREEN)} Explicitly set: {', '.join(cli_args)}")
    
    print(c("="*60 + "\n", Fore.CYAN))

# =========================
# TIMEFRAME
# =========================
TF_MAP = {
    "M1": mt5.TIMEFRAME_M1,
    "M5": mt5.TIMEFRAME_M5,
    "M10": mt5.TIMEFRAME_M10,
    "M15": mt5.TIMEFRAME_M15,
    "M30": mt5.TIMEFRAME_M30,
    "H1": mt5.TIMEFRAME_H1,
    "H2": mt5.TIMEFRAME_H2,
    "H3": mt5.TIMEFRAME_H3,
    "H4": mt5.TIMEFRAME_H4,
}
TIMEFRAME = TF_MAP[args.timeframe]

# Display parameters BEFORE MT5 init
print_parameters()

# =========================
# MT5 INIT
# =========================
print(c("[INIT] Initializing MetaTrader 5...", Fore.CYAN))
if not mt5.initialize():
    raise RuntimeError("MT5 init failed")

mt5.symbol_select(args.symbol, True)

account = mt5.account_info()
start_balance = account.balance if account else 0

print(c("\n=== ONNX TRADER STARTED (3-CLASS MODE) ===", Fore.CYAN))
print(f"Symbol: {args.symbol} | TF: {args.timeframe}")
print(f"Initial balance: {c(f'{start_balance:.2f}', Fore.YELLOW)}\n")

# =========================
# ONNX
# =========================
print(c(f"[LOAD] Loading 3-class ONNX model: {args.model}", Fore.CYAN))
if not os.path.exists(args.model):
    raise FileNotFoundError(f"Model file not found: {args.model}")

session = ort.InferenceSession(args.model)
input_name = session.get_inputs()[0].name
print(c(f"[OK] Model loaded. Input name: {input_name}", Fore.GREEN))

# =========================
# LOG INIT
# =========================
if not os.path.exists(args.log_file):
    print(c(f"[INIT] Creating log file: {args.log_file}", Fore.CYAN))
    with open(args.log_file, "w") as f:
        f.write("timestamp,symbol,timeframe,candle_time,hold_prob,buy_prob,sell_prob,predicted_class,raw_signal,buffer,signal,action,price,sl,tp,atr,balance,equity\n")
else:
    print(c(f"[INFO] Appending to existing log: {args.log_file}", Fore.CYAN))

SIGNAL_LABEL = {1: "BUY", -1: "SELL", 0: "HOLD", None: "NONE"}

def log_event(action, probs, predicted_class, raw_signal, history, signal, price, sl, tp, atr, balance, equity, candle_time):
    """Log event with three-class probabilities"""
    with open(args.log_file, "a") as f:
        f.write(
            f"{time.strftime('%Y-%m-%d %H:%M:%S')},"
            f"{args.symbol},{args.timeframe},{candle_time},"
            f"{probs[0]:.5f},{probs[1]:.5f},{probs[2]:.5f},"
            f"{predicted_class},{SIGNAL_LABEL[raw_signal]},{history},{SIGNAL_LABEL[signal]},{action},"
            f"{price},{sl},{tp},{atr},{balance},{equity}\n"
        )

# =========================
# FEATURES
# =========================
def safe(s):
    return s.replace([np.inf, -np.inf], np.nan).fillna(0)

def build_features(df):
    atr = ta.volatility.AverageTrueRange(df['high'], df['low'], df['close'], window=args.atr_period).average_true_range()
    atr_safe = atr.replace(0, np.nan).ffill().fillna(1)

    df['feat_body'] = safe((df['close'] - df['open']) / atr_safe)
    df['feat_range'] = safe((df['high'] - df['low']) / atr_safe)

    stoch = ta.momentum.StochasticOscillator(df['high'], df['low'], df['close'], window=args.stoch_period)
    k, d = stoch.stoch(), stoch.stoch_signal()

    df['feat_stoch_momentum'] = safe((k - d) / 100.0)
    df['feat_stoch_position'] = safe((k - 50.0) / 50.0)
    df['feat_stoch_velocity'] = safe(k.diff() / 100.0)

    overbought = np.where(k > 80, -(k - 80)/20.0, 0)
    oversold   = np.where(k < 20, (20 - k)/20.0, 0)
    df['feat_stoch_divergence'] = safe(pd.Series(overbought + oversold))

    vol = df['tick_volume']
    vol_ma = vol.rolling(args.vol_window).mean()
    vol_std = vol.rolling(args.vol_window).std()

    df['feat_vol_ratio'] = safe(vol / vol_ma.replace(0,1))
    df['feat_vol_momentum'] = safe((vol.ewm(span=5).mean() - vol.ewm(span=20).mean()) / vol_ma.replace(0,1))

    price_change = safe(df['close'].pct_change().abs())
    vol_change = safe(vol.pct_change().abs())

    df['feat_vol_price_div'] = safe(vol_change - price_change)
    df['feat_vol_percentile'] = safe(vol.rolling(args.vol_window).apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1]))
    df['feat_vol_zscore'] = safe((vol - vol_ma) / vol_std.replace(0,1))

    # ADX features (5 features)
    adx_indicator = ta.trend.ADXIndicator(df['high'], df['low'], df['close'], window=args.adx_period)
    adx = adx_indicator.adx()
    di_plus = adx_indicator.adx_pos()
    di_minus = adx_indicator.adx_neg()
    di_sum = di_plus + di_minus
    di_sum_safe = di_sum.replace(0, np.nan).fillna(1.0)

    # Feature 1: ADX strength - normalized around adx_min
    df['feat_adx_strength'] = safe((adx - args.adx_min) / (100.0 - args.adx_min))
    
    # Feature 2: DI signal
    df['feat_di_signal'] = safe(pd.Series(np.where(di_plus > di_minus, 1.0,
                                                   np.where(di_minus > di_plus, -1.0, 0.0))))
    
    # Feature 3: DI separation - directional conviction
    df['feat_di_separation'] = safe((di_plus - di_minus) / di_sum_safe)
    
    # Feature 4: ADX momentum - rate of change
    df['feat_adx_momentum'] = safe(adx.diff() / 100.0)
    
    # Feature 5: ADX regime - categorical (0=no trend, 0.5=developing, 1=strong)
    regime = np.where(adx < args.adx_min, 0.0,
             np.where(adx < 40.0, 0.5, 1.0))
    df['feat_adx_regime'] = safe(pd.Series(regime))

    return df

FEATURES = [
    'feat_body','feat_range',
    'feat_stoch_momentum','feat_stoch_position',
    'feat_stoch_velocity','feat_stoch_divergence',
    'feat_vol_ratio','feat_vol_momentum',
    'feat_vol_price_div','feat_vol_percentile',
    'feat_vol_zscore',
    'feat_adx_strength','feat_di_signal',
    'feat_di_separation','feat_adx_momentum',
    'feat_adx_regime'
]

# =========================
# H1 TREND FILTER
# =========================
def h1_trend_allows(direction):
    bars = mt5.copy_rates_from_pos(args.symbol, mt5.TIMEFRAME_H1, 1, 1)
    if bars is None or len(bars) == 0:
        return True
    candle = bars[0]
    bullish = candle['close'] > candle['open']
    if direction == 1:
        return bullish
    else:
        return not bullish

# ============================================================================
# ONNX THREE-CLASS OUTPUT PARSER
# ============================================================================
def extract_three_class_probabilities(outputs):
    """
    Extract probabilities for 3-class model: [HOLD, BUY, SELL]
    
    Returns:
        probs: numpy array [p_hold, p_buy, p_sell]
        predicted_class: int (0=HOLD, 1=BUY, 2=SELL)
    """
    # Check if we have probability output (usually second output)
    if len(outputs) > 1:
        probs = np.array(outputs[1])
        
        # Shape can be (1, 3) or (3,)
        if probs.ndim == 2:
            probs = probs[0]  # Get first row if 2D
        
        # Ensure we have 3 probabilities
        if len(probs) == 3:
            predicted_class = int(np.argmax(probs))
            return probs, predicted_class
    
    # Fallback: use predicted class directly
    pred = np.array(outputs[0])
    
    if pred.ndim == 0:
        predicted_class = int(pred)
    elif pred.ndim == 1:
        predicted_class = int(pred[0])
    elif pred.ndim == 2:
        predicted_class = int(pred[0][0])
    else:
        raise ValueError(f"Unexpected ONNX output shape: {pred.shape}")
    
    # Create one-hot probabilities if we only have class prediction
    probs = np.zeros(3)
    probs[predicted_class] = 1.0
    
    return probs, predicted_class

# =========================
# POSITION HANDLING
# =========================
def get_filling_mode():
    """Detect the correct filling mode for the symbol."""
    info = mt5.symbol_info(args.symbol)
    if info is None:
        return mt5.ORDER_FILLING_IOC
    
    # Filling mode bitmask: FOK=1, IOC=2, RETURN=4
    filling_mode = info.filling_mode
    if filling_mode & 1:  # SYMBOL_TRADE_FILLING_FOK
        return mt5.ORDER_FILLING_FOK
    elif filling_mode & 2:  # SYMBOL_TRADE_FILLING_IOC
        return mt5.ORDER_FILLING_IOC
    else:
        return mt5.ORDER_FILLING_IOC

FILLING_MODE = get_filling_mode()
print(c(f"[INFO] Order filling mode: {FILLING_MODE}", Fore.CYAN))

def get_positions():
    positions = mt5.positions_get(symbol=args.symbol)
    if positions is None:
        return []
    return [p for p in positions if p.magic == args.magic]

def get_buy_positions():
    return [p for p in get_positions() if p.type == mt5.ORDER_TYPE_BUY]

def get_sell_positions():
    return [p for p in get_positions() if p.type == mt5.ORDER_TYPE_SELL]

def send_buy(price, sl, tp, buy_prob):
    # Some brokers reject tp=0, use distant TP for trailing mode
    if tp == 0:
        tp = price + 1000.0  # Very distant TP for trailing
    
    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": args.symbol,
        "volume": args.lot,
        "type": mt5.ORDER_TYPE_BUY,
        "price": price,
        "sl": sl,
        "tp": tp,
        "magic": args.magic,
        "deviation": 10,
        "type_filling": FILLING_MODE,
        "comment": f"3Class BUY@{buy_prob:.3f}"
    }
    result = mt5.order_send(request)
    if result is None:
        error_code, error_msg = mt5.last_error()
        print(c(f"[MT5 ERROR] {error_code}: {error_msg}", Fore.RED))
    return result

def send_sell(price, sl, tp, sell_prob):
    # Some brokers reject tp=0, use distant TP for trailing mode
    if tp == 0:
        tp = price - 1000.0  # Very distant TP for trailing
    
    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": args.symbol,
        "volume": args.lot,
        "type": mt5.ORDER_TYPE_SELL,
        "price": price,
        "sl": sl,
        "tp": tp,
        "magic": args.magic,
        "deviation": 10,
        "type_filling": FILLING_MODE,
        "comment": f"3Class SELL@{sell_prob:.3f}"
    }
    result = mt5.order_send(request)
    if result is None:
        error_code, error_msg = mt5.last_error()
        print(c(f"[MT5 ERROR] {error_code}: {error_msg}", Fore.RED))
    return result

def close_position(pos):
    tick = mt5.symbol_info_tick(args.symbol)
    if pos.type == mt5.ORDER_TYPE_BUY:
        close_type = mt5.ORDER_TYPE_SELL
        price = tick.bid
    else:
        close_type = mt5.ORDER_TYPE_BUY
        price = tick.ask

    return mt5.order_send({
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": args.symbol,
        "volume": pos.volume,
        "type": close_type,
        "position": pos.ticket,
        "price": price,
        "magic": args.magic,
        "deviation": 10,
        "type_filling": FILLING_MODE,
    })

def get_last_completed_m1_candle():
    """Get the last completed M1 candle (index 1, not the forming one at index 0)."""
    m1_bars = mt5.copy_rates_from_pos(args.symbol, mt5.TIMEFRAME_M1, 0, 2)
    if m1_bars is None or len(m1_bars) < 2:
        return None
    return m1_bars[1]  # Index 1 is the last completed candle

def is_m1_candle_opposite(candle, direction):
    """Check if a completed M1 candle is opposite to the position direction."""
    if candle is None:
        return False
    bullish = candle['close'] > candle['open']
    if direction == 1:  # BUY position - check for bearish candle
        return not bullish
    else:  # SELL position - check for bullish candle
        return bullish

# =========================
# LOOP
# =========================
history = []
last_trade_time = 0
last_outside_hours_msg = 0
trailing_positions = {}  # ticket -> {'tp_price': float, 'tp_reached': bool, 'last_m1_time': int}

print(c("[READY] Starting 3-class trading loop...\n", Fore.GREEN))
print(c("Model outputs: 0=HOLD | 1=BUY | 2=SELL\n", Fore.YELLOW))

try:
    while True:
        # Get MT5 server time from latest candle
        df = mt5.copy_rates_from_pos(args.symbol, TIMEFRAME, 0, args.window + 50)
        if df is None or len(df) == 0:
            time.sleep(1)
            continue

        server_time = pd.to_datetime(df[-1]['time'], unit='s')
        current_hour = server_time.hour

        if not (args.start_hour <= current_hour <= args.end_hour):
            if time.time() - last_outside_hours_msg >= 60:
                local_time = time.strftime('%H:%M:%S')
                print(c(f"[WAIT] {args.start_hour:02d}:00-{args.end_hour:02d}:00 | MT5: {current_hour:02d}:00 | Local: {local_time}", Fore.YELLOW))
                last_outside_hours_msg = time.time()
            time.sleep(1)
            continue

        df = pd.DataFrame(df)

        df = build_features(df).dropna()
        if len(df) < args.window:
            time.sleep(1)
            continue

        candle_time = df.iloc[-1]['time']

        X = df[FEATURES].iloc[-args.window:].values.flatten().astype(np.float32).reshape(1, -1)

        # ========================================
        # THREE-CLASS INFERENCE
        # ========================================
        outputs = session.run(None, {input_name: X})
        probs, predicted_class = extract_three_class_probabilities(outputs)
        
        # probs = [p_hold, p_buy, p_sell]
        # predicted_class = 0 (HOLD), 1 (BUY), or 2 (SELL)
        
        hold_prob = probs[0]
        buy_prob  = probs[1]
        sell_prob = probs[2]
        
        # Determine raw signal based on predicted class AND confidence threshold
        if predicted_class == 1 and buy_prob >= args.confidence:
            raw_signal = 1  # BUY
        elif predicted_class == 2 and sell_prob >= args.confidence:
            raw_signal = -1  # SELL
        else:
            raw_signal = 0  # HOLD

        history.append(raw_signal)
        if len(history) > args.consistency_bars:
            history.pop(0)

        consistent_buy  = len(history) == args.consistency_bars and all(s == 1  for s in history)
        consistent_sell = len(history) == args.consistency_bars and all(s == -1 for s in history)
        consistent_flat = len(history) == args.consistency_bars and all(s == 0  for s in history)

        if args.force_consistency:
            if consistent_buy:
                signal = 1
            elif consistent_sell:
                signal = -1
            elif consistent_flat:
                signal = 0
            else:
                signal = None
        else:
            signal = raw_signal

        buy_positions  = get_buy_positions()
        sell_positions = get_sell_positions()
        pos_count = len(buy_positions) + len(sell_positions)

        # Check trailing positions
        if args.trailing and pos_count > 0:
            for pos in buy_positions + sell_positions:
                if pos.ticket in trailing_positions:
                    tp_data = trailing_positions[pos.ticket]
                    if not tp_data['tp_reached']:
                        # Check if TP level reached
                        tick = mt5.symbol_info_tick(args.symbol)
                        if pos.type == mt5.ORDER_TYPE_BUY and tick.bid >= tp_data['tp_price']:
                            tp_data['tp_reached'] = True
                            print(c(f"[TRAILING] BUY {pos.ticket} TP level reached, monitoring for exit", Fore.MAGENTA))
                        elif pos.type == mt5.ORDER_TYPE_SELL and tick.ask <= tp_data['tp_price']:
                            tp_data['tp_reached'] = True
                            print(c(f"[TRAILING] SELL {pos.ticket} TP level reached, monitoring for exit", Fore.MAGENTA))
                    else:
                        # TP reached, check for NEW opposite M1 candle only
                        m1_candle = get_last_completed_m1_candle()
                        if m1_candle is not None and m1_candle['time'] != tp_data.get('last_m1_time', 0):
                            # New M1 candle completed, update tracking
                            tp_data['last_m1_time'] = m1_candle['time']
                            direction = 1 if pos.type == mt5.ORDER_TYPE_BUY else -1
                            if is_m1_candle_opposite(m1_candle, direction):
                                result = close_position(pos)
                                if result and result.retcode == mt5.TRADE_RETCODE_DONE:
                                    m1_time_str = pd.to_datetime(m1_candle['time'], unit='s').strftime('%H:%M:%S')
                                    reason = "Opposite M1 candle after TP reached"
                                    print(c(f"[TRAILING EXIT] {pos.ticket} closed. Cause: {reason} (Candle time: {m1_time_str})", Fore.MAGENTA))
                                    del trailing_positions[pos.ticket]
                                    tick = mt5.symbol_info_tick(args.symbol)
                                    log_event("CLOSE", probs, predicted_class, raw_signal, history.copy(), signal, tick.bid if direction == 1 else tick.ask, 0, 0, atr, balance, equity, candle_time)

        tick = mt5.symbol_info_tick(args.symbol)

        account = mt5.account_info()
        balance = account.balance if account else 0
        equity  = account.equity  if account else 0

        atr = ta.volatility.AverageTrueRange(df['high'], df['low'], df['close'], window=args.atr_period).average_true_range().iloc[-1]

        print(c("--------------------------------------------------", Fore.BLUE))
        print(f"Hour: {server_time.strftime('%H:%M:%S')} | Class: {c(predicted_class, Fore.MAGENTA)} | Expected: {c(args.confidence, Fore.MAGENTA)}")
        print(f"HOLD:{c(f'{hold_prob:.3f}', Fore.YELLOW)} | BUY:{c(f'{buy_prob:.3f}', Fore.GREEN)} | SELL:{c(f'{sell_prob:.3f}', Fore.RED)}")
        buffer_display = [SIGNAL_LABEL[s] for s in history]
        print(f"Buffer: {c(buffer_display, Fore.CYAN)}")
        print(f"Signal: {c(SIGNAL_LABEL[signal], Fore.MAGENTA)} | Positions: {c(pos_count, Fore.RED)}")

        # ================= ENTRY BUY =================
        if signal == 1 and len(buy_positions) == 0 and (time.time() - last_trade_time > args.cooldown):
            if args.h1_trend and not h1_trend_allows(1):
                print(c("[BUY BLOCKED by H1 trend]", Fore.YELLOW))
                log_event("HOLD", probs, predicted_class, raw_signal, history.copy(), signal, 0, 0, 0, atr, balance, equity, candle_time)
            else:
                price = tick.ask
                sl = price - atr * args.sl_mult
                tp = price + atr * args.tp_mult

                # Determine TP and trailing mode
                if args.trailing:
                    entry_tp = 0  # No TP set
                    tp_for_log = price + atr * args.tp_mult  # For logging only
                else:
                    entry_tp = tp
                    tp_for_log = tp

                result = send_buy(price, sl, entry_tp, buy_prob)

                if result and result.retcode == mt5.TRADE_RETCODE_DONE:
                    print(c("[BUY ENTRY OK]" + (" [TRAILING]" if args.trailing else ""), Fore.GREEN))
                    last_trade_time = time.time()
                    if args.trailing:
                        trailing_positions[result.order] = {'tp_price': tp_for_log, 'tp_reached': False, 'last_m1_time': 0}
                    log_event("BUY", probs, predicted_class, raw_signal, history.copy(), signal, price, sl, tp_for_log, atr, balance, equity, candle_time)
                else:
                    error_code = result.retcode if result else "No result"
                    error_msg = result.comment if result else "Unknown error"
                    print(c(f"[BUY ENTRY ERROR] Code: {error_code} | {error_msg}", Fore.RED))
                    print(c(f"  Details: price={price:.5f} sl={sl:.5f} tp={entry_tp:.5f} lot={args.lot}", Fore.RED))

        # ================= ENTRY SELL =================
        elif signal == -1 and len(sell_positions) == 0 and (time.time() - last_trade_time > args.cooldown):
            if args.h1_trend and not h1_trend_allows(-1):
                print(c("[SELL BLOCKED by H1 trend]", Fore.YELLOW))
                log_event("HOLD", probs, predicted_class, raw_signal, history.copy(), signal, 0, 0, 0, atr, balance, equity, candle_time)
            else:
                price = tick.bid
                sl = price + atr * args.sl_mult
                tp = price - atr * args.tp_mult

                # Determine TP and trailing mode
                if args.trailing:
                    entry_tp = 0  # No TP set
                    tp_for_log = price - atr * args.tp_mult  # For logging only
                else:
                    entry_tp = tp
                    tp_for_log = tp

                result = send_sell(price, sl, entry_tp, sell_prob)

                if result and result.retcode == mt5.TRADE_RETCODE_DONE:
                    print(c("[SELL ENTRY OK]" + (" [TRAILING]" if args.trailing else ""), Fore.CYAN))
                    last_trade_time = time.time()
                    if args.trailing:
                        trailing_positions[result.order] = {'tp_price': tp_for_log, 'tp_reached': False, 'last_m1_time': 0}
                    log_event("SELL", probs, predicted_class, raw_signal, history.copy(), signal, price, sl, tp_for_log, atr, balance, equity, candle_time)
                else:
                    error_code = result.retcode if result else "No result"
                    error_msg = result.comment if result else "Unknown error"
                    print(c(f"[SELL ENTRY ERROR] Code: {error_code} | {error_msg}", Fore.RED))
                    print(c(f"  Details: price={price:.5f} sl={sl:.5f} tp={entry_tp:.5f} lot={args.lot}", Fore.RED))

        else:
            log_event("HOLD", probs, predicted_class, raw_signal, history.copy(), signal, 0, 0, 0, atr, balance, equity, candle_time)

        time.sleep(args.interval)

except KeyboardInterrupt:
    print(c("\n[SHUTDOWN] Trading loop stopped by user (Ctrl+C)", Fore.MAGENTA))
    print(c("[INFO] Exit completed", Fore.CYAN))
    sys.exit(0)
