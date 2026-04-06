import time
import argparse
import numpy as np
import pandas as pd
import MetaTrader5 as mt5
import onnxruntime as ort
import ta
import os

# =========================
# COLORS
# =========================
try:
    from colorama import Fore, Style, init
    init(autoreset=True)
except:
    class Dummy:
        RESET_ALL=""
    Fore = Style = Dummy()

def c(text, color):
    return f"{color}{text}{Style.RESET_ALL}"

# =========================
# ARGUMENTS
# =========================
parser = argparse.ArgumentParser()

parser.add_argument("--model", required=True)
parser.add_argument("--symbol", default="EURUSD")
parser.add_argument("--timeframe", default="M1")

parser.add_argument("--confidence_buy", type=float, default=0.55)
parser.add_argument("--confidence_sell", type=float, default=0.55)
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

parser.add_argument("--log_file", default="trading_log.csv")
parser.add_argument("--cooldown", type=int, default=5)

args = parser.parse_args()

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

# =========================
# MT5 INIT
# =========================
if not mt5.initialize():
    raise RuntimeError("MT5 init failed")

mt5.symbol_select(args.symbol, True)

account = mt5.account_info()
start_balance = account.balance if account else 0

print(c("\n=== ONNX TRADER STARTED ===", Fore.CYAN))
print(f"Symbol: {args.symbol} | TF: {args.timeframe}")
print(f"Initial balance: {c(f'{start_balance:.2f}', Fore.YELLOW)}\n")

# =========================
# ONNX
# =========================
session = ort.InferenceSession(args.model)
input_name = session.get_inputs()[0].name

# =========================
# LOG INIT
# =========================
if not os.path.exists(args.log_file):
    with open(args.log_file, "w") as f:
        f.write("timestamp,symbol,timeframe,candle_time,prob,raw_signal,buffer,signal,action,price,sl,tp,atr,balance,equity\n")

def log_event(action, prob, raw_signal, history, signal, price, sl, tp, atr, balance, equity, candle_time):
    with open(args.log_file, "a") as f:
        f.write(
            f"{time.strftime('%Y-%m-%d %H:%M:%S')},"
            f"{args.symbol},{args.timeframe},{candle_time},"
            f"{prob:.5f},{raw_signal},{history},{signal},{action},"
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

    return df

FEATURES = [
    'feat_body','feat_range',
    'feat_stoch_momentum','feat_stoch_position',
    'feat_stoch_velocity','feat_stoch_divergence',
    'feat_vol_ratio','feat_vol_momentum',
    'feat_vol_price_div','feat_vol_percentile',
    'feat_vol_zscore'
]

# =========================
# ONNX OUTPUT PARSER
# =========================
def extract_probability(outputs):
    if len(outputs) > 1:
        probs = np.array(outputs[1])
        if probs.ndim == 2 and probs.shape[1] > 1:
            return float(probs[0][1])

    pred = np.array(outputs[0])

    if pred.ndim == 0:
        return float(pred)
    elif pred.ndim == 1:
        return float(pred[0])
    elif pred.ndim == 2:
        return float(pred[0][1] if pred.shape[1] > 1 else pred[0][0])

    raise ValueError(f"Unexpected ONNX output: {pred}")

# =========================
# POSITION HANDLING
# =========================
def get_positions():
    positions = mt5.positions_get(symbol=args.symbol)
    if positions is None:
        return []
    return [p for p in positions if p.magic == args.magic]

def get_buy_positions():
    return [p for p in get_positions() if p.type == mt5.ORDER_TYPE_BUY]

def get_sell_positions():
    return [p for p in get_positions() if p.type == mt5.ORDER_TYPE_SELL]

def send_buy(price, sl, tp):
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
        "type_filling": mt5.ORDER_FILLING_IOC,
        "comment": f"ElevenFeat BUY@{price}"
    }
    return mt5.order_send(request)

def send_sell(price, sl, tp):
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
        "type_filling": mt5.ORDER_FILLING_IOC,
        "comment": f"ElevenFeat SELL@{price}"
    }
    return mt5.order_send(request)

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
    })

# =========================
# LOOP
# =========================
history = []
last_trade_time = 0

while True:
    now = time.localtime()

    if not (args.start_hour <= now.tm_hour <= args.end_hour):
        time.sleep(1)
        continue

    df = mt5.copy_rates_from_pos(args.symbol, TIMEFRAME, 0, args.window + 50)
    df = pd.DataFrame(df)

    df = build_features(df).dropna()
    if len(df) < args.window:
        time.sleep(1)
        continue

    candle_time = df.iloc[-1]['time']

    X = df[FEATURES].iloc[-args.window:].values.flatten().astype(np.float32).reshape(1, -1)

    outputs = session.run(None, {input_name: X})
    prob = extract_probability(outputs)

    # prob >= confidence_buy → 1 (buy), prob <= confidence_sell → -1 (sell), else 0 (flat)
    if prob >= args.confidence_buy:
        raw_signal = 1
    elif prob <= args.confidence_sell:
        raw_signal = -1
    else:
        raw_signal = 0

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
            signal = None  # buffer not yet consistent, hold
    else:
        signal = raw_signal

    buy_positions  = get_buy_positions()
    sell_positions = get_sell_positions()
    pos_count = len(buy_positions) + len(sell_positions)

    tick = mt5.symbol_info_tick(args.symbol)

    account = mt5.account_info()
    balance = account.balance if account else 0
    equity  = account.equity  if account else 0

    atr = ta.volatility.AverageTrueRange(df['high'], df['low'], df['close'], window=args.atr_period).average_true_range().iloc[-1]

    print(c("--------------------------------------------------", Fore.BLUE))
    print(f"Hour: {time.strftime('%H:%M:%S')} | Prob: {prob:.3f}")
    print(f"Buffer: {history} | Signal: {signal} | Positions: {pos_count}")

    # ================= ENTRY BUY =================
    if signal == 1 and len(buy_positions) == 0 and (time.time() - last_trade_time > args.cooldown):
        # Close any open sell before buying
        for p in sell_positions:
            result = close_position(p)
            if result and result.retcode == mt5.TRADE_RETCODE_DONE:
                print(c(f"[CLOSE SELL before BUY] {p.ticket}", Fore.YELLOW))

        price = tick.ask
        sl = price - atr * args.sl_mult
        tp = price + atr * args.tp_mult

        result = send_buy(price, sl, tp)

        if result and result.retcode == mt5.TRADE_RETCODE_DONE:
            print(c("[BUY ENTRY OK]", Fore.GREEN))
            last_trade_time = time.time()
            log_event("BUY", prob, raw_signal, history.copy(), signal, price, sl, tp, atr, balance, equity, candle_time)
        else:
            print(c("[BUY ENTRY ERROR]", Fore.RED))

    # ================= ENTRY SELL =================
    elif signal == -1 and len(sell_positions) == 0 and (time.time() - last_trade_time > args.cooldown):
        # Close any open buy before selling
        for p in buy_positions:
            result = close_position(p)
            if result and result.retcode == mt5.TRADE_RETCODE_DONE:
                print(c(f"[CLOSE BUY before SELL] {p.ticket}", Fore.YELLOW))

        price = tick.bid
        sl = price + atr * args.sl_mult
        tp = price - atr * args.tp_mult

        result = send_sell(price, sl, tp)

        if result and result.retcode == mt5.TRADE_RETCODE_DONE:
            print(c("[SELL ENTRY OK]", Fore.CYAN))
            last_trade_time = time.time()
            log_event("SELL", prob, raw_signal, history.copy(), signal, price, sl, tp, atr, balance, equity, candle_time)
        else:
            print(c("[SELL ENTRY ERROR]", Fore.RED))

    # ================= EXIT ALL =================
    elif signal == 0 and pos_count > 0:
        for p in buy_positions + sell_positions:
            result = close_position(p)
            if result and result.retcode == mt5.TRADE_RETCODE_DONE:
                print(c(f"[EXIT OK] {p.ticket}", Fore.MAGENTA))

        log_event("CLOSE", prob, raw_signal, history.copy(), signal, tick.bid, 0, 0, atr, balance, equity, candle_time)

    else:
        log_event("HOLD", prob, raw_signal, history.copy(), signal, 0, 0, 0, atr, balance, equity, candle_time)

    time.sleep(args.interval)
