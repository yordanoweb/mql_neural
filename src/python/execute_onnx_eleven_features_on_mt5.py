import time
import argparse
import numpy as np
import pandas as pd
import MetaTrader5 as mt5
import onnxruntime as ort
import ta

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

parser.add_argument("--debug_onnx", action="store_true")

args = parser.parse_args()

# =========================
# TIMEFRAME
# =========================
TF_MAP = {
    "M1": mt5.TIMEFRAME_M1,
    "M5": mt5.TIMEFRAME_M5,
    "M15": mt5.TIMEFRAME_M15,
    "H1": mt5.TIMEFRAME_H1,
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
# FEATURES
# =========================
def safe(s):
    return s.replace([np.inf, -np.inf], np.nan).fillna(0)

def build_features(df):
    atr = ta.volatility.AverageTrueRange(
        df['high'], df['low'], df['close'], window=args.atr_period
    ).average_true_range()

    atr_safe = atr.replace(0, np.nan).ffill().fillna(1)

    df['feat_body'] = safe((df['close'] - df['open']) / atr_safe)
    df['feat_range'] = safe((df['high'] - df['low']) / atr_safe)

    stoch = ta.momentum.StochasticOscillator(
        df['high'], df['low'], df['close'],
        window=args.stoch_period
    )

    k = stoch.stoch()
    d = stoch.stoch_signal()

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
    df['feat_vol_momentum'] = safe(
        (vol.ewm(span=5).mean() - vol.ewm(span=20).mean()) / vol_ma.replace(0,1)
    )

    price_change = safe(df['close'].pct_change().abs())
    vol_change = safe(vol.pct_change().abs())

    df['feat_vol_price_div'] = safe(vol_change - price_change)

    df['feat_vol_percentile'] = safe(
        vol.rolling(args.vol_window).apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1])
    )

    df['feat_vol_zscore'] = safe(
        (vol - vol_ma) / vol_std.replace(0,1)
    )

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
# HELPERS
# =========================
def get_data(n):
    rates = mt5.copy_rates_from_pos(args.symbol, TIMEFRAME, 0, n)
    return pd.DataFrame(rates)

def get_position():
    pos = mt5.positions_get(symbol=args.symbol)
    if pos:
        for p in pos:
            if p.magic == args.magic:
                return p
    return None

def send_order(direction, price, sl, tp):
    print(c(f"[ENTRY] BUY @ {price:.5f}", Fore.GREEN))

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
        "type_filling": mt5.ORDER_FILLING_IOC
    }

    result = mt5.order_send(request)
    print(c("[OK]" if result and result.retcode == mt5.TRADE_RETCODE_DONE else "[ERROR]", 
            Fore.GREEN if result and result.retcode == mt5.TRADE_RETCODE_DONE else Fore.RED))

def close_position(pos):
    print(c(f"[EXIT] Closing position {pos.ticket}", Fore.MAGENTA))

    price = mt5.symbol_info_tick(args.symbol).bid

    result = mt5.order_send({
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": args.symbol,
        "volume": pos.volume,
        "type": mt5.ORDER_TYPE_SELL,
        "position": pos.ticket,
        "price": price,
        "magic": args.magic,
        "deviation": 10,
    })

    print(c("[OK]" if result and result.retcode == mt5.TRADE_RETCODE_DONE else "[ERROR]",
            Fore.GREEN if result and result.retcode == mt5.TRADE_RETCODE_DONE else Fore.RED))

# =========================
# LOOP
# =========================
history = []

while True:
    now = time.localtime()

    if not (args.start_hour <= now.tm_hour <= args.end_hour):
        time.sleep(1)
        continue

    df = get_data(args.window + 50)
    df = build_features(df).dropna()

    if len(df) < args.window:
        time.sleep(1)
        continue

    X = df[FEATURES].iloc[-args.window:].values.flatten().astype(np.float32)
    X = X.reshape(1, -1)

    outputs = session.run(None, {input_name: X})

    if args.debug_onnx:
        print("RAW ONNX:", outputs)

    prob = extract_probability(outputs)

    raw_signal = 1 if prob >= args.confidence else 0

    history.append(raw_signal)
    if len(history) > args.consistency_bars:
        history.pop(0)

    consistent_buy = len(history) == args.consistency_bars and all(s == 1 for s in history)
    consistent_flat = len(history) == args.consistency_bars and all(s == 0 for s in history)

    if args.force_consistency:
        if consistent_buy:
            signal = 1
        elif consistent_flat:
            signal = 0
        else:
            signal = -1
    else:
        signal = raw_signal

    pos = get_position()
    tick = mt5.symbol_info_tick(args.symbol)

    account = mt5.account_info()
    balance = account.balance if account else 0
    equity  = account.equity if account else 0

    print(c("--------------------------------------------------", Fore.BLUE))
    print(f"Time: {time.strftime('%H:%M:%S')}")
    print(f"Prob: {c(f'{prob:.3f}', Fore.CYAN)}")
    print(f"Raw: {raw_signal} | Buffer: {history}")
    print(f"Consistency BUY: {consistent_buy} | FLAT: {consistent_flat}")

    if pos:
        print(c(f"Open position | Profit: {pos.profit:.2f}", Fore.MAGENTA))
    else:
        print("No position")

    print(f"Balance: {c(f'{balance:.2f}', Fore.YELLOW)} | Equity: {c(f'{equity:.2f}', Fore.CYAN)}")

    atr = ta.volatility.AverageTrueRange(
        df['high'], df['low'], df['close'], window=args.atr_period
    ).average_true_range().iloc[-1]

    if signal == 1 and pos is None:
        price = tick.ask
        sl = price - atr * args.sl_mult
        tp = price + atr * args.tp_mult
        send_order(1, price, sl, tp)

    elif signal == 0 and pos is not None:
        close_position(pos)

    time.sleep(args.interval)