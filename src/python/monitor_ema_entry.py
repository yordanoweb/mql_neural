#!/usr/bin/env python3
"""
MT5 Trading Script with EMA-based Entry/Exit and ATR-based Stop Loss / Take Profit
"""

import time
import argparse
import MetaTrader5 as mt5
import pandas as pd
import ta

# ---------- Color setup ----------
try:
    import colorama
    colorama.init(autoreset=True)
    COLORS_SUPPORTED = True
except ImportError:
    COLORS_SUPPORTED = False

class Colors:
    if COLORS_SUPPORTED:
        RESET   = '\033[0m'
        RED     = '\033[91m'
        GREEN   = '\033[92m'
        YELLOW  = '\033[93m'
        BLUE    = '\033[94m'
        MAGENTA = '\033[95m'
        CYAN    = '\033[96m'
        WHITE   = '\033[97m'
    else:
        RESET = RED = GREEN = YELLOW = BLUE = MAGENTA = CYAN = WHITE = ""

def colorize(text, color):
    if COLORS_SUPPORTED:
        return f"{color}{text}{Colors.RESET}"
    return text

# ---------- Timeframe mapping ----------
TIMEFRAME_MAP = {
    "M1":  mt5.TIMEFRAME_M1, "M2":  mt5.TIMEFRAME_M2, "M3":  mt5.TIMEFRAME_M3,
    "M4":  mt5.TIMEFRAME_M4, "M5":  mt5.TIMEFRAME_M5, "M6":  mt5.TIMEFRAME_M6,
    "M10": mt5.TIMEFRAME_M10, "M12": mt5.TIMEFRAME_M12, "M15": mt5.TIMEFRAME_M15,
    "M20": mt5.TIMEFRAME_M20, "M30": mt5.TIMEFRAME_M30,
    "H1":  mt5.TIMEFRAME_H1, "H2":  mt5.TIMEFRAME_H2, "H3":  mt5.TIMEFRAME_H3,
    "H4":  mt5.TIMEFRAME_H4, "H6":  mt5.TIMEFRAME_H6, "H8":  mt5.TIMEFRAME_H8,
    "H12": mt5.TIMEFRAME_H12,
    "D1":  mt5.TIMEFRAME_D1, "W1":  mt5.TIMEFRAME_W1, "MN1": mt5.TIMEFRAME_MN1,
}

def parse_timeframe(tf_str):
    tf_str = tf_str.upper()
    if tf_str not in TIMEFRAME_MAP:
        raise ValueError(f"Unsupported timeframe: {tf_str}. Use: {', '.join(TIMEFRAME_MAP.keys())}")
    return TIMEFRAME_MAP[tf_str]

# ---------- Argument parsing ----------
parser = argparse.ArgumentParser(description="MT5 Trading Script - EMA Cross with ATR SL/TP")
parser.add_argument("--symbol", type=str, default="EURUSD", help="Trading symbol (default: EURUSD)")
parser.add_argument("--timeframe", type=str, default="M1", help="Candle timeframe (e.g., M1, M5, H1, D1) (default: M1)")
parser.add_argument("--ema_period", type=int, default=9, help="EMA period (default: 9)")
parser.add_argument("--atr_period", type=int, default=14, help="ATR period for SL/TP (default: 14)")
parser.add_argument("--sl_multiplier", type=float, default=2.0, help="Stop loss multiplier of ATR (default: 2.0)")
parser.add_argument("--tp_multiplier", type=float, default=3.0, help="Take profit multiplier of ATR (default: 3.0)")
parser.add_argument("--magic", type=int, default=91234569, help="Magic number (default: 91234569)")
parser.add_argument("--volume", type=float, default=0.1, help="Order volume in lots (default: 0.1)")
parser.add_argument("--entry_points", type=float, default=10.0, help="Points away from EMA to trigger entry (default: 10)")
parser.add_argument("--interval", type=float, default=5.0, help="Seconds between processing steps (default: 5)")
args = parser.parse_args()

SYMBOL = args.symbol
TIMEFRAME = parse_timeframe(args.timeframe)
EMA_PERIOD = args.ema_period
ATR_PERIOD = args.atr_period
SL_MULT = args.sl_multiplier
TP_MULT = args.tp_multiplier
MAGIC = args.magic
VOLUME = args.volume
ENTRY_POINTS = args.entry_points
INTERVAL = args.interval

# ---------- MT5 Initialization ----------
if not mt5.initialize():
    print(colorize("Failed to initialize MT5, error code = " + str(mt5.last_error()), Colors.RED))
    quit()

symbol_info = mt5.symbol_info(SYMBOL)
if symbol_info is None:
    print(colorize(f"Symbol {SYMBOL} not found, error = {mt5.last_error()}", Colors.RED))
    mt5.shutdown()
    quit()

if not symbol_info.visible:
    if not mt5.symbol_select(SYMBOL, True):
        print(colorize(f"Failed to select {SYMBOL}", Colors.RED))
        mt5.shutdown()
        quit()

point = symbol_info.point
entry_threshold = ENTRY_POINTS * point

print(colorize("MT5 initialized", Colors.GREEN) +
      f" – trading {SYMBOL} on {args.timeframe} with EMA{EMA_PERIOD}")
print(f"Magic: {MAGIC}, Volume: {VOLUME}, Entry threshold: {ENTRY_POINTS} points ({entry_threshold:.5f})")
print(f"ATR period: {ATR_PERIOD}, SL multiplier: {SL_MULT}, TP multiplier: {TP_MULT}")
print(f"Checking every {INTERVAL} seconds\n")

# ---------- Helper functions ----------
def get_current_prices():
    tick = mt5.symbol_info_tick(SYMBOL)
    if tick is None:
        return None, None
    return tick.bid, tick.ask

def get_candles(count):
    rates = mt5.copy_rates_from_pos(SYMBOL, TIMEFRAME, 0, count)
    if rates is None or len(rates) < count:
        return None
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    return df

def compute_ema(df, period):
    if df is None or len(df) < period:
        return None
    return ta.trend.EMAIndicator(df['close'], window=period).ema_indicator()

def compute_atr(df, period):
    """Return the latest ATR value from the dataframe."""
    if df is None or len(df) < period:
        return None
    atr_ind = ta.volatility.AverageTrueRange(high=df['high'], low=df['low'], close=df['close'], window=period)
    return atr_ind.average_true_range().iloc[-1]

def get_open_position():
    positions = mt5.positions_get(symbol=SYMBOL)
    if positions is None:
        return None
    for pos in positions:
        if pos.magic == MAGIC:
            return pos
    return None

def send_order(order_type, volume, price, sl=0, tp=0, comment=""):
    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": SYMBOL,
        "volume": volume,
        "type": order_type,
        "price": price,
        "deviation": 10,
        "magic": MAGIC,
        "comment": comment,
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }
    if sl > 0:
        request["sl"] = sl
    if tp > 0:
        request["tp"] = tp

    result = mt5.order_send(request)
    if result is None:
        error_code = mt5.last_error()
        print(colorize(f"Order send failed: mt5.last_error() = {error_code}", Colors.RED))
        return False
    if result.retcode != mt5.TRADE_RETCODE_DONE:
        print(colorize(f"Order failed: retcode={result.retcode}, comment={result.comment}", Colors.RED))
        return False
    print(colorize(f"Order executed: {'BUY' if order_type == mt5.ORDER_TYPE_BUY else 'SELL'} {volume} at {price}", Colors.GREEN))
    if sl > 0:
        print(colorize(f"  SL set at {sl}", Colors.YELLOW))
    if tp > 0:
        print(colorize(f"  TP set at {tp}", Colors.YELLOW))
    return True

def close_position(position):
    if position.type == 0:   # buy
        close_type = mt5.ORDER_TYPE_SELL
        price = mt5.symbol_info_tick(SYMBOL).bid
    else:                    # sell
        close_type = mt5.ORDER_TYPE_BUY
        price = mt5.symbol_info_tick(SYMBOL).ask

    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": SYMBOL,
        "volume": position.volume,
        "type": close_type,
        "position": position.ticket,
        "price": price,
        "deviation": 10,
        "magic": MAGIC,
        "comment": "close",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }

    result = mt5.order_send(request)
    if result is None:
        error_code = mt5.last_error()
        print(colorize(f"Close order failed: mt5.last_error() = {error_code}", Colors.RED))
        return False
    if result.retcode != mt5.TRADE_RETCODE_DONE:
        print(colorize(f"Failed to close position: retcode={result.retcode}, comment={result.comment}", Colors.RED))
        return False
    print(colorize(f"Position closed: {position.ticket}", Colors.GREEN))
    return True

# ---------- Main loop ----------
last_candle_time = None
last_process = time.time() - INTERVAL

try:
    while True:
        now = time.time()
        if now - last_process < INTERVAL:
            time.sleep(0.05)
            continue
        last_process = now

        bid, ask = get_current_prices()
        if bid is None or ask is None:
            print(colorize("No tick data, waiting...", Colors.YELLOW))
            continue
        current_price = (bid + ask) / 2.0

        # Fetch enough candles: need at least max(EMA, ATR) + 2
        candles_needed = max(EMA_PERIOD, ATR_PERIOD) + 2
        df = get_candles(candles_needed)
        if df is None or len(df) < candles_needed:
            print(colorize("Not enough candles yet, waiting...", Colors.YELLOW))
            continue

        # Compute EMA series
        ema_series = compute_ema(df, EMA_PERIOD)
        if ema_series is None:
            print(colorize("Could not compute EMA, waiting...", Colors.YELLOW))
            continue

        prev_candle = df.iloc[-2]
        current_ema = ema_series.iloc[-1]
        prev_ema = ema_series.iloc[-2]

        current_candle_time = prev_candle['time']
        if last_candle_time != current_candle_time:
            print(colorize(f"\n--- New candle: {current_candle_time} ---", Colors.CYAN))
            last_candle_time = current_candle_time

        position = get_open_position()
        if position:
            # Exit when price crosses EMA
            if (position.type == 0 and current_price < current_ema) or \
               (position.type == 1 and current_price > current_ema):
                print(colorize(f"Exit signal: price {current_price:.5f} crossed EMA {current_ema:.5f}", Colors.YELLOW))
                close_position(position)
                position = None
        else:
            # Check entry signals
            atr_value = compute_atr(df, ATR_PERIOD)
            if atr_value is None:
                print(colorize("ATR not available, skipping entry check", Colors.YELLOW))
                # but we still continue to log
            else:
                # Sell condition
                if (prev_candle['open'] > prev_ema and prev_candle['close'] < current_ema and
                    current_price < current_ema - entry_threshold):
                    print(colorize("Sell signal detected.", Colors.YELLOW))
                    # Calculate SL and TP for sell
                    sl_price = bid + (atr_value * SL_MULT)
                    tp_price = bid - (atr_value * TP_MULT)
                    print(colorize(f"ATR: {atr_value:.5f}, SL: {sl_price:.5f}, TP: {tp_price:.5f}", Colors.CYAN))
                    send_order(mt5.ORDER_TYPE_SELL, VOLUME, bid, sl=sl_price, tp=tp_price, comment=f"Python SELL@{bid}")
                # Buy condition
                elif (prev_candle['open'] < prev_ema and prev_candle['close'] > current_ema and
                      current_price > current_ema + entry_threshold):
                    print(colorize("Buy signal detected.", Colors.YELLOW))
                    # Calculate SL and TP for buy
                    sl_price = ask - (atr_value * SL_MULT)
                    tp_price = ask + (atr_value * TP_MULT)
                    print(colorize(f"ATR: {atr_value:.5f}, SL: {sl_price:.5f}, TP: {tp_price:.5f}", Colors.CYAN))
                    send_order(mt5.ORDER_TYPE_BUY, VOLUME, ask, sl=sl_price, tp=tp_price, comment=f"Python BUY@{ask}")

        # --- Logging ---
        pos_status = "No position" if position is None else f"Position: {'BUY' if position.type == 0 else 'SELL'} {position.volume} lots, profit={position.profit:.2f}"
        timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
        msg = (f"[{colorize(timestamp, Colors.BLUE)}] "
               f"{colorize(SYMBOL, Colors.WHITE)} price={colorize(f'{current_price:.5f}', Colors.CYAN)} "
               f"EMA{EMA_PERIOD}={colorize(f'{current_ema:.5f}', Colors.MAGENTA)} | "
               f"{colorize(pos_status, Colors.GREEN if position else Colors.WHITE)}")
        print(msg)

        if position:
            dist = current_price - current_ema
            dist_text = "above" if dist > 0 else "below"
            color_dist = Colors.GREEN if (position.type == 0 and dist > 0) or (position.type == 1 and dist < 0) else Colors.RED
            print(f"  Price is {colorize(f'{abs(dist):.5f}', color_dist)} {dist_text} EMA")

        time.sleep(0.05)

except KeyboardInterrupt:
    print(colorize("\nScript stopped by user.", Colors.YELLOW))

mt5.shutdown()
