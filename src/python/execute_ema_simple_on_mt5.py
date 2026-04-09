"""
MT5 Trading Script - SimpleEMAStrategy (from backtesting)
Entry: 3 consecutive rising EMAs -> LONG, 3 consecutive falling EMAs -> SHORT
Exit: price change by exit_percent OR first opposing EMA direction
Filters: trading hours, optional cooldown between trades
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
parser = argparse.ArgumentParser(description="MT5 Simple EMA Strategy - 3-consecutive EMA direction")
parser.add_argument("--symbol", type=str, default="EURUSD", help="Trading symbol")
parser.add_argument("--timeframe", type=str, default="M5", help="Candle timeframe")
parser.add_argument("--ema_period", type=int, default=20, help="EMA period")
parser.add_argument("--trade_start_hour", type=int, default=14, help="Start hour (0-23)")
parser.add_argument("--trade_end_hour", type=int, default=22, help="End hour (0-23)")
parser.add_argument("--cooldown", type=int, default=120, help="Cooldown in minutes after exit before new entry")
parser.add_argument("--exit_percent", type=float, default=1.0, help="Price change percent to exit (e.g., 1.0 = 1%%)")
parser.add_argument("--magic", type=int, default=int(time.time()), help="Magic number")
parser.add_argument("--volume", type=float, default=0.1, help="Order volume in lots")
parser.add_argument("--interval", type=float, default=5.0, help="Seconds between processing steps")
parser.add_argument("--debug", action="store_true", help="Enable debug logging")

args = parser.parse_args()

SYMBOL = args.symbol
TIMEFRAME = parse_timeframe(args.timeframe)
EMA_PERIOD = args.ema_period
TRADE_START_HOUR = args.trade_start_hour
TRADE_END_HOUR = args.trade_end_hour
COOLDOWN_MINUTES = args.cooldown
EXIT_PERCENT = args.exit_percent / 100.0  # convert to decimal
MAGIC = args.magic
VOLUME = args.volume
INTERVAL = args.interval
DEBUG = args.debug

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

print(colorize("MT5 initialized", Colors.GREEN) +
      f" – trading {SYMBOL} on {args.timeframe} with EMA{EMA_PERIOD}")
print(f"Magic: {MAGIC}, Volume: {VOLUME}")
print(f"Trading hours: {TRADE_START_HOUR}:00 - {TRADE_END_HOUR}:59")
print(f"Cooldown: {COOLDOWN_MINUTES} minutes after exit")
print(f"Exit percent: {EXIT_PERCENT*100}%")
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

def get_open_position():
    positions = mt5.positions_get(symbol=SYMBOL)
    if positions is None:
        return None
    for pos in positions:
        if pos.magic == MAGIC:
            return pos
    return None

def send_order(order_type, volume, price, comment=""):
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
    result = mt5.order_send(request)
    if result is None:
        error_code = mt5.last_error()
        print(colorize(f"Order send failed: mt5.last_error() = {error_code}", Colors.RED))
        return False
    if result.retcode != mt5.TRADE_RETCODE_DONE:
        print(colorize(f"Order failed: retcode={result.retcode}, comment={result.comment}", Colors.RED))
        return False
    print(colorize(f"Order executed: {'BUY' if order_type == mt5.ORDER_TYPE_BUY else 'SELL'} {volume} at {price}", Colors.GREEN))
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
        "comment": "EMA strategy exit",
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
last_exit_time = None  # timestamp of last exit (for cooldown)
entry_price = 0.0
entry_direction = None  # 1 for long, -1 for short

# Store EMA direction pattern for last N candles
pattern = []  # 1 for rising EMA, -1 for falling EMA
MAX_PATTERN_LEN = 10

try:
    while True:
        now = time.time()
        if now - last_process < INTERVAL:
            time.sleep(0.05)
            continue
        last_process = now

        # Trading hours check
        _h = int(time.strftime("%H"))
        _m = int(time.strftime("%M"))
        if _h < TRADE_START_HOUR or _h >= TRADE_END_HOUR:
            print(colorize(f"Outside trading hours ({_h}:{_m:02d}), waiting...", Colors.YELLOW))
            time.sleep(30)
            continue

        # Cooldown check (only matters if no position)
        position = get_open_position()
        if position is None and last_exit_time is not None:
            cooldown_seconds = COOLDOWN_MINUTES * 60
            if now - last_exit_time < cooldown_seconds:
                remaining = int(cooldown_seconds - (now - last_exit_time))
                print(colorize(f"Cooldown active: {remaining} seconds remaining", Colors.YELLOW))
                time.sleep(5)
                continue
            else:
                last_exit_time = None  # cooldown over

        bid, ask = get_current_prices()
        if bid is None or ask is None:
            print(colorize("No tick data, waiting...", Colors.YELLOW))
            continue
        current_price = (bid + ask) / 2.0

        # Need at least EMA_PERIOD + 5 candles
        candles_needed = max(50, EMA_PERIOD + 10)
        df = get_candles(candles_needed)
        if df is None or len(df) < candles_needed:
            print(colorize("Not enough candles yet, waiting...", Colors.YELLOW))
            continue

        ema_series = compute_ema(df, EMA_PERIOD)
        if ema_series is None:
            print(colorize("Could not compute EMA, waiting...", Colors.YELLOW))
            continue

        # Determine EMA direction for last completed candles
        # We need at least 3 completed candles to detect pattern
        if len(ema_series) < 4:
            continue

        # Last completed candle is index -2 (since -1 is forming)
        ema_prev = ema_series.iloc[-2]
        ema_prev2 = ema_series.iloc[-3]
        ema_prev3 = ema_series.iloc[-4]

        # Direction: 1 if rising, -1 if falling
        dir1 = 1 if ema_prev > ema_prev2 else (-1 if ema_prev < ema_prev2 else 0)
        dir2 = 1 if ema_prev2 > ema_prev3 else (-1 if ema_prev2 < ema_prev3 else 0)
        # We need a third direction from older candle for 3-consecutive check
        ema_prev4 = ema_series.iloc[-5]
        dir3 = 1 if ema_prev3 > ema_prev4 else (-1 if ema_prev3 < ema_prev4 else 0)

        # Update pattern list (keep last 10)
        if dir1 != 0:
            pattern.append(dir1)
        if len(pattern) > MAX_PATTERN_LEN:
            pattern = pattern[-MAX_PATTERN_LEN:]

        # Current candle time for logging
        current_candle_time = df.iloc[-2]['time']
        if last_candle_time != current_candle_time:
            print(colorize(f"\n--- New candle: {current_candle_time} ---", Colors.CYAN))
            last_candle_time = current_candle_time

        # Log EMA directions
        if DEBUG:
            print(f"EMA directions: [{dir3}, {dir2}, {dir1}]")

        # ---------- EXIT LOGIC (if position exists) ----------
        if position is not None:
            should_exit = False
            exit_reason = ""

            # Calculate price change percent from entry
            if position.type == 0:  # long
                price_change_pct = (current_price - entry_price) / entry_price
                # Exit if price increased by exit_percent
                if price_change_pct >= EXIT_PERCENT:
                    should_exit = True
                    exit_reason = f"price +{price_change_pct*100:.2f}% >= {EXIT_PERCENT*100}%"
                # Or if EMA direction turns opposite (first reversal)
                elif entry_direction == 1 and dir1 == -1:
                    should_exit = True
                    exit_reason = "EMA reversal (rising -> falling)"
            else:  # short
                price_change_pct = (entry_price - current_price) / entry_price
                if price_change_pct >= EXIT_PERCENT:
                    should_exit = True
                    exit_reason = f"price -{price_change_pct*100:.2f}% >= {EXIT_PERCENT*100}%"
                elif entry_direction == -1 and dir1 == 1:
                    should_exit = True
                    exit_reason = "EMA reversal (falling -> rising)"

            if should_exit:
                print(colorize(f"Exit signal: {exit_reason}", Colors.YELLOW))
                if close_position(position):
                    last_exit_time = time.time()
                    entry_price = 0.0
                    entry_direction = None
                continue  # skip entry this iteration

        # ---------- ENTRY LOGIC (no position) ----------
        if position is None:
            # Need at least 3 consecutive directions
            # Check if last 3 directions (dir3, dir2, dir1) are all 1 (rising) or all -1 (falling)
            if dir1 != 0 and dir2 != 0 and dir3 != 0:
                if dir1 == 1 and dir2 == 1 and dir3 == 1:
                    print(colorize(f"3 consecutive rising EMAs detected -> LONG entry", Colors.GREEN))
                    order_ok = send_order(mt5.ORDER_TYPE_BUY, VOLUME, ask, comment=f"EMA long@{ask}")
                    if order_ok:
                        entry_price = ask
                        entry_direction = 1
                        # reset pattern to avoid re-entry immediately? Not needed because position exists
                elif dir1 == -1 and dir2 == -1 and dir3 == -1:
                    print(colorize(f"3 consecutive falling EMAs detected -> SHORT entry", Colors.GREEN))
                    order_ok = send_order(mt5.ORDER_TYPE_SELL, VOLUME, bid, comment=f"EMA short@{bid}")
                    if order_ok:
                        entry_price = bid
                        entry_direction = -1

        # ---------- Logging ----------
        pos_status = "No position"
        if position is not None:
            if position.type == 0:
                pnl_pct = (current_price - entry_price) / entry_price * 100
            else:
                pnl_pct = (entry_price - current_price) / entry_price * 100
            pos_status = f"Position: {'BUY' if position.type == 0 else 'SELL'} {position.volume} lots, profit={position.profit:.2f} ({pnl_pct:+.2f}%)"

        timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
        msg = (f"[{colorize(timestamp, Colors.BLUE)}] "
               f"{colorize(SYMBOL, Colors.WHITE)} price={colorize(f'{current_price:.5f}', Colors.CYAN)} "
               f"EMA{EMA_PERIOD}={colorize(f'{ema_prev:.5f}', Colors.MAGENTA)} | "
               f"{colorize(pos_status, Colors.GREEN if position else Colors.WHITE)}")
        print(msg)

        # Show EMA direction trend
        trend = "rising" if dir1 == 1 else "falling" if dir1 == -1 else "flat"
        trend_color = Colors.GREEN if dir1 == 1 else Colors.RED if dir1 == -1 else Colors.WHITE
        print(f"  EMA direction: {colorize(trend, trend_color)} (last 3: {dir3},{dir2},{dir1})")

        time.sleep(0.05)

except KeyboardInterrupt:
    print(colorize("\nScript stopped by user.", Colors.YELLOW))

mt5.shutdown()
