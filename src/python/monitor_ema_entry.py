"""
MT5 Trading Script with EMA-based Entry and Exit – Colorful Output
"""

import os
import time
import argparse
import sys
import MetaTrader5 as mt5
import pandas as pd
import ta

# Try to import colorama for Windows ANSI support; if not available, fallback to plain text
try:
    import colorama
    colorama.init(autoreset=True)
    COLORS_SUPPORTED = True
except ImportError:
    COLORS_SUPPORTED = False

# ----------------------------------------------------------------------
# ANSI color codes (if supported)
# ----------------------------------------------------------------------
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
    """Wrap text with ANSI color codes if supported."""
    if COLORS_SUPPORTED:
        return f"{color}{text}{Colors.RESET}"
    return text

# ----------------------------------------------------------------------
# Timeframe mapping
# ----------------------------------------------------------------------
TIMEFRAME_MAP = {
    "M1":  mt5.TIMEFRAME_M1,
    "M2":  mt5.TIMEFRAME_M2,
    "M3":  mt5.TIMEFRAME_M3,
    "M4":  mt5.TIMEFRAME_M4,
    "M5":  mt5.TIMEFRAME_M5,
    "M6":  mt5.TIMEFRAME_M6,
    "M10": mt5.TIMEFRAME_M10,
    "M12": mt5.TIMEFRAME_M12,
    "M15": mt5.TIMEFRAME_M15,
    "M20": mt5.TIMEFRAME_M20,
    "M30": mt5.TIMEFRAME_M30,
    "H1":  mt5.TIMEFRAME_H1,
    "H2":  mt5.TIMEFRAME_H2,
    "H3":  mt5.TIMEFRAME_H3,
    "H4":  mt5.TIMEFRAME_H4,
    "H6":  mt5.TIMEFRAME_H6,
    "H8":  mt5.TIMEFRAME_H8,
    "H12": mt5.TIMEFRAME_H12,
    "D1":  mt5.TIMEFRAME_D1,
    "W1":  mt5.TIMEFRAME_W1,
    "MN1": mt5.TIMEFRAME_MN1,
}

def parse_timeframe(tf_str):
    """Convert timeframe string to MT5 timeframe constant."""
    tf_str = tf_str.upper()
    if tf_str not in TIMEFRAME_MAP:
        raise ValueError(f"Unsupported timeframe: {tf_str}. Use one of: {', '.join(TIMEFRAME_MAP.keys())}")
    return TIMEFRAME_MAP[tf_str]

# ----------------------------------------------------------------------
# Argument parsing
# ----------------------------------------------------------------------
parser = argparse.ArgumentParser(
    description="MT5 Trading Script - EMA Cross with Threshold"
)
parser.add_argument(
    "--symbol", type=str, default="EURUSD",
    help="Trading symbol (default: EURUSD)"
)
parser.add_argument(
    "--timeframe", type=str, default="M1",
    help="Candle timeframe (e.g., M1, M5, H1, D1) (default: M1)"
)
parser.add_argument(
    "--ema_period", type=int, default=9,
    help="EMA period (default: 9)"
)
parser.add_argument(
    "--magic", type=int, default=91234569,
    help="Magic number to identify orders (default: 91234569)"
)
parser.add_argument(
    "--volume", type=float, default=0.1,
    help="Order volume in lots (default: 0.1)"
)
parser.add_argument(
    "--entry_points", type=float, default=10.0,
    help="Number of points (pips) away from EMA to trigger entry (default: 10)"
)
parser.add_argument(
    "--interval", type=float, default=5.0,
    help="Seconds between processing steps (default: 5)"
)
args = parser.parse_args()

# ----------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------
SYMBOL = args.symbol
TIMEFRAME = parse_timeframe(args.timeframe)
EMA_PERIOD = args.ema_period
MAGIC = args.magic
VOLUME = args.volume
ENTRY_POINTS = args.entry_points
INTERVAL = args.interval

# ----------------------------------------------------------------------
# MT5 Initialization
# ----------------------------------------------------------------------
if not mt5.initialize():
    print(colorize("Failed to initialize MT5, error code = " + str(mt5.last_error()), Colors.RED))
    quit()

# Get symbol info
symbol_info = mt5.symbol_info(SYMBOL)
if symbol_info is None:
    print(colorize(f"Symbol {SYMBOL} not found, error = {mt5.last_error()}", Colors.RED))
    mt5.shutdown()
    quit()

# Ensure symbol is visible in MarketWatch
if not symbol_info.visible:
    if not mt5.symbol_select(SYMBOL, True):
        print(colorize(f"Failed to select {SYMBOL}", Colors.RED))
        mt5.shutdown()
        quit()

# Calculate point value for the symbol
point = symbol_info.point
entry_threshold = ENTRY_POINTS * point

print(colorize("MT5 initialized", Colors.GREEN) + f" – trading {SYMBOL} on {args.timeframe} timeframe with EMA{EMA_PERIOD}")
print(f"Magic: {MAGIC}, Volume: {VOLUME}, Entry threshold: {ENTRY_POINTS} points ({entry_threshold:.5f})")
print(f"Checking every {INTERVAL} seconds\n")

# ----------------------------------------------------------------------
# Helper functions
# ----------------------------------------------------------------------
def get_current_prices():
    """Return current bid and ask prices."""
    tick = mt5.symbol_info_tick(SYMBOL)
    if tick is None:
        return None, None
    return tick.bid, tick.ask

def get_candles(count):
    """Fetch last 'count' candles (OHLC) for the symbol using selected timeframe."""
    rates = mt5.copy_rates_from_pos(SYMBOL, TIMEFRAME, 0, count)
    if rates is None or len(rates) < count:
        return None
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    return df

def compute_ema(df, period):
    """Compute EMA on close prices."""
    if df is None or len(df) < period:
        return None
    return ta.trend.EMAIndicator(df['close'], window=period).ema_indicator()

def has_open_position():
    """Check if there is an open position with our magic number."""
    positions = mt5.positions_get(symbol=SYMBOL)
    if positions is None:
        return False
    for pos in positions:
        if pos.magic == MAGIC:
            return True
    return False

def get_open_position():
    """Return the open position with our magic number, or None."""
    positions = mt5.positions_get(symbol=SYMBOL)
    if positions is None:
        return None
    for pos in positions:
        if pos.magic == MAGIC:
            return pos
    return None

def send_order(order_type, volume, price, sl=0, tp=0, comment=""):
    """
    Send a market order.
    order_type: mt5.ORDER_TYPE_BUY or mt5.ORDER_TYPE_SELL
    """
    # Build the base request without sl/tp if they are 0
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
    # Only add sl and tp if they are set (greater than 0)
    if sl > 0:
        request["sl"] = sl
    if tp > 0:
        request["tp"] = tp

    result = mt5.order_send(request)

    # Check if the request failed entirely
    if result is None:
        error_code = mt5.last_error()
        print(colorize(f"Order send failed: mt5.last_error() = {error_code}", Colors.RED))
        return False

    # Check if the order was accepted by the server
    if result.retcode != mt5.TRADE_RETCODE_DONE:
        print(colorize(f"Order failed: retcode={result.retcode}, comment={result.comment}", Colors.RED))
        return False

    print(colorize(f"Order executed: {'BUY' if order_type == mt5.ORDER_TYPE_BUY else 'SELL'} {volume} at {price}", Colors.GREEN))
    return True

def close_position(position):
    """
    Close an open position.
    position: a position object returned by mt5.positions_get()
    """
    if position.type == 0:  # Buy position
        close_type = mt5.ORDER_TYPE_SELL
        price = mt5.symbol_info_tick(SYMBOL).bid
    else:  # Sell position
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

# ----------------------------------------------------------------------
# Main loop
# ----------------------------------------------------------------------
last_candle_time = None
last_process = time.time() - INTERVAL  # Force immediate first run

try:
    while True:
        now = time.time()
        if now - last_process < INTERVAL:
            time.sleep(0.05)
            continue
        last_process = now

        # Get current prices
        bid, ask = get_current_prices()
        if bid is None or ask is None:
            print(colorize("No tick data, waiting...", Colors.YELLOW))
            continue
        current_price = (bid + ask) / 2.0

        # Get enough candles to compute EMA (need EMA_PERIOD + 2 for previous candle check)
        candles_needed = EMA_PERIOD + 2
        df = get_candles(candles_needed)
        if df is None or len(df) < candles_needed:
            print(colorize("Not enough candles yet, waiting...", Colors.YELLOW))
            continue

        # Compute EMA series
        ema_series = compute_ema(df, EMA_PERIOD)
        if ema_series is None:
            print(colorize("Could not compute EMA, waiting...", Colors.YELLOW))
            continue

        # Get the latest two candles
        prev_candle = df.iloc[-2]   # previous completed candle
        current_ema = ema_series.iloc[-1]      # EMA after the last completed candle
        prev_ema = ema_series.iloc[-2]         # EMA after the previous candle

        # Check if a new candle has formed
        current_candle_time = prev_candle['time']
        if last_candle_time != current_candle_time:
            print(colorize(f"\n--- New candle: {current_candle_time} ---", Colors.CYAN))
            last_candle_time = current_candle_time

        # Check for open position
        position = get_open_position()
        if position:
            # Exit condition: price crosses the EMA
            if (position.type == 0 and current_price < current_ema) or \
               (position.type == 1 and current_price > current_ema):
                print(colorize(f"Exit signal: price {current_price:.5f} crossed EMA {current_ema:.5f}", Colors.YELLOW))
                close_position(position)
                position = None
        else:
            # No open position – check entry conditions
            # Sell condition
            if (prev_candle['open'] > prev_ema and prev_candle['close'] < current_ema and
                current_price < current_ema - entry_threshold):
                print(colorize("Sell signal detected.", Colors.YELLOW))
                send_order(mt5.ORDER_TYPE_SELL, VOLUME, bid)
            # Buy condition
            elif (prev_candle['open'] < prev_ema and prev_candle['close'] > current_ema and
                  current_price > current_ema + entry_threshold):
                print(colorize("Buy signal detected.", Colors.YELLOW))
                send_order(mt5.ORDER_TYPE_BUY, VOLUME, ask)
            # else: no signal

        # ------------------------------------------------------------------
        # Logging – status every interval
        # ------------------------------------------------------------------
        pos_status = "No position" if position is None else f"Position: {'BUY' if position.type == 0 else 'SELL'} {position.volume} lots, profit={position.profit:.2f}"
        timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
        # Color the timestamp and price/EMA
        msg = (f"[{colorize(timestamp, Colors.BLUE)}] "
               f"{colorize(SYMBOL, Colors.WHITE)} price={colorize(f'{current_price:.5f}', Colors.CYAN)} "
               f"EMA{EMA_PERIOD}={colorize(f'{current_ema:.5f}', Colors.MAGENTA)} | "
               f"{colorize(pos_status, Colors.GREEN if position else Colors.WHITE)}")
        print(msg)

        # If we have a position, log also the distance to EMA
        if position:
            dist = current_price - current_ema
            dist_text = f"above" if dist > 0 else "below"
            color_dist = Colors.GREEN if (position.type == 0 and dist > 0) or (position.type == 1 and dist < 0) else Colors.RED
            print(f"  Price is {colorize(f'{abs(dist):.5f}', color_dist)} {dist_text} EMA")

        time.sleep(0.05)

except KeyboardInterrupt:
    print(colorize("\nScript stopped by user.", Colors.YELLOW))

mt5.shutdown()
