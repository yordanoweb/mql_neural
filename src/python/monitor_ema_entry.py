"""
MT5 Trading Script with EMA-based Entry and Exit

- Sell: previous candle opened above EMA and closed below EMA,
       and current price is N points below EMA.
- Buy:  previous candle opened below EMA and closed above EMA,
       and current price is N points above EMA.
- Close: when price crosses the EMA (exit).
- Only one order at a time.
"""

import time
import argparse
import MetaTrader5 as mt5
import pandas as pd
import ta

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
    "--ema_period", type=int, default=9,
    help="EMA period (default: 9)"
)
parser.add_argument(
    "--magic", type=int, default=123456,
    help="Magic number to identify orders (default: 123456)"
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
    "--interval", type=float, default=15.0,
    help="Seconds between processing steps (default: 5)"
)
args = parser.parse_args()

# ----------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------
SYMBOL = args.symbol
EMA_PERIOD = args.ema_period
MAGIC = args.magic
VOLUME = args.volume
ENTRY_POINTS = args.entry_points
INTERVAL = args.interval

# ----------------------------------------------------------------------
# MT5 Initialization
# ----------------------------------------------------------------------
if not mt5.initialize():
    print("Failed to initialize MT5, error code =", mt5.last_error())
    quit()

# Get symbol info
symbol_info = mt5.symbol_info(SYMBOL)
if symbol_info is None:
    print(f"Symbol {SYMBOL} not found, error = {mt5.last_error()}")
    mt5.shutdown()
    quit()

# Ensure symbol is visible in MarketWatch
if not symbol_info.visible:
    if not mt5.symbol_select(SYMBOL, True):
        print(f"Failed to select {SYMBOL}")
        mt5.shutdown()
        quit()

# Calculate point value for the symbol
point = symbol_info.point
entry_threshold = ENTRY_POINTS * point

print(f"MT5 initialized – trading {SYMBOL} with EMA{EMA_PERIOD}")
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
    """Fetch last 'count' candles (OHLC) for the symbol."""
    rates = mt5.copy_rates_from_pos(SYMBOL, mt5.TIMEFRAME_M1, 0, count)
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
    # For buy, price is ask; for sell, price is bid
    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": SYMBOL,
        "volume": volume,
        "type": order_type,
        "price": price,
        "sl": sl,
        "tp": tp,
        "deviation": 10,
        "magic": MAGIC,
        "comment": comment,
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }
    result = mt5.order_send(request)
    if result.retcode != mt5.TRADE_RETCODE_DONE:
        print(f"Order failed: {result.comment} (retcode={result.retcode})")
        return False
    print(f"Order executed: {'BUY' if order_type == mt5.ORDER_TYPE_BUY else 'SELL'} {volume} at {price}")
    return True

def close_position(position):
    """Close the given position."""
    order_type = mt5.ORDER_TYPE_BUY if position.type == 1 else mt5.ORDER_TYPE_SELL
    # Close with opposite order type
    close_type = mt5.ORDER_TYPE_SELL if order_type == mt5.ORDER_TYPE_BUY else mt5.ORDER_TYPE_BUY
    price = mt5.symbol_info_tick(SYMBOL).bid if close_type == mt5.ORDER_TYPE_SELL else mt5.symbol_info_tick(SYMBOL).ask
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
    if result.retcode != mt5.TRADE_RETCODE_DONE:
        print(f"Failed to close position: {result.comment} (retcode={result.retcode})")
        return False
    print(f"Position closed: {position.ticket}")
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
            print("No tick data, waiting...")
            continue
        current_price = (bid + ask) / 2.0

        # Get enough candles to compute EMA (need EMA_PERIOD + 2 for previous candle check)
        candles_needed = EMA_PERIOD + 2
        df = get_candles(candles_needed)
        if df is None or len(df) < candles_needed:
            print("Not enough candles yet, waiting...")
            continue

        # Compute EMA series
        ema_series = compute_ema(df, EMA_PERIOD)
        if ema_series is None:
            print("Could not compute EMA, waiting...")
            continue

        # Get the latest two candles (index -1 is current forming, -2 is previous completed)
        # We need the previous completed candle (index -2) for open/close condition
        prev_candle = df.iloc[-2]   # previous completed candle
        current_ema = ema_series.iloc[-1]      # EMA after the last completed candle (based on close)
        prev_ema = ema_series.iloc[-2]         # EMA after the previous candle

        # Check if a new candle has formed (comparing timestamps)
        current_candle_time = prev_candle['time']
        if last_candle_time != current_candle_time:
            # New candle detected – good time to log and evaluate entry
            print(f"\n--- New candle: {current_candle_time} ---")
            last_candle_time = current_candle_time

        # ------------------------------------------------------------------
        # Check for open position
        # ------------------------------------------------------------------
        position = get_open_position()
        if position:
            # Exit condition: price crosses the EMA
            if (position.type == 0 and current_price < current_ema) or \
               (position.type == 1 and current_price > current_ema):
                print(f"Exit signal: price {current_price:.5f} crossed EMA {current_ema:.5f}")
                close_position(position)
                position = None  # just closed
        else:
            # No open position – check entry conditions
            # Sell condition:
            # previous candle opened above prev_ema AND closed below current_ema
            # AND current price is below current_ema by entry_threshold
            if (prev_candle['open'] > prev_ema and prev_candle['close'] < current_ema and
                current_price < current_ema - entry_threshold):
                print("Sell signal detected.")
                send_order(mt5.ORDER_TYPE_SELL, VOLUME, bid)
            # Buy condition:
            # previous candle opened below prev_ema AND closed above current_ema
            # AND current price is above current_ema by entry_threshold
            elif (prev_candle['open'] < prev_ema and prev_candle['close'] > current_ema and
                  current_price > current_ema + entry_threshold):
                print("Buy signal detected.")
                send_order(mt5.ORDER_TYPE_BUY, VOLUME, ask)
            else:
                # No signal – just log status
                pass

        # ------------------------------------------------------------------
        # Logging – status every interval
        # ------------------------------------------------------------------
        pos_status = "No position" if position is None else f"Position: {'BUY' if position.type == 0 else 'SELL'} {position.volume} lots, profit={position.profit:.2f}"
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {SYMBOL} price={current_price:.5f} EMA{EMA_PERIOD}={current_ema:.5f} | {pos_status}")

        # If we have a position, log also the distance to EMA
        if position:
            dist = current_price - current_ema
            dist_text = f"above" if dist > 0 else "below"
            print(f"  Price is {abs(dist):.5f} {dist_text} EMA")

        time.sleep(0.05)

except KeyboardInterrupt:
    print("\nScript stopped by user.")

mt5.shutdown()
