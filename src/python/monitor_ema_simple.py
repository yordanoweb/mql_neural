"""
MT5 Trading Script - EMA Cross Entry/Exit Strategy
Entry: EMA Cross + Candle Close Confirmation + ATR Distance
Exit: ATR-based SL/TP + Inverse EMA Cross + Candle Close + ATR Distance
Filters: H1 Candle Direction + Trading Hours
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
parser = argparse.ArgumentParser(description="MT5 EMA Cross Strategy - Simple Entry/Exit")
parser.add_argument("--symbol", type=str, default="EURUSD", help="Trading symbol (default: EURUSD)")
parser.add_argument("--timeframe", type=str, default="M1", help="Candle timeframe (default: M1)")
parser.add_argument("--ema_period", type=int, default=9, help="EMA period (default: 9)")
parser.add_argument("--atr_period", type=int, default=14, help="ATR period (default: 14)")
parser.add_argument("--sl_multiplier", type=float, default=2.0, help="Stop loss multiplier of ATR (default: 2.0)")
parser.add_argument("--tp_multiplier", type=float, default=3.0, help="Take profit multiplier of ATR (default: 3.0)")
parser.add_argument("--magic", type=int, default=91234569, help="Magic number (default: 91234569)")
parser.add_argument("--volume", type=float, default=0.1, help="Order volume in lots (default: 0.1)")
parser.add_argument("--interval", type=float, default=5.0, help="Seconds between processing steps (default: 5)")

# Entry confirmation parameters
parser.add_argument("--entry_confirmation_atr", type=float, default=0.5, 
                    help="ATR multiplier for entry confirmation distance (default: 0.5)")

# Exit parameters
parser.add_argument("--exit_cross_atr", type=float, default=1.0, 
                    help="ATR multiplier for exit cross distance (default: 1.0)")

# H1 Candle direction gate parameter
parser.add_argument("--use_h1_candle_gate", action="store_true", 
                    help="Only buy on H1 bullish candles, sell on H1 bearish candles")

# Time parameters
parser.add_argument("--start_hour", type=int, default=0, help="Start trading hour (0-23)")
parser.add_argument("--end_hour", type=int, default=23, help="End trading hour (0-23)")

args = parser.parse_args()

SYMBOL = args.symbol
TIMEFRAME = parse_timeframe(args.timeframe)
EMA_PERIOD = args.ema_period
ATR_PERIOD = args.atr_period
SL_MULT = args.sl_multiplier
TP_MULT = args.tp_multiplier
MAGIC = args.magic
VOLUME = args.volume
INTERVAL = args.interval
ENTRY_CONFIRM_ATR = args.entry_confirmation_atr
EXIT_CROSS_ATR = args.exit_cross_atr
USE_H1_CANDLE_GATE = args.use_h1_candle_gate
START_HOUR = args.start_hour
END_HOUR = args.end_hour

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
print(f"Entry confirmation ATR multiplier: {ENTRY_CONFIRM_ATR}")
print(f"ATR period: {ATR_PERIOD}, SL multiplier: {SL_MULT}, TP multiplier: {TP_MULT}")
print(f"Exit cross ATR multiplier: {EXIT_CROSS_ATR}")
print(f"H1 candle gate: {'ENABLED' if USE_H1_CANDLE_GATE else 'DISABLED'}")
print(f"Trading hours: {START_HOUR}:00 - {END_HOUR}:59")
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
    if df is None or len(df) < period:
        return None
    atr_ind = ta.volatility.AverageTrueRange(high=df['high'], low=df['low'], close=df['close'], window=period)
    return atr_ind.average_true_range()

def get_h1_candle_direction():
    """
    Check the direction of the last COMPLETED H1 candle.
    Returns: 1 for bullish (close > open), -1 for bearish, 0 for neutral or error
    """
    rates = mt5.copy_rates_from_pos(SYMBOL, mt5.TIMEFRAME_H1, 1, 1)
    if rates is None or len(rates) < 1:
        return 0

    prev_h1 = rates[0]

    if prev_h1['close'] > prev_h1['open']:
        return 1  # Bullish
    elif prev_h1['close'] < prev_h1['open']:
        return -1  # Bearish
    else:
        return 0  # Neutral

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
        "comment": "EMA cross exit",
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

# Variables to track entry conditions across candles
entry_setup_buy = False  # EMA cross detected, waiting for confirmation
entry_setup_sell = False
entry_cross_candle_time = None

try:
    while True:
        now = time.time()
        if now - last_process < INTERVAL:
            time.sleep(0.05)
            continue
        last_process = now

        # Do not trade outside of time
        _h = int(time.strftime("%H")) 
        _m = int(time.strftime("%M"))
        if _h < START_HOUR or _h > END_HOUR:
            print(colorize(f"Outside trading hours ({_h}:{_m}), waiting...", Colors.YELLOW))
            time.sleep(30)
            continue

        bid, ask = get_current_prices()
        if bid is None or ask is None:
            print(colorize("No tick data, waiting...", Colors.YELLOW))
            continue
        current_price = (bid + ask) / 2.0

        # Get candles for EMA and ATR calculation
        candles_needed = max(50, EMA_PERIOD + 10, ATR_PERIOD + 10)
        df = get_candles(candles_needed)
        if df is None or len(df) < candles_needed:
            print(colorize("Not enough candles yet, waiting...", Colors.YELLOW))
            continue

        # Compute indicators
        ema_series = compute_ema(df, EMA_PERIOD)
        atr_series = compute_atr(df, ATR_PERIOD)

        if ema_series is None or atr_series is None:
            print(colorize("Could not compute indicators, waiting...", Colors.YELLOW))
            continue

        # Get current ATR value
        current_atr = atr_series.iloc[-1]

        # Get completed candles data (indices -3, -2 are completed, -1 is forming)
        # For entry/exit confirmation we use completed candles
        prev_candle = df.iloc[-2]      # Last completed candle
        prev_prev_candle = df.iloc[-3]   # Second last completed candle
        current_ema = ema_series.iloc[-1]
        prev_ema = ema_series.iloc[-2]
        prev_prev_ema = ema_series.iloc[-3]

        current_candle_time = prev_candle['time']

        # Log new candle
        if last_candle_time != current_candle_time:
            print(colorize(f"\n--- New candle: {current_candle_time} ---", Colors.CYAN))
            last_candle_time = current_candle_time

        # Get H1 candle direction for filtering
        h1_direction = get_h1_candle_direction()
        h1_bullish = (h1_direction == 1)
        h1_bearish = (h1_direction == -1)

        h1_allow_buy = h1_bullish or not USE_H1_CANDLE_GATE
        h1_allow_sell = h1_bearish or not USE_H1_CANDLE_GATE

        # Check current position
        position = get_open_position()

        if position is None:
            # ========== NO POSITION - LOOK FOR ENTRY ==========

            # Detect EMA cross on completed candles
            # Bullish cross: price was below EMA, now above
            bullish_cross = (prev_prev_candle['close'] < prev_prev_ema and 
                           prev_candle['close'] > prev_ema)

            # Bearish cross: price was above EMA, now below  
            bearish_cross = (prev_prev_candle['close'] > prev_prev_ema and 
                             prev_candle['close'] < prev_ema)

            # Check entry confirmation: price distance from EMA
            price_ema_distance = abs(prev_candle['close'] - prev_ema)
            min_distance = current_atr * ENTRY_CONFIRM_ATR
            distance_ok = price_ema_distance >= min_distance

            # Entry execution
            if bullish_cross and distance_ok and h1_allow_buy:
                print(colorize(f"BUY signal: EMA cross confirmed + distance {price_ema_distance:.5f} >= {min_distance:.5f}", Colors.GREEN))
                sl_price = ask - (current_atr * SL_MULT)
                tp_price = ask + (current_atr * TP_MULT)
                send_order(mt5.ORDER_TYPE_BUY, VOLUME, ask, sl=sl_price, tp=tp_price, 
                          comment=f"EMA Buy@{ask}")

            elif bearish_cross and distance_ok and h1_allow_sell:
                print(colorize(f"SELL signal: EMA cross confirmed + distance {price_ema_distance:.5f} >= {min_distance:.5f}", Colors.GREEN))
                sl_price = bid + (current_atr * SL_MULT)
                tp_price = bid - (current_atr * TP_MULT)
                send_order(mt5.ORDER_TYPE_SELL, VOLUME, bid, sl=sl_price, tp=tp_price,
                          comment=f"EMA Sell@{bid}")

            else:
                # Log why entry didn't happen
                if bullish_cross or bearish_cross:
                    reasons = []
                    if not distance_ok:
                        reasons.append(f"distance {price_ema_distance:.5f} < {min_distance:.5f}")
                    if bullish_cross and not h1_allow_buy and USE_H1_CANDLE_GATE:
                        reasons.append("H1 not bullish")
                    if bearish_cross and not h1_allow_sell and USE_H1_CANDLE_GATE:
                        reasons.append("H1 not bearish")
                    if reasons:
                        direction = "BUY" if bullish_cross else "SELL"
                        print(colorize(f"EMA {direction} cross detected but: {', '.join(reasons)}", Colors.YELLOW))

        else:
            # ========== POSITION OPEN - CHECK EXIT CONDITIONS ==========

            # Exit condition: Inverse EMA cross + candle close confirmation + ATR distance

            if position.type == 0:  # BUY position
                # Inverse cross: price was above EMA, now below (on completed candles)
                inverse_cross = (prev_prev_candle['close'] > prev_prev_ema and 
                                prev_candle['close'] < prev_ema)

                # Distance below EMA
                distance_below = prev_ema - prev_candle['close']
                min_exit_distance = current_atr * EXIT_CROSS_ATR
                distance_ok = distance_below >= min_exit_distance

                if inverse_cross and distance_ok:
                    print(colorize(f"BUY exit: Inverse EMA cross + distance {distance_below:.5f} >= {min_exit_distance:.5f}", Colors.YELLOW))
                    close_position(position)

            else:  # SELL position
                # Inverse cross: price was below EMA, now above
                inverse_cross = (prev_prev_candle['close'] < prev_prev_ema and 
                                prev_candle['close'] > prev_ema)

                # Distance above EMA
                distance_above = prev_candle['close'] - prev_ema
                min_exit_distance = current_atr * EXIT_CROSS_ATR
                distance_ok = distance_above >= min_exit_distance

                if inverse_cross and distance_ok:
                    print(colorize(f"SELL exit: Inverse EMA cross + distance {distance_above:.5f} >= {min_exit_distance:.5f}", Colors.YELLOW))
                    close_position(position)

        # ---------- Logging ----------
        pos_status = "No position"
        if position is not None:
            pos_status = f"Position: {'BUY' if position.type == 0 else 'SELL'} {position.volume} lots, profit={position.profit:.2f}"

        timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
        msg = (f"[{colorize(timestamp, Colors.BLUE)}] "
               f"{colorize(SYMBOL, Colors.WHITE)} price={colorize(f'{current_price:.5f}', Colors.CYAN)} "
               f"EMA{EMA_PERIOD}={colorize(f'{current_ema:.5f}', Colors.MAGENTA)} "
               f"ATR={colorize(f'{current_atr:.5f}', Colors.YELLOW)} | "
               f"{colorize(pos_status, Colors.GREEN if position else Colors.WHITE)}")
        print(msg)

        if position:
            dist = current_price - current_ema
            dist_text = "above" if dist > 0 else "below"
            color_dist = Colors.GREEN if (position.type == 0 and dist > 0) or (position.type == 1 and dist < 0) else Colors.RED
            print(f"  Price is {colorize(f'{abs(dist):.5f}', color_dist)} {dist_text} EMA")

        # H1 status
        h1_str = "BULLISH" if h1_direction == 1 else "BEARISH" if h1_direction == -1 else "NEUTRAL"
        h1_color = Colors.GREEN if h1_direction == 1 else Colors.RED if h1_direction == -1 else Colors.WHITE
        h1_status = f"H1: {colorize(h1_str, h1_color)}"
        if USE_H1_CANDLE_GATE:
            h1_status += f" | BuyOK:{h1_allow_buy} SellOK:{h1_allow_sell}"
        print(f"  {h1_status}")

        time.sleep(0.05)

except KeyboardInterrupt:
    print(colorize("\nScript stopped by user.", Colors.YELLOW))

mt5.shutdown()
