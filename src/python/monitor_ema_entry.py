"""
MT5 Trading Script with EMA Entry + Stochastic + ADX Confirmation
Exit only via ATR-based Stop Loss / Take Profit
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
parser = argparse.ArgumentParser(description="MT5 Trading Script - EMA + Stochastic + ADX Confirmation")
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

# Stochastic parameters
parser.add_argument("--stoch_k", type=int, default=7, help="Stochastic %K period (default: 7)")
parser.add_argument("--stoch_d", type=int, default=3, help="Stochastic %D period (default: 3)")
parser.add_argument("--stoch_slowing", type=int, default=3, help="Stochastic slowing (default: 3)")
parser.add_argument("--stoch_overbought", type=float, default=80.0, help="Stochastic overbought level (default: 80)")
parser.add_argument("--stoch_oversold", type=float, default=20.0, help="Stochastic oversold level (default: 20)")
parser.add_argument("--stoch_bypass", action="store_true", help="Bypass Stochastic signals (always True)")

# ADX parameters
parser.add_argument("--adx_period", type=int, default=8, help="ADX period (default: 8)")
parser.add_argument("--adx_limit", type=float, default=32.0, help="ADX trend strength threshold (default: 32)")
parser.add_argument("--adx_bypass", action="store_true", help="Bypass ADX signals (always True)")
parser.add_argument("--adx_di_over", action="store_true", help="Require DI+ > DI- for buy and DI- > DI+ for sell")

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

# Stochastic args
STOCH_K = args.stoch_k
STOCH_D = args.stoch_d
STOCH_SLOW = args.stoch_slowing
STOCH_OB = args.stoch_overbought
STOCH_OS = args.stoch_oversold
STOCH_BYPASS = args.stoch_bypass

# ADX args
ADX_PERIOD = args.adx_period
ADX_LIMIT = args.adx_limit
ADX_BYPASS = args.adx_bypass
ADX_DI_OVER = args.adx_di_over

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
print(f"Stochastic: K={STOCH_K}, D={STOCH_D}, Slow={STOCH_SLOW}, OB={STOCH_OB}, OS={STOCH_OS}, bypass={STOCH_BYPASS}")
print(f"ADX: period={ADX_PERIOD}, limit={ADX_LIMIT}, bypass={ADX_BYPASS}, DI_OVER={ADX_DI_OVER}")
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
    return atr_ind.average_true_range().iloc[-1]

def compute_stochastic(df, k_period, d_period, slowing):
    """Return Stochastic %K and %D as pandas Series."""
    stoch = ta.momentum.StochasticOscillator(high=df['high'], low=df['low'], close=df['close'],
                                              window=k_period, smooth_window=d_period,
                                              fillna=True)
    # %K is the main line, %D is the signal line
    # The library's output is: stoch.stoch() = %K, stoch.stoch_signal() = %D
    # But we need to apply slowing (the built-in slowing is usually the smoothing of %K before %D)
    # Actually, ta.momentum.StochasticOscillator uses:
    #   window = %K period
    #   smooth_window = moving average of %K to produce %D
    # The slowing parameter is often the number of periods for %K smoothing; here we'll use the default
    # and let the user set k_period as the raw %K period and d_period as the moving average of %K.
    # The provided input 'slowing' is usually the same as d_period, but we'll just use the given values.
    k = stoch.stoch()
    d = stoch.stoch_signal()
    return k, d

def compute_adx(df, period):
    """Return ADX, +DI, -DI as pandas Series."""
    adx_ind = ta.trend.ADXIndicator(high=df['high'], low=df['low'], close=df['close'], window=period)
    adx = adx_ind.adx()
    plus_di = adx_ind.adx_pos()
    minus_di = adx_ind.adx_neg()
    return adx, plus_di, minus_di

def stochastic_buy_signal(k, d, idx):
    """
    Check if a buy signal is present according to the strategy.
    k, d are pandas Series; idx is the index of the most recent completed candle (e.g., -2 for prev).
    We need to look back at least 3 candles.
    """
    # Oversold crossover (2 candles ago)
    cond1 = (k.iloc[-3] < d.iloc[-3] and k.iloc[-2] > d.iloc[-2] and k.iloc[-2] <= STOCH_OS)
    # Alternative lookback (3 candles ago)
    cond2 = (k.iloc[-4] < d.iloc[-4] and k.iloc[-3] > d.iloc[-3] and k.iloc[-3] <= STOCH_OS)
    # Strong upward momentum (current candle index -1 is the last completed? Actually we want the current candle (the forming one) but we only have completed candles.
    # In the doc, "current candle" likely means the last completed one (index -1) and previous two (index -2, -3)
    # We'll use the latest 3 completed candles: -1 (most recent), -2, -3.
    # For momentum, we need: k[-1] > k[-2] + 7 and k[-2] > k[-3] + 7
    cond3 = (k.iloc[-1] > k.iloc[-2] + 7 and k.iloc[-2] > k.iloc[-3] + 7)
    return cond1 or cond2 or cond3

def stochastic_sell_signal(k, d, idx):
    cond1 = (k.iloc[-3] > d.iloc[-3] and k.iloc[-2] < d.iloc[-2] and k.iloc[-2] >= STOCH_OB)
    cond2 = (k.iloc[-4] > d.iloc[-4] and k.iloc[-3] < d.iloc[-3] and k.iloc[-3] >= STOCH_OB)
    cond3 = (k.iloc[-1] < k.iloc[-2] - 7 and k.iloc[-2] < k.iloc[-3] - 7)
    return cond1 or cond2 or cond3

def adx_trending(adx):
    """Check if ADX pre-condition for entry is satisfied.
    adx is a pandas Series (last 2 values enough)."""
    # ADX value for current or previous candle above limit
    cond1 = adx.iloc[-1] > ADX_LIMIT or adx.iloc[-2] > ADX_LIMIT
    # Strong upward movement (>5)
    cond2 = (adx.iloc[-2] - adx.iloc[-3] > 5) or (adx.iloc[-1] - adx.iloc[-2] > 5)
    return cond1 or cond2

def adx_buy_signal(plus_di, minus_di):
    """Check ADX buy signal conditions."""
    # Need at least 4 candles (indices -4 to -1)
    # Condition 1: DI+ trending up, DI- trending down
    cond1 = (plus_di.iloc[-1] > plus_di.iloc[-3] and plus_di.iloc[-2] > plus_di.iloc[-3] and plus_di.iloc[-1] > plus_di.iloc[-2]) and \
            (minus_di.iloc[-1] < minus_di.iloc[-2] and minus_di.iloc[-2] < minus_di.iloc[-3])
    # Condition 2: -DI sustained downtrend and +DI begins to rise
    cond2 = (minus_di.iloc[-4] < minus_di.iloc[-5] and minus_di.iloc[-3] < minus_di.iloc[-4] and minus_di.iloc[-2] < minus_di.iloc[-3] and \
             plus_di.iloc[-2] > plus_di.iloc[-4])
    # Optional DI+ > DI- if flag set
    if ADX_DI_OVER:
        cond1 = cond1 and (plus_di.iloc[-1] > minus_di.iloc[-1])
        cond2 = cond2 and (plus_di.iloc[-2] > minus_di.iloc[-2])
    return cond1 or cond2

def adx_sell_signal(plus_di, minus_di):
    """Check ADX sell signal conditions."""
    cond1 = (minus_di.iloc[-1] > minus_di.iloc[-3] and minus_di.iloc[-2] > minus_di.iloc[-3] and minus_di.iloc[-1] > minus_di.iloc[-2]) and \
            (plus_di.iloc[-1] < plus_di.iloc[-2] and plus_di.iloc[-2] < plus_di.iloc[-3])
    cond2 = (plus_di.iloc[-4] < plus_di.iloc[-5] and plus_di.iloc[-3] < plus_di.iloc[-4] and plus_di.iloc[-2] < plus_di.iloc[-3] and \
             minus_di.iloc[-2] > minus_di.iloc[-4])
    if ADX_DI_OVER:
        cond1 = cond1 and (minus_di.iloc[-1] > plus_di.iloc[-1])
        cond2 = cond2 and (minus_di.iloc[-2] > plus_di.iloc[-2])
    return cond1 or cond2

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

        # We need enough candles for all indicators: EMA, ATR, ADX, Stochastic.
        # We'll fetch at least 50 to be safe.
        candles_needed = max(50, EMA_PERIOD, ATR_PERIOD, ADX_PERIOD, STOCH_K + STOCH_D + STOCH_SLOW + 5)
        df = get_candles(candles_needed)
        if df is None or len(df) < candles_needed:
            print(colorize("Not enough candles yet, waiting...", Colors.YELLOW))
            continue

        # Compute EMA series
        ema_series = compute_ema(df, EMA_PERIOD)
        if ema_series is None:
            print(colorize("Could not compute EMA, waiting...", Colors.YELLOW))
            continue

        # Compute ATR (only needed for entry to set SL/TP)
        atr_value = compute_atr(df, ATR_PERIOD)

        # Compute Stochastic
        stoch_k, stoch_d = compute_stochastic(df, STOCH_K, STOCH_D, STOCH_SLOW)
        # Compute ADX
        adx, plus_di, minus_di = compute_adx(df, ADX_PERIOD)

        # We need at least 6 completed candles for the conditions (indices -1 to -5)
        if stoch_k.isnull().any() or stoch_d.isnull().any() or adx.isnull().any():
            print(colorize("Not enough data for indicators, waiting...", Colors.YELLOW))
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
            # No automatic exit – only SL/TP will close
            pass
        else:
            # Check EMA entry condition
            sell_ema_cond = (prev_candle['open'] > prev_ema and prev_candle['close'] < current_ema and
                              current_price < current_ema - entry_threshold)
            buy_ema_cond = (prev_candle['open'] < prev_ema and prev_candle['close'] > current_ema and
                            current_price > current_ema + entry_threshold)

            # Determine if Stochastic signals are present
            stoch_buy = STOCH_BYPASS or stochastic_buy_signal(stoch_k, stoch_d, -1)
            stoch_sell = STOCH_BYPASS or stochastic_sell_signal(stoch_k, stoch_d, -1)

            # Determine ADX conditions
            adx_trend = adx_trending(adx)
            adx_buy = ADX_BYPASS or (adx_trend and adx_buy_signal(plus_di, minus_di))
            adx_sell = ADX_BYPASS or (adx_trend and adx_sell_signal(plus_di, minus_di))

            if buy_ema_cond and stoch_buy and adx_buy:
                print(colorize("Buy signal confirmed: EMA + Stochastic + ADX", Colors.GREEN))
                if atr_value is None:
                    print(colorize("ATR not available, cannot set SL/TP", Colors.YELLOW))
                else:
                    sl_price = ask - (atr_value * SL_MULT)
                    tp_price = ask + (atr_value * TP_MULT)
                    print(colorize(f"ATR: {atr_value:.5f}, SL: {sl_price:.5f}, TP: {tp_price:.5f}", Colors.CYAN))
                    send_order(mt5.ORDER_TYPE_BUY, VOLUME, ask, sl=sl_price, tp=tp_price, comment=f"Python BUY@{ask}")
            elif sell_ema_cond and stoch_sell and adx_sell:
                print(colorize("Sell signal confirmed: EMA + Stochastic + ADX", Colors.GREEN))
                if atr_value is None:
                    print(colorize("ATR not available, cannot set SL/TP", Colors.YELLOW))
                else:
                    sl_price = bid + (atr_value * SL_MULT)
                    tp_price = bid - (atr_value * TP_MULT)
                    print(colorize(f"ATR: {atr_value:.5f}, SL: {sl_price:.5f}, TP: {tp_price:.5f}", Colors.CYAN))
                    send_order(mt5.ORDER_TYPE_SELL, VOLUME, bid, sl=sl_price, tp=tp_price, comment=f"Python SELL@{bid}")
            else:
                # Optional: log which conditions failed
                if buy_ema_cond and (not stoch_buy or not adx_buy):
                    missing = []
                    if not stoch_buy: missing.append("Stochastic")
                    if not adx_buy: missing.append("ADX")
                    print(colorize(f"EMA buy signal but {', '.join(missing)} condition(s) not met", Colors.YELLOW))
                elif sell_ema_cond and (not stoch_sell or not adx_sell):
                    missing = []
                    if not stoch_sell: missing.append("Stochastic")
                    if not adx_sell: missing.append("ADX")
                    print(colorize(f"EMA sell signal but {', '.join(missing)} condition(s) not met", Colors.YELLOW))

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

        # Also log current Stochastic and ADX values for debugging
        print(f"  Stoch %K={stoch_k.iloc[-1]:.2f} %D={stoch_d.iloc[-1]:.2f} | ADX={adx.iloc[-1]:.2f} +DI={plus_di.iloc[-1]:.2f} -DI={minus_di.iloc[-1]:.2f}")

        time.sleep(0.05)

except KeyboardInterrupt:
    print(colorize("\nScript stopped by user.", Colors.YELLOW))

mt5.shutdown()
