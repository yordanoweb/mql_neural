"""
Live inference loop: loads an ONNX model, polls MT5 for the latest candles,
runs feature engineering, and places buy/sell market orders based on model output.

Exit logic:
  - Hard SL set at order open: ATR(atr_period) * sl_mult
  - No TP sent to broker. An imaginary TP is tracked internally: ATR * tp_mult
  - Once imaginary TP is reached, trailing mode activates:
      BUY  → close on first M1 bearish candle (close < open)
      SELL → close on first M1 bullish candle (close > open)
  - Profit lock: on every new candle at the trading timeframe, SL is moved to
    the previous candle's low (BUY) or high (SELL), but only if that level is
    better (higher for BUY, lower for SELL) than the current SL.
  - Script never opens a new position while one is already open.

Usage:
    python execute_onnx_adx_stoch_vol_on_mt5.py \
        --model onnx/ndx100_m5_12_feat_adx_stoch_vol.onnx \
        --symbol NAS100 --timeframe M5 \
        --window 20 --confidence 0.60 --lot 1.0 --interval 60 \
        --atr_period 14 --sl_mult 1.5 --tp_mult 2.0
"""

import argparse
import time
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np
import onnxruntime as rt
import pandas as pd
import ta

from utils.features import add_price_features, add_adx_features, add_stoch_features, add_volume_features
from utils.colors import Colors, colorize as c

TIMEFRAME_MAP = {
    'M1': 1, 'M5': 5, 'M15': 15, 'M30': 30, 'H1': 16385, 'H4': 16388, 'D1': 16408,
}

FEATURE_COLS = [
    'feat_body', 'feat_range',
    'adx_strength', 'adx_di_signal', 'adx_di_sep', 'adx_momentum', 'adx_regime',
    'stoch_momentum', 'stoch_position', 'stoch_velocity', 'stoch_divergence',
    'vol_ratio', 'vol_momentum', 'vol_price_div', 'vol_percentile', 'vol_zscore',
]


@dataclass
class TradeState:
    """Tracks internal state for the open position."""
    ticket:           int   = 0
    is_buy:           bool  = False
    entry_price:      float = 0.0
    sl_price:         float = 0.0
    tp_target:        float = 0.0   # imaginary TP level
    trailing:         bool  = False # True once imaginary TP has been reached
    last_candle_time: object = None # timestamp of last seen closed candle (profit lock)


_state = TradeState()


def load_session(model_path: str) -> rt.InferenceSession:
    sess    = rt.InferenceSession(model_path)
    outputs = {o.name: o for o in sess.get_outputs()}
    assert 'probabilities' in outputs and outputs['probabilities'].shape[1] == 2, \
        f"Model must have 'probabilities' output [*, 2]. Got: {list(outputs.keys())}"
    return sess


def fetch_candles(symbol: str, tf, n: int) -> pd.DataFrame:
    import MetaTrader5 as mt5
    rates = mt5.copy_rates_from_pos(symbol, tf, 0, n)
    if rates is None or len(rates) < n:
        raise RuntimeError(f"Not enough candles: {mt5.last_error()}")
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    return df


def current_atr(symbol: str, tf: int, atr_period: int) -> float:
    """Return the latest ATR value on the trading timeframe."""
    df = fetch_candles(symbol, tf, atr_period + 50)
    return ta.volatility.AverageTrueRange(
        df['high'], df['low'], df['close'], window=atr_period
    ).average_true_range().iloc[-1]


def last_m1_candle(symbol: str) -> pd.Series:
    df = fetch_candles(symbol, TIMEFRAME_MAP['M1'], 2)
    return df.iloc[-2]   # last *closed* M1 candle


def build_input(df: pd.DataFrame, window: int,
                atr_period: int, adx_period: int, adx_min: float,
                stoch_k: int, stoch_d: int, vol_window: int) -> np.ndarray:
    df = add_price_features(df.copy(), atr_period)
    df = add_adx_features(df, adx_period, adx_min)
    df = add_stoch_features(df, stoch_k, stoch_d)
    df = add_volume_features(df, vol_window)
    df.dropna(inplace=True)
    if len(df) < window:
        raise RuntimeError("Not enough clean rows after feature computation")
    arr = df[FEATURE_COLS].values[-window:].flatten().astype(np.float32)
    return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0).reshape(1, -1)


def get_open_position(symbol: str):
    import MetaTrader5 as mt5
    positions = mt5.positions_get(symbol=symbol)
    return positions[0] if positions else None


def close_position(pos, lot: float, reason: str) -> None:
    import MetaTrader5 as mt5
    direction = mt5.ORDER_TYPE_SELL if pos.type == mt5.POSITION_TYPE_BUY else mt5.ORDER_TYPE_BUY
    tick      = mt5.symbol_info_tick(pos.symbol)
    price     = tick.bid if direction == mt5.ORDER_TYPE_SELL else tick.ask
    req = {
        'action':       mt5.TRADE_ACTION_DEAL,
        'symbol':       pos.symbol,
        'volume':       lot,
        'type':         direction,
        'position':     pos.ticket,
        'price':        price,
        'deviation':    20,
        'magic':        0,
        'comment':      f'onnx_{reason}',
        'type_time':    mt5.ORDER_TIME_GTC,
        'type_filling': mt5.ORDER_FILLING_IOC,
    }
    result = mt5.order_send(req)
    print(c(f"  → CLOSED ({reason}): retcode={result.retcode}", Colors.MAGENTA))
    # reset state
    global _state
    _state = TradeState()


def open_position(symbol: str, is_buy: bool, lot: float, tf: int,
                  atr_period: int, sl_mult: float, tp_mult: float) -> None:
    import MetaTrader5 as mt5
    global _state
    tick  = mt5.symbol_info_tick(symbol)
    price = tick.ask if is_buy else tick.bid
    atr   = current_atr(symbol, tf, atr_period)
    sl    = price - atr * sl_mult if is_buy else price + atr * sl_mult
    tp_target = price + atr * tp_mult if is_buy else price - atr * tp_mult

    order_type = mt5.ORDER_TYPE_BUY if is_buy else mt5.ORDER_TYPE_SELL
    req = {
        'action':       mt5.TRADE_ACTION_DEAL,
        'symbol':       symbol,
        'volume':       lot,
        'type':         order_type,
        'price':        price,
        'sl':           sl,
        'deviation':    20,
        'magic':        0,
        'comment':      f'F16_{"B" if is_buy else "S"}@{price}',
        'type_time':    mt5.ORDER_TIME_GTC,
        'type_filling': mt5.ORDER_FILLING_IOC,
    }
    result = mt5.order_send(req)
    label  = 'BUY' if is_buy else 'SELL'
    color  = Colors.GREEN if is_buy else Colors.RED
    print(c(f"  → {label} opened: retcode={result.retcode}  SL={sl:.5f}  iTP={tp_target:.5f}", color))

    if result.retcode == 10009:   # TRADE_RETCODE_DONE
        _state = TradeState(
            ticket=result.order,
            is_buy=is_buy,
            entry_price=price,
            sl_price=sl,
            tp_target=tp_target,
            trailing=False,
        )


def print_state(pos, current_price: float, lot: float) -> None:
    """Print current trade state to stdout."""
    import MetaTrader5 as mt5
    now = datetime.now().strftime('%H:%M:%S')
    if not pos:
        print(c(f"[{now}] FLAT — no open position", Colors.CYAN))
        return
    direction = c('BUY',  Colors.GREEN) if _state.is_buy else c('SELL', Colors.RED)
    trailing  = c('TRAILING', Colors.MAGENTA) if _state.trailing else c('HOLDING', Colors.YELLOW)
    pnl_pts   = (current_price - _state.entry_price) * (1 if _state.is_buy else -1)
    info      = mt5.symbol_info(pos.symbol)
    tick_value = info.trade_tick_value if info else 1.0
    tick_size  = info.trade_tick_size  if info else 1.0
    pnl_money  = pnl_pts / tick_size * tick_value * lot
    pnl_color  = Colors.GREEN if pnl_money >= 0 else Colors.RED
    print(
        f"[{now}] {direction} | {trailing} | "
        f"entry={c(f'{_state.entry_price:.5f}', Colors.WHITE)} "
        f"price={c(f'{current_price:.5f}', Colors.WHITE)} "
        f"PnL={c(f'${pnl_money:+.2f}', pnl_color)} | "
        f"SL={c(f'{_state.sl_price:.5f}', Colors.RED)} "
        f"iTP={c(f'{_state.tp_target:.5f}', Colors.GREEN)}"
    )


def move_sl_to_previous_candle(pos, tf: int) -> None:
    """Move SL to previous candle's low (BUY) or high (SELL) on each new candle."""
    import MetaTrader5 as mt5
    global _state

    candles = fetch_candles(pos.symbol, tf, 3)
    last_closed = candles.iloc[-2]   # index -1 is the forming candle
    candle_time = last_closed['time']

    if candle_time == _state.last_candle_time:
        return   # same candle, nothing to do

    _state.last_candle_time = candle_time
    new_sl = float(last_closed['low']) if _state.is_buy else float(last_closed['high'])

    # only move SL if it improves (locks more profit)
    if (_state.is_buy and new_sl <= _state.sl_price) or \
       (not _state.is_buy and new_sl >= _state.sl_price):
        return

    req = {
        'action':   mt5.TRADE_ACTION_SLTP,
        'symbol':   pos.symbol,
        'position': pos.ticket,
        'sl':       new_sl,
        'tp':       0.0,
    }
    result = mt5.order_send(req)
    if result.retcode == 10009:
        print(c(f"  → SL moved to {new_sl:.5f} (prev candle {'low' if _state.is_buy else 'high'})", Colors.MAGENTA))
        _state.sl_price = new_sl
    else:
        print(c(f"  → SL move failed: retcode={result.retcode}", Colors.RED))


def manage_open_trade(pos, lot: float, tf: int) -> None:
    """Check imaginary TP, profit-lock SL, and trailing exit on every cycle."""
    import MetaTrader5 as mt5
    global _state

    tick          = mt5.symbol_info_tick(pos.symbol)
    current_price = tick.bid if _state.is_buy else tick.ask

    # profit lock: move SL to previous candle on each new candle
    move_sl_to_previous_candle(pos, tf)

    # activate trailing once imaginary TP is reached
    if not _state.trailing:
        tp_hit = (current_price >= _state.tp_target) if _state.is_buy \
                 else (current_price <= _state.tp_target)
        if tp_hit:
            print(c(f"  → imaginary TP reached ({current_price:.5f}), trailing mode ON", Colors.MAGENTA))
            _state.trailing = True

    if _state.trailing:
        candle = last_m1_candle(pos.symbol)
        bearish = candle['close'] < candle['open']
        bullish = candle['close'] > candle['open']
        if (_state.is_buy and bearish) or (not _state.is_buy and bullish):
            close_position(pos, lot, reason='trailing_exit')


def run(args):
    import MetaTrader5 as mt5
    if not mt5.initialize():
        raise RuntimeError(f"MT5 init failed: {mt5.last_error()}")

    sess     = load_session(args.model)
    inp_name = sess.get_inputs()[0].name
    tf       = TIMEFRAME_MAP[args.timeframe.upper()]
    n_candles = args.window + 100

    print(f"Symbol={args.symbol}  TF={args.timeframe}  interval={args.interval}s")
    print(f"Model={args.model}  confidence={args.confidence}")
    print(f"SL={args.sl_mult}×ATR  imaginary_TP={args.tp_mult}×ATR  ATR_period={args.atr_period}")

    try:
        while True:
            now = datetime.now().strftime('%H:%M:%S')
            pos = get_open_position(args.symbol)

            if pos:
                tick          = mt5.symbol_info_tick(args.symbol)
                current_price = tick.bid if _state.is_buy else tick.ask
                print_state(pos, current_price, args.lot)
                manage_open_trade(pos, args.lot, tf)
            else:
                print(c(f"[{now}] FLAT — running inference...", Colors.CYAN))
                df = fetch_candles(args.symbol, tf, n_candles)
                X  = build_input(df, args.window, args.atr_period, args.adx_period,
                                 args.adx_min, args.stoch_k, args.stoch_d, args.vol_window)
                results       = {o.name: v for o, v in zip(sess.get_outputs(), sess.run(None, {inp_name: X}))}
                p_sell, p_buy = results['probabilities'][0]

                p_buy_str  = c(f'P(buy)={p_buy:.3f}',  Colors.GREEN if p_buy  >= args.confidence else Colors.WHITE)
                p_sell_str = c(f'P(sell)={p_sell:.3f}', Colors.RED   if p_sell >= args.confidence else Colors.WHITE)
                print(f"  {p_buy_str}  {p_sell_str}", end='')

                if p_buy >= args.confidence:
                    print(c('  → BUY signal', Colors.GREEN))
                    open_position(args.symbol, is_buy=True, lot=args.lot, tf=tf,
                                  atr_period=args.atr_period,
                                  sl_mult=args.sl_mult, tp_mult=args.tp_mult)
                elif p_sell >= args.confidence:
                    print(c('  → SELL signal', Colors.RED))
                    open_position(args.symbol, is_buy=False, lot=args.lot, tf=tf,
                                  atr_period=args.atr_period,
                                  sl_mult=args.sl_mult, tp_mult=args.tp_mult)
                else:
                    print(c('  → no signal', Colors.YELLOW))

            time.sleep(args.interval)

    except KeyboardInterrupt:
        print(c("\nStopped.", Colors.YELLOW))
    finally:
        mt5.shutdown()


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--model',      required=True,  help='Path to ONNX model')
    parser.add_argument('--symbol',     required=True,  help='MT5 symbol, e.g. NAS100')
    parser.add_argument('--timeframe',  required=True,  help='M1 M5 M15 M30 H1 H4 D1')
    parser.add_argument('--window',     type=int,   default=20,   help='Window size (must match training)')
    parser.add_argument('--confidence', type=float, default=0.60, help='Min probability to open a trade')
    parser.add_argument('--lot',        type=float, default=1.0,  help='Order lot size')
    parser.add_argument('--interval',   type=int,   default=60,   help='Seconds between cycles')
    parser.add_argument('--atr_period', type=int,   default=14,   help='ATR period for SL/TP calculation')
    parser.add_argument('--sl_mult',    type=float, default=1.5,  help='SL = ATR * sl_mult')
    parser.add_argument('--tp_mult',    type=float, default=2.0,  help='Imaginary TP = ATR * tp_mult')
    parser.add_argument('--adx_period', type=int,   default=8,    help='ADX indicator period')
    parser.add_argument('--adx_min',    type=float, default=20.0, help='ADX minimum threshold')
    parser.add_argument('--stoch_k',    type=int,   default=14,   help='Stochastic K period')
    parser.add_argument('--stoch_d',    type=int,   default=3,    help='Stochastic D period')
    parser.add_argument('--vol_window', type=int,   default=10,   help='Volume rolling window')
    args = parser.parse_args()
    run(args)


if __name__ == '__main__':
    main()
