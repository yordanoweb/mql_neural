"""
Live inference loop: loads an ONNX model, polls MT5 for the latest candles,
runs feature engineering, and places buy/sell market orders based on model output.

Exit logic:
  - Hard SL set at order open: ATR(atr_period) * sl_mult
  - No TP sent to broker. An imaginary TP is tracked internally: ATR * tp_mult
  - Once imaginary TP is reached, trailing mode activates:
      BUY  → close on first opposite candle one timeframe below (e.g. M5→M1, M15→M5)
      SELL → close on first opposite candle one timeframe below
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
import csv
import os
import time
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np
import onnxruntime as rt
import pandas as pd
import ta

from utils.features import add_price_features, add_adx_features, add_stoch_features, add_volume_features
from utils.colors import Colors, colorize as c
from utils.telegram import notify
from utils.sound import play_sound

TIMEFRAME_MAP = {
    'M1': 1, 'M5': 5, 'M15': 15, 'M30': 30, 'H1': 16385, 'H4': 16388, 'D1': 16408,
}

# Reverse mapping from MT5 timeframe integer to string key
TIMEFRAME_REVERSE_MAP = {v: k for k, v in TIMEFRAME_MAP.items()}

# One step below the trading timeframe
TRAILING_TF = {'M5': 'M1', 'M15': 'M5', 'M30': 'M15', 'H1': 'M30', 'H4': 'H1', 'D1': 'H4'}

# Profit lock timeframe: same mapping as trailing, but M15 uses M1 instead of M5
PROFIT_LOCK_TF = {'M5': 'M1', 'M15': 'M1', 'M30': 'M15', 'H1': 'M30', 'H4': 'H1', 'D1': 'H4'}

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
    lot_size:         float = 0.0   # actual lot size used for the trade
    trailing:         bool  = False # True once imaginary TP has been reached
    last_candle_time: object = None # timestamp of last seen closed candle (profit lock)


_state = TradeState()

# Global state for inference tracking
_last_signal = None  # 'BUY', 'SELL', or None
_consecutive_signal_count = 0
_last_trade_time = None  # datetime of last trade open/close


@dataclass
class InferenceStats:
    """Running min/max of raw model probabilities since script start."""
    max_buy:  float = field(default=None)
    min_buy:  float = field(default=None)
    max_sell: float = field(default=None)
    min_sell: float = field(default=None)
    max_hold: float = field(default=None)
    min_hold: float = field(default=None)
    count:    int   = 0

    def update(self, p_buy: float, p_sell: float, p_hold: float) -> None:
        self.count    += 1
        self.max_buy   = p_buy  if self.max_buy  is None else max(self.max_buy,  p_buy)
        self.min_buy   = p_buy  if self.min_buy  is None else min(self.min_buy,  p_buy)
        self.max_sell  = p_sell if self.max_sell is None else max(self.max_sell, p_sell)
        self.min_sell  = p_sell if self.min_sell is None else min(self.min_sell, p_sell)
        self.max_hold  = p_hold if self.max_hold is None else max(self.max_hold, p_hold)
        self.min_hold  = p_hold if self.min_hold is None else min(self.min_hold, p_hold)

    def __str__(self) -> str:
        if self.count == 0:
            return "  stats: no data yet"
        return (
            f"  stats({self.count}): "
            f"buy=[{self.min_buy:.3f}–{self.max_buy:.3f}]  "
            f"sell=[{self.min_sell:.3f}–{self.max_sell:.3f}]  "
            f"hold=[{self.min_hold:.3f}–{self.max_hold:.3f}]"
        )


_stats = InferenceStats()

# Paths relative to project root
_LOG_FILE   = os.path.join('log', 'trading_log.csv')
_INFERENCE_LOG_FILE = os.path.join('log', 'inference_log.csv')

_LOG_FIELDS = ['timestamp', 'event', 'symbol', 'direction', 'price',
               'sl', 'tp_target', 'atr', 'confidence', 'pnl_pts', 'reason']

def _log(symbol: str, event: str, **kwargs) -> None:
    exists = os.path.exists(_LOG_FILE)
    # Ensure log directory exists
    log_dir = os.path.dirname(_LOG_FILE)
    os.makedirs(log_dir, exist_ok=True)
    
    with open(_LOG_FILE, 'a', newline='') as f:
        w = csv.DictWriter(f, fieldnames=_LOG_FIELDS)
        if not exists:
            w.writeheader()
        row = {k: '' for k in _LOG_FIELDS}
        row.update({'timestamp': datetime.now().isoformat(), 'event': event, 'symbol': symbol})
        row.update(kwargs)
        w.writerow(row)


_INFERENCE_LOG_FIELDS = [
    'timestamp', 'symbol', 'timeframe', 'p_buy', 'p_sell', 'p_hold',
    'close_price', 'ema_value', 'ema_distance_pct', 'ema_filter_passed', 'ema_distance_filter_passed',
    'confidence_threshold', 'signal_decision', 'signal_reason', 'signal_strength',
    'atr_value', 'volume', 'account_balance', 'has_open_position',
    'consecutive_signal_count', 'time_since_last_trade_sec'
]

def _log_inference(symbol: str, timeframe: str, p_buy: float, p_sell: float, p_hold: float,
                   close_price: float, ema_value: float, ema_distance_pct: float,
                   ema_filter_passed: bool, ema_distance_filter_passed: bool,
                   confidence_threshold: float, signal_decision: str, signal_reason: str,
                   atr_value: float, volume: float, account_balance: float,
                   has_open_position: bool, consecutive_signal_count: int = 0,
                   time_since_last_trade_sec: int = 0) -> None:
    """Log inference statistics for later analysis."""
    signal_strength = max(p_buy, p_sell, p_hold) - sorted([p_buy, p_sell, p_hold])[1]
    
    # Ensure log directory exists
    log_dir = os.path.dirname(_INFERENCE_LOG_FILE)
    os.makedirs(log_dir, exist_ok=True)
    
    exists = os.path.exists(_INFERENCE_LOG_FILE)
    with open(_INFERENCE_LOG_FILE, 'a', newline='') as f:
        w = csv.DictWriter(f, fieldnames=_INFERENCE_LOG_FIELDS)
        if not exists:
            w.writeheader()
        row = {
            'timestamp': datetime.now().isoformat(),
            'symbol': symbol,
            'timeframe': timeframe,
            'p_buy': p_buy,
            'p_sell': p_sell,
            'p_hold': p_hold,
            'close_price': close_price,
            'ema_value': ema_value,
            'ema_distance_pct': ema_distance_pct,
            'ema_filter_passed': ema_filter_passed,
            'ema_distance_filter_passed': ema_distance_filter_passed,
            'confidence_threshold': confidence_threshold,
            'signal_decision': signal_decision,
            'signal_reason': signal_reason,
            'signal_strength': signal_strength,
            'atr_value': atr_value,
            'volume': volume,
            'account_balance': account_balance,
            'has_open_position': has_open_position,
            'consecutive_signal_count': consecutive_signal_count,
            'time_since_last_trade_sec': time_since_last_trade_sec
        }
        w.writerow(row)


def _pnl_usd(symbol: str, pnl_pts: float, lot: float) -> float:
    import MetaTrader5 as mt5
    info = mt5.symbol_info(symbol)
    if info is None:
        return 0.0
    return pnl_pts / info.trade_tick_size * info.trade_tick_value * lot


def _daily_loss(symbol: str) -> float:
    """Sum of today's realized losses (negative number) for *symbol* from MT5 deal history."""
    import MetaTrader5 as mt5
    from datetime import date
    today_start = int(datetime.combine(date.today(), datetime.min.time()).timestamp())
    deals = mt5.history_deals_get(today_start, int(time.time()) + 1)
    if not deals:
        return 0.0
    return sum(d.profit for d in deals if d.symbol == symbol and d.profit < 0)


def usd_to_lot(symbol: str, usd_amount: float) -> float:
    """Convert account currency amount to lot size."""
    import MetaTrader5 as mt5
    tick = mt5.symbol_info_tick(symbol)
    margin_per_lot = mt5.order_calc_margin(mt5.ORDER_TYPE_BUY, symbol, 1.0, tick.ask)
    if not margin_per_lot or margin_per_lot <= 0:
        return 1.0  # fallback
    
    lot = usd_amount / margin_per_lot
    
    info = mt5.symbol_info(symbol)
    step = info.volume_step
    lot = step * round(lot / step)
    
    # Check if calculated lot is below minimum
    if lot < info.volume_min:
        print(c(f"  Warning: ${usd_amount:.2f} translates to {usd_amount/margin_per_lot:.4f} lots, "
                f"but minimum is {info.volume_min} lots (step={step})", Colors.YELLOW))
        lot = info.volume_min
    
    return max(info.volume_min, min(info.volume_max, lot))


def calc_lot(symbol: str, magic: int, max_risk: float, decrease_factor: float, fallback_lot: float) -> float:
    """Dynamic lot sizing: risk-based + consecutive-loss reduction."""
    import MetaTrader5 as mt5
    tick   = mt5.symbol_info_tick(symbol)
    margin = mt5.order_calc_margin(mt5.ORDER_TYPE_BUY, symbol, 1.0, tick.ask)
    if not margin:
        return fallback_lot
    account = mt5.account_info()
    lot     = round(account.margin_free * max_risk / margin, 2)

    if decrease_factor > 0:
        deals = mt5.history_deals_get(0, int(time.time()) + 1)
        losses = 0
        if deals:
            for deal in reversed(deals):
                if deal.symbol != symbol or deal.magic != magic:
                    continue
                if deal.profit > 0:
                    break
                if deal.profit < 0:
                    losses += 1
        if losses > 1:
            lot = round(lot - lot * losses / decrease_factor, 1)

    info = mt5.symbol_info(symbol)
    step = info.volume_step
    lot  = step * round(lot / step)
    return max(info.volume_min, min(info.volume_max, lot))


def load_session(model_path: str) -> rt.InferenceSession:
    sess    = rt.InferenceSession(model_path)
    outputs = {o.name: o for o in sess.get_outputs()}
    assert 'probabilities' in outputs and outputs['probabilities'].shape[1] == 3, \
        f"Model must have 'probabilities' output [*, 3]. Got: {list(outputs.keys())}"
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
    """Return the latest ATR value on the given timeframe."""
    df = fetch_candles(symbol, tf, atr_period * 3)
    atr = ta.volatility.AverageTrueRange(df['high'], df['low'], df['close'], window=atr_period)
    return float(atr.average_true_range().iloc[-1])


def compute_atr(symbol: str, tf: int, atr_period: int) -> float:
    """Return the latest ATR value (alias for current_atr)."""
    return current_atr(symbol, tf, atr_period)


def last_trailing_candle(symbol: str, trailing_tf: int) -> pd.Series:
    df = fetch_candles(symbol, trailing_tf, 2)
    return df.iloc[-2]   # last *closed* candle on trailing timeframe


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


def close_position(pos, reason: str, deviation: int, magic: int, dry_run: bool = False) -> None:
    import MetaTrader5 as mt5
    global _state, _last_trade_time
    direction = mt5.ORDER_TYPE_SELL if pos.type == mt5.POSITION_TYPE_BUY else mt5.ORDER_TYPE_BUY
    tick      = mt5.symbol_info_tick(pos.symbol)
    price     = tick.bid if direction == mt5.ORDER_TYPE_SELL else tick.ask
    
    if dry_run:
        print(c(f"  → WOULD_CLOSE ({reason}): {pos.symbol} at {price:.5f}", Colors.MAGENTA))
        return
    
    req = {
        'action':       mt5.TRADE_ACTION_DEAL,
        'symbol':       pos.symbol,
        'volume':       _state.lot_size,
        'type':         direction,
        'position':     pos.ticket,
        'price':        price,
        'deviation':    deviation,
        'magic':        magic,
        'comment':      f'onnx_{reason}',
        'type_time':    mt5.ORDER_TIME_GTC,
        'type_filling': mt5.ORDER_FILLING_IOC,
    }
    result = mt5.order_send(req)
    print(c(f"  → CLOSED ({reason}): retcode={result.retcode}", Colors.MAGENTA))
    # reset state
    pnl_pts = round((price - _state.entry_price) * (1 if _state.is_buy else -1), 5)
    pnl_usd = _pnl_usd(pos.symbol, pnl_pts, _state.lot_size)
    balance = mt5.account_info().balance
    _log(pos.symbol, 'CLOSE', direction='BUY' if _state.is_buy else 'SELL',
         price=price, sl=_state.sl_price, tp_target=_state.tp_target,
         pnl_pts=pnl_pts, reason=reason)
    direction = 'BUY' if _state.is_buy else 'SELL'
    notify(f"{'🟢' if _state.is_buy else '🔴'} {direction} CLOSED — {pos.symbol}\n"
           f"💵 Price:  {price:.5f}\n"
           f"{'✅' if pnl_usd >= 0 else '🔻'} PnL:    {pnl_usd:+.2f} USD\n"
           f"{'📈' if pnl_usd >= 0 else '📉'} Balance: {balance:.2f} USD\n"
           f"🛑 Reason: {reason}")
    # Play alert sound for trade close
    play_sound("alert" if pnl_usd >= 0 else "error")
    _state = TradeState()
    # Update last trade time
    _last_trade_time = datetime.now()


def open_position(symbol: str, is_buy: bool, lot: float, tf: int,
                  atr_period: int, sl_mult: float, tp_mult: float,
                  deviation: int, magic: int, confidence: float = 0.0,
                  lot_usd: float = 0.0, dry_run: bool = False) -> None:
    import MetaTrader5 as mt5
    global _state
    if lot_usd > 0:
        lot = usd_to_lot(symbol, lot_usd)
    tick  = mt5.symbol_info_tick(symbol)
    price = tick.ask if is_buy else tick.bid
    atr   = current_atr(symbol, tf, atr_period)
    sl    = price - atr * sl_mult if is_buy else price + atr * sl_mult
    tp_target = price + atr * tp_mult if is_buy else price - atr * tp_mult

    label  = 'BUY' if is_buy else 'SELL'
    color  = Colors.GREEN if is_buy else Colors.RED
    
    if dry_run:
        print(c(f"  → WOULD_{label}: price={price:.5f}  SL={sl:.5f}  iTP={tp_target:.5f}", color))
        return

    order_type = mt5.ORDER_TYPE_BUY if is_buy else mt5.ORDER_TYPE_SELL
    req = {
        'action':       mt5.TRADE_ACTION_DEAL,
        'symbol':       symbol,
        'volume':       lot,
        'type':         order_type,
        'price':        price,
        'sl':           sl,
        'deviation':    deviation,
        'magic':        magic,
        'comment':      f'F16_{"B" if is_buy else "S"}@{confidence:.3f}',
        'type_time':    mt5.ORDER_TIME_GTC,
        'type_filling': mt5.ORDER_FILLING_IOC,
    }
    result = mt5.order_send(req)
    print(c(f"  → {label} opened: retcode={result.retcode}  SL={sl:.5f}  iTP={tp_target:.5f}", color))

    if result.retcode == 10009:   # TRADE_RETCODE_DONE
        _state = TradeState(
            ticket=result.order,
            is_buy=is_buy,
            entry_price=price,
            sl_price=sl,
            tp_target=tp_target,
            lot_size=lot,           # store actual lot size
            trailing=False,
        )
        _log(symbol, 'OPEN', direction='BUY' if is_buy else 'SELL',
             price=price, sl=sl, tp_target=tp_target, atr=round(atr, 5), confidence=confidence)
        notify(f"{'🟢' if is_buy else '🔴'} {'BUY' if is_buy else 'SELL'} OPENED — {symbol}\n"
               f"💵 Price:      {price:.5f}\n"
               f"🛡 SL:         {sl:.5f}\n"
               f"🎯 iTP:        {tp_target:.5f}\n"
               f"📈 Confidence: {confidence:.3f}")
        # Play alert sound for trade open
        play_sound("alert")


def print_state(pos, current_price: float, symbol: str) -> None:
    """Print current trade state to stdout."""
    import MetaTrader5 as mt5
    now = datetime.now().strftime('%H:%M:%S')
    if not pos:
        print(c(f"[{now}] {symbol} FLAT — no open position", Colors.CYAN))
        return
    direction = c('BUY',  Colors.GREEN) if _state.is_buy else c('SELL', Colors.RED)
    trailing  = c('TRAILING', Colors.MAGENTA) if _state.trailing else c('HOLDING', Colors.YELLOW)
    pnl_pts   = (current_price - _state.entry_price) * (1 if _state.is_buy else -1)
    info      = mt5.symbol_info(pos.symbol)
    tick_value = info.trade_tick_value if info else 1.0
    tick_size  = info.trade_tick_size  if info else 1.0
    pnl_money  = pnl_pts / tick_size * tick_value * _state.lot_size
    pnl_color  = Colors.GREEN if pnl_money >= 0 else Colors.RED
    print(
        f"[{now}] {symbol} {direction} | {trailing} | "
        f"entry={c(f'{_state.entry_price:.5f}', Colors.WHITE)} "
        f"price={c(f'{current_price:.5f}', Colors.WHITE)} "
        f"PnL={c(f'${pnl_money:+.2f}', pnl_color)} | "
        f"SL={c(f'{_state.sl_price:.5f}', Colors.RED)} "
        f"iTP={c(f'{_state.tp_target:.5f}', Colors.GREEN)}"
    )


def move_sl_to_previous_candle(pos, tf: int, profit_lock_tf: int) -> None:
    """Move SL to previous candle's low (BUY) or high (SELL) on each new candle."""
    import MetaTrader5 as mt5
    global _state

    candles = fetch_candles(pos.symbol, profit_lock_tf, 3)
    last_closed = candles.iloc[-2]   # index -1 is the forming candle
    candle_time = last_closed['time']

    if candle_time == _state.last_candle_time:
        return   # same candle, nothing to do

    # only ratchet once price is in profit
    import MetaTrader5 as mt5
    tick = mt5.symbol_info_tick(pos.symbol)
    current_price = tick.bid if _state.is_buy else tick.ask
    in_profit = (current_price > _state.entry_price) if _state.is_buy else (current_price < _state.entry_price)
    if not in_profit:
        return

    _state.last_candle_time = candle_time
    new_sl = float(last_closed['low']) if _state.is_buy else float(last_closed['high'])

    # new SL must itself be in profit territory (past entry)
    if (_state.is_buy and new_sl <= _state.entry_price) or \
       (not _state.is_buy and new_sl >= _state.entry_price):
        return

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
        print(c(f"  → SL moved to {new_sl:.5f} (prev {profit_lock_tf} candle {'low' if _state.is_buy else 'high'})", Colors.MAGENTA))
        _state.sl_price = new_sl
        # Play alert sound for profit lock SL move
        play_sound("alert")
    else:
        print(c(f"  → SL move failed: retcode={result.retcode}", Colors.RED))


def manage_open_trade(pos, tf: int, trailing_tf: int, deviation: int, magic: int, atr_period: int, sl_mult: float, dry_run: bool = False) -> None:
    """Check imaginary TP, profit-lock SL, and trailing exit on every cycle."""
    import MetaTrader5 as mt5
    global _state

    tick          = mt5.symbol_info_tick(pos.symbol)
    current_price = tick.bid if _state.is_buy else tick.ask

    # breakeven: move SL to entry when profit reaches 0.5× ATR
    if not _state.trailing:
        atr = compute_atr(pos.symbol, tf, atr_period)
        profit_pts = (current_price - _state.entry_price) if _state.is_buy else (_state.entry_price - current_price)
        profit_atr = profit_pts / atr if atr > 0 else 0
        if profit_atr >= 0.5 and _state.sl_price != _state.entry_price:
            # Check if SL is not already at or better than entry
            if (_state.is_buy and _state.sl_price < _state.entry_price) or \
               (not _state.is_buy and _state.sl_price > _state.entry_price):
                req = {
                    'action':   mt5.TRADE_ACTION_SLTP,
                    'symbol':   pos.symbol,
                    'position': pos.ticket,
                    'sl':       _state.entry_price,
                    'tp':       0.0,
                }
                result = mt5.order_send(req)
                if result.retcode == 10009:
                    print(c(f"  → SL moved to breakeven ({_state.entry_price:.5f}) at {profit_atr:.1f}×ATR profit", Colors.MAGENTA))
                    _state.sl_price = _state.entry_price
                    # Play alert sound for breakeven SL move
                    play_sound("alert")
                else:
                    print(c(f"  → breakeven move failed: retcode={result.retcode}", Colors.RED))

    # profit lock: move SL to previous candle on each new candle (using profit lock TF)
    tf_key = TIMEFRAME_REVERSE_MAP.get(tf, 'M1')
    profit_lock_tf_key = PROFIT_LOCK_TF.get(tf_key, 'M1')
    profit_lock_tf = TIMEFRAME_MAP.get(profit_lock_tf_key, tf)  # fallback to trading TF if no mapping
    move_sl_to_previous_candle(pos, tf, profit_lock_tf)

    # activate trailing once imaginary TP is reached
    if not _state.trailing:
        tp_hit = (current_price >= _state.tp_target) if _state.is_buy \
                 else (current_price <= _state.tp_target)
        if tp_hit:
            print(c(f"  → imaginary TP reached ({current_price:.5f}), trailing mode ON", Colors.MAGENTA))
            _state.trailing = True
            # Play alert sound for TP hit
            play_sound("alert")

    if _state.trailing:
        candle = last_trailing_candle(pos.symbol, trailing_tf)
        bearish = candle['close'] < candle['open']
        bullish = candle['close'] > candle['open']
        if (_state.is_buy and bearish) or (not _state.is_buy and bullish):
            close_position(pos, reason='trailing_exit', deviation=deviation, magic=magic, dry_run=dry_run)


def run(args):
    import MetaTrader5 as mt5
    if not mt5.initialize():
        raise RuntimeError(f"MT5 init failed: {mt5.last_error()}")

    sess     = load_session(args.model)
    inp_name = sess.get_inputs()[0].name
    tf       = TIMEFRAME_MAP[args.timeframe.upper()]
    tf_key   = args.timeframe.upper()
    trailing_tf_key = TRAILING_TF.get(tf_key, 'M1')
    trailing_tf     = TIMEFRAME_MAP[trailing_tf_key]
    n_candles = args.window + args.atr_period + args.adx_period + args.stoch_k + args.vol_window + 10

    print(f"Symbol={args.symbol}  TF={args.timeframe}  trailing_TF={trailing_tf_key}  interval={args.interval}s")
    print(f"Model={args.model}  confidence={args.confidence}")
    print(f"SL={args.sl_mult}×ATR  imaginary_TP={args.tp_mult}×ATR  ATR_period={args.atr_period}")
    print(f"EMA_filter={args.ema_period}  EMA_distance={args.ema_distance}%  magic={args.magic}  deviation={args.deviation}")
    if args.max_daily_loss > 0:
        print(f"Max_daily_loss={args.max_daily_loss} USD")
    balance = mt5.account_info().balance
    daily_loss_line = f"\n📉 Max daily loss: {args.max_daily_loss:.2f} USD" if args.max_daily_loss > 0 else ""
    if args.lot_usd > 0:
        # Calculate equivalent lot size
        calculated_lot = usd_to_lot(args.symbol, args.lot_usd)
        lot_info = f"📊 USD: {args.lot_usd:.2f} (Lot: {calculated_lot:.2f})"
    else:
        lot_info = f"📊 Lot: {args.lot}"
    notify(f"🚀 Bot started\n"
           f"📊 {args.symbol} ({args.timeframe})\n"
           f"💰 Balance: {balance:.2f} USD\n"
           f"{lot_info}\n"
           f"🎯 Confidence: {args.confidence:.2f}\n"
           f"🛡 SL: {args.sl_mult}×ATR\n"
           f"🎯 TP: {args.tp_mult}×ATR\n"
           f"📈 EMA: {args.ema_period} ({args.ema_distance:.3f}% distance)\n"
           f"⏱ Interval: {args.interval}s\n"
           f"🔢 Magic: {args.magic}\n"
           f"📁 {os.path.basename(args.model)}"
           f"{daily_loss_line}")
    # Play success sound for bot start
    play_sound("success")

    try:
        while True:
            now = datetime.now().strftime('%H:%M:%S')
            pos = get_open_position(args.symbol)

            if pos:
                tick          = mt5.symbol_info_tick(args.symbol)
                current_price = tick.bid if _state.is_buy else tick.ask
                print_state(pos, current_price, args.symbol)
                had_ticket = _state.ticket
                manage_open_trade(pos, tf, trailing_tf, args.deviation, args.magic, args.atr_period, args.sl_mult, args.dry_run)
                # daily loss check after trailing exit
                if had_ticket and _state.ticket == 0 and args.max_daily_loss > 0:
                    loss = _daily_loss(args.symbol)
                    if abs(loss) >= args.max_daily_loss:
                        msg = f"Daily loss limit reached: {loss:+.2f} USD (limit: -{args.max_daily_loss:.2f})"
                        print(c(f"  🛑 {msg}", Colors.RED))
                        notify(f"🛑 {msg} — {args.symbol}\nBot stopped.")
                        # Play error sound for daily loss limit
                        play_sound("error")
                        break
            else:
                # detect broker-closed position (SL hit)
                if _state.ticket != 0:
                    tick = mt5.symbol_info_tick(args.symbol)
                    close_price = tick.bid if _state.is_buy else tick.ask
                    pnl_pts = round((close_price - _state.entry_price) * (1 if _state.is_buy else -1), 5)
                    pnl_usd = _pnl_usd(args.symbol, pnl_pts, _state.lot_size)
                    balance = mt5.account_info().balance
                    print(c(f"[{now}] {args.symbol} position closed by broker (SL hit), PnL={pnl_usd:+.2f} USD", Colors.RED))
                    _log(args.symbol, 'CLOSE', direction='BUY' if _state.is_buy else 'SELL',
                         price=close_price, sl=_state.sl_price, tp_target=_state.tp_target,
                         pnl_pts=pnl_pts, reason='sl_hit')
                    direction = 'BUY' if _state.is_buy else 'SELL'
                    notify(f"{'🟢' if _state.is_buy else '🔴'} {direction} CLOSED — {args.symbol}\n"
                           f"💵 Price:  {close_price:.5f}\n"
                           f"{'✅' if pnl_usd >= 0 else '🔻'} PnL:    {pnl_usd:+.2f} USD\n"
                           f"{'📈' if pnl_usd >= 0 else '📉'} Balance: {balance:.2f} USD\n"
                           f"🛑 Reason: SL hit")
                    # Play alert sound for SL hit trade close
                    play_sound("alert" if pnl_usd >= 0 else "error")
                    _state.__init__()
                    # Update last trade time
                    global _last_trade_time
                    _last_trade_time = datetime.now()
                    # daily loss check after SL hit
                    if args.max_daily_loss > 0:
                        loss = _daily_loss(args.symbol)
                        if abs(loss) >= args.max_daily_loss:
                            msg = f"Daily loss limit reached: {loss:+.2f} USD (limit: -{args.max_daily_loss:.2f})"
                            print(c(f"  🛑 {msg}", Colors.RED))
                            notify(f"🛑 {msg} — {args.symbol}\nBot stopped.")
                            break
                print(c(f"[{now}] {args.symbol} FLAT — running inference...", Colors.CYAN))
                df = fetch_candles(args.symbol, tf, n_candles)
                X  = build_input(df, args.window, args.atr_period, args.adx_period,
                                 args.adx_min, args.stoch_k, args.stoch_d, args.vol_window)
                results                = {o.name: v for o, v in zip(sess.get_outputs(), sess.run(None, {inp_name: X}))}
                p_hold, p_buy, p_sell  = results['probabilities'][0]
                _stats.update(p_buy, p_sell, p_hold)

                p_buy_str  = c(f'P(buy)={p_buy:.3f}',   Colors.GREEN  if p_buy  >= args.confidence else Colors.WHITE)
                p_sell_str = c(f'P(sell)={p_sell:.3f}',  Colors.RED    if p_sell >= args.confidence else Colors.WHITE)
                p_hold_str = c(f'P(hold)={p_hold:.3f}',  Colors.YELLOW if p_hold >= args.confidence else Colors.WHITE)
                exp_str    = c(f'Exp={args.confidence:.3f}', Colors.YELLOW)
                print(f"  {p_buy_str}  {p_sell_str}  {p_hold_str}  {exp_str}", end='')
                print(c(f"\n{_stats}", Colors.CYAN), end='')

                last_close = df['close'].iloc[-1]   # forming candle (live price)
                ema_val    = ta.trend.EMAIndicator(df['close'], window=args.ema_period).ema_indicator().iloc[-1]
                ema_distance_pct = ((last_close - ema_val) / ema_val) * 100
                
                # Calculate ATR and volume for logging
                atr_val = ta.volatility.AverageTrueRange(df['high'], df['low'], df['close'], window=args.atr_period).average_true_range().iloc[-1]
                # Handle volume column - MT5 may use 'tick_volume' or 'volume'
                volume_val = 0.0
                if 'tick_volume' in df.columns:
                    volume_val = df['tick_volume'].iloc[-1]
                elif 'volume' in df.columns:
                    volume_val = df['volume'].iloc[-1]
                account_balance = mt5.account_info().balance
                has_open_position = _state.ticket != 0
                
                # Track consecutive signals
                global _last_signal, _consecutive_signal_count
                current_signal = None
                signal_decision = 'HOLD'
                signal_reason = 'no_signal'
                ema_filter_passed = False
                ema_distance_filter_passed = False
                ema_n_candles_filter_passed = False
                
                # Determine signal and filter status
                if p_buy >= args.confidence:
                    current_signal = 'BUY'
                    ema_filter_passed = last_close > ema_val
                    ema_distance_filter_passed = abs(ema_distance_pct) <= args.ema_distance

                    # Check if EMA is greater than the N previous candles
                    ema_n_candles_filter_passed = all(ema_val > prev_ema for prev_ema in df['close'].iloc[-args.ema_period:])

                    if ema_filter_passed and ema_distance_filter_passed and ema_n_candles_filter_passed:
                        signal_decision = 'BUY'
                        signal_reason = 'signal_passed_filters'
                    elif not ema_filter_passed:
                        signal_reason = 'ema_filter_failed'
                    elif not ema_n_candles_filter_passed:
                        signal_reason = 'ema_n_candles_filter_failed'
                    else:
                        signal_reason = 'ema_distance_filter_failed'
                elif p_sell >= args.confidence:
                    current_signal = 'SELL'
                    ema_filter_passed = last_close < ema_val
                    ema_distance_filter_passed = abs(ema_distance_pct) <= args.ema_distance

                    # Check if EMA is less than the N previous candles
                    ema_n_candles_filter_passed = all(ema_val < prev_ema for prev_ema in df['close'].iloc[-args.ema_period:])

                    if ema_filter_passed and ema_distance_filter_passed and ema_n_candles_filter_passed:
                        signal_decision = 'SELL'
                        signal_reason = 'signal_passed_filters'
                    elif not ema_filter_passed:
                        signal_reason = 'ema_filter_failed'
                    elif not ema_n_candles_filter_passed:
                        signal_reason = 'ema_n_candles_filter_failed'
                    else:
                        signal_reason = 'ema_distance_filter_failed'
                
                # Update consecutive signal count
                if current_signal == _last_signal:
                    _consecutive_signal_count += 1
                else:
                    _consecutive_signal_count = 1 if current_signal is not None else 0
                    _last_signal = current_signal
                
                # Calculate time since last trade
                time_since_last_trade_sec = 0
                if _last_trade_time:
                    time_since_last_trade_sec = int((datetime.now() - _last_trade_time).total_seconds())
                
                # Log inference statistics
                _log_inference(
                    symbol=args.symbol,
                    timeframe=args.timeframe,
                    p_buy=p_buy,
                    p_sell=p_sell,
                    p_hold=p_hold,
                    close_price=last_close,
                    ema_value=ema_val,
                    ema_distance_pct=ema_distance_pct,
                    ema_filter_passed=ema_filter_passed,
                    ema_distance_filter_passed=ema_distance_filter_passed,
                    confidence_threshold=args.confidence,
                    signal_decision=signal_decision,
                    signal_reason=signal_reason,
                    atr_value=atr_val,
                    volume=volume_val,
                    account_balance=account_balance,
                    has_open_position=has_open_position,
                    consecutive_signal_count=_consecutive_signal_count,
                    time_since_last_trade_sec=time_since_last_trade_sec
                )

                # daily loss gate — block new entries if limit reached
                daily_loss_hit = False
                if args.max_daily_loss > 0:
                    loss = _daily_loss(args.symbol)
                    if abs(loss) >= args.max_daily_loss:
                        daily_loss_hit = True
                        print(c(f'  → Daily loss limit reached: {loss:+.2f} USD — no new trades', Colors.RED))

                if daily_loss_hit:
                    pass  # inference ran, stats updated, but no trade opened
                elif p_buy >= args.confidence:
                    if last_close > ema_val and abs(ema_distance_pct) <= args.ema_distance:
                        print(c(f'  → BUY signal (EMA distance: {ema_distance_pct:.3f}%)', Colors.GREEN))
                        open_position(args.symbol, is_buy=True, lot=args.lot, tf=tf,
                                      atr_period=args.atr_period,
                                      sl_mult=args.sl_mult, tp_mult=args.tp_mult,
                                      deviation=args.deviation, magic=args.magic,
                                      confidence=p_buy,
                                      lot_usd=args.lot_usd,
                                      dry_run=args.dry_run)
                        # Update last trade time
                        _last_trade_time = datetime.now()
                    else:
                        if last_close <= ema_val:
                            print(c(f'  → EMA filter: close={last_close:.5f} <= EMA={ema_val:.5f} — BUY blocked', Colors.YELLOW))
                        else:
                            print(c(f'  → EMA distance filter: {ema_distance_pct:.3f}% > {args.ema_distance:.3f}% — BUY blocked', Colors.YELLOW))
                elif p_sell >= args.confidence:
                    if last_close < ema_val and abs(ema_distance_pct) <= args.ema_distance:
                        print(c(f'  → SELL signal (EMA distance: {ema_distance_pct:.3f}%)', Colors.RED))
                        open_position(args.symbol, is_buy=False, lot=args.lot, tf=tf,
                                      atr_period=args.atr_period,
                                      sl_mult=args.sl_mult, tp_mult=args.tp_mult,
                                      deviation=args.deviation, magic=args.magic,
                                      confidence=p_sell,
                                      lot_usd=args.lot_usd,
                                      dry_run=args.dry_run)
                        # Update last trade time
                        _last_trade_time = datetime.now()
                    else:
                        if last_close >= ema_val:
                            print(c(f'  → EMA filter: close={last_close:.5f} >= EMA={ema_val:.5f} — SELL blocked', Colors.YELLOW))
                        else:
                            print(c(f'  → EMA distance filter: {abs(ema_distance_pct):.3f}% > {args.ema_distance:.3f}% — SELL blocked', Colors.YELLOW))
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
    parser.add_argument('--lot',        type=float, default=1.0,  help='Order lot size (used when --lot_usd is not set)')
    parser.add_argument('--lot_usd',    type=float, default=0.0,  help='Trade amount in account currency (overrides --lot when > 0)')
    parser.add_argument('--interval',   type=int,   default=60,   help='Seconds between cycles')
    parser.add_argument('--atr_period', type=int,   default=14,   help='ATR period for SL/TP calculation')
    parser.add_argument('--sl_mult',    type=float, default=1.0,  help='SL = ATR * sl_mult')
    parser.add_argument('--tp_mult',    type=float, default=1.0,  help='Imaginary TP = ATR * tp_mult')
    parser.add_argument('--max_daily_loss', type=float, default=5.0, help='Max daily loss in USD before stopping (0 = disabled)')
    parser.add_argument('--adx_period', type=int,   default=8,    help='ADX indicator period')
    parser.add_argument('--adx_min',    type=float, default=20.0, help='ADX minimum threshold')
    parser.add_argument('--stoch_k',    type=int,   default=5,    help='Stochastic K period')
    parser.add_argument('--stoch_d',    type=int,   default=3,    help='Stochastic D period')
    parser.add_argument('--vol_window', type=int,   default=20,   help='Volume rolling window')
    parser.add_argument('--deviation',  type=int,   default=20,   help='Max price deviation (slippage) in points')
    parser.add_argument('--magic',      type=int,   default=int(time.time()),    help='Magic number for orders')
    parser.add_argument('--ema_period', type=int,   default=18,   help='EMA period for trend filter')
    parser.add_argument('--ema_distance', type=float, default=0.05, help='Max percentage distance from EMA to allow trades (e.g., 0.02 for 0.02%%)')
    parser.add_argument('--dry_run',    action='store_true', help='Prevent trades, log WOULD_EXECUTE instead')
    args = parser.parse_args()
    run(args)


if __name__ == '__main__':
    main()
