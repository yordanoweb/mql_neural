"""
Backtest an ONNX model on historical CSV/Parquet data using execution script logic.

Simulates the full execution pipeline:
  - Feature engineering (same as training)
  - Entry logic (confidence threshold + EMA filter)
  - Exit logic (SL hit, imaginary TP → trailing on M1 reversals, profit lock)
  - Trade logging with PnL tracking

Usage:
    python backtest_onnx.py \
        --model onnx/ustec_m5_16_feat_adx_stoch_vol.onnx \
        --input data/USTEC/ustec_m5.parquet \
        --m1_input data/USTEC/ustec_m1.parquet \
        --symbol USTEC --timeframe M5 \
        --window 20 --confidence 0.60 \
        --atr_period 14 --sl_mult 1.5 --tp_mult 2.0 \
        --ema_period 18 --output backtest_trades.csv
"""

import argparse
import csv
import os
from dataclasses import dataclass
from datetime import datetime

import numpy as np
import onnxruntime as rt
import pandas as pd

from utils.features import add_price_features, add_adx_features, add_stoch_features, add_volume_features
from utils.colors import Colors, colorize as c

FEATURE_COLS = [
    'feat_body', 'feat_range',
    'adx_strength', 'adx_di_signal', 'adx_di_sep', 'adx_momentum', 'adx_regime',
    'stoch_momentum', 'stoch_position', 'stoch_velocity', 'stoch_divergence',
    'vol_ratio', 'vol_momentum', 'vol_price_div', 'vol_percentile', 'vol_zscore',
]


@dataclass
class Position:
    """Simulated open position state."""
    is_buy:           bool
    entry_price:      float
    entry_bar:        int
    sl_price:         float
    tp_target:        float
    trailing:         bool = False
    last_candle_time: object = None
    confidence:       float = 0.0
    atr:              float = 0.0


def load_data(path: str) -> pd.DataFrame:
    df = pd.read_parquet(path) if path.endswith('.parquet') else pd.read_csv(path)
    df['time'] = pd.to_datetime(df['time'])
    return df.sort_values('time').reset_index(drop=True)


def load_session(model_path: str) -> rt.InferenceSession:
    sess    = rt.InferenceSession(model_path)
    outputs = {o.name: o for o in sess.get_outputs()}
    assert 'probabilities' in outputs and outputs['probabilities'].shape[1] == 3, \
        f"Model must have 'probabilities' output [*, 3]. Got: {list(outputs.keys())}"
    return sess


def build_features(df: pd.DataFrame, atr_period: int, adx_period: int, adx_min: float,
                   stoch_k: int, stoch_d: int, vol_window: int) -> pd.DataFrame:
    df = add_price_features(df.copy(), atr_period)
    df = add_adx_features(df, adx_period, adx_min)
    df = add_stoch_features(df, stoch_k, stoch_d)
    df = add_volume_features(df, vol_window)
    df.dropna(inplace=True)
    return df


def make_input(df: pd.DataFrame, idx: int, window: int) -> np.ndarray:
    """Extract window ending at idx (inclusive) and flatten."""
    if idx < window - 1:
        raise ValueError(f"Not enough history at idx={idx}, need {window}")
    arr = df[FEATURE_COLS].iloc[idx - window + 1 : idx + 1].values.flatten().astype(np.float32)
    return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0).reshape(1, -1)


def check_sl_hit(pos: Position, bar: pd.Series) -> bool:
    """Check if SL was hit during this bar."""
    if pos.is_buy:
        return bar['low'] <= pos.sl_price
    else:
        return bar['high'] >= pos.sl_price


def check_trailing_exit(pos: Position, m1_df: pd.DataFrame, bar_time: pd.Timestamp) -> bool:
    """Check if M1 reversal candle appeared after imaginary TP was reached."""
    if not pos.trailing:
        return False
    # Find M1 candles between entry and current bar
    m1_slice = m1_df[(m1_df['time'] > bar_time - pd.Timedelta(minutes=5)) & (m1_df['time'] <= bar_time)]
    if m1_slice.empty:
        return False
    last_m1 = m1_slice.iloc[-1]
    bearish = last_m1['close'] < last_m1['open']
    bullish = last_m1['close'] > last_m1['open']
    return (pos.is_buy and bearish) or (not pos.is_buy and bullish)


def update_trailing(pos: Position, current_price: float) -> None:
    """Activate trailing mode if imaginary TP is reached."""
    if pos.trailing:
        return
    tp_hit = (current_price >= pos.tp_target) if pos.is_buy else (current_price <= pos.tp_target)
    if tp_hit:
        pos.trailing = True


def move_sl_to_previous_candle(pos: Position, df: pd.DataFrame, idx: int) -> None:
    """Move SL to previous candle's low (BUY) or high (SELL) if in profit."""
    if idx < 1:
        return
    current_bar = df.iloc[idx]
    prev_bar    = df.iloc[idx - 1]
    bar_time    = current_bar['time']

    if bar_time == pos.last_candle_time:
        return

    current_price = current_bar['close']
    in_profit = (current_price > pos.entry_price) if pos.is_buy else (current_price < pos.entry_price)
    if not in_profit:
        return

    pos.last_candle_time = bar_time
    new_sl = prev_bar['low'] if pos.is_buy else prev_bar['high']

    # new SL must be past entry
    if (pos.is_buy and new_sl <= pos.entry_price) or (not pos.is_buy and new_sl >= pos.entry_price):
        return

    # only improve SL
    if (pos.is_buy and new_sl <= pos.sl_price) or (not pos.is_buy and new_sl >= pos.sl_price):
        return

    pos.sl_price = new_sl


def log_trade(log_file: str, symbol: str, event: str, **kwargs) -> None:
    fields = ['timestamp', 'event', 'symbol', 'direction', 'price', 'sl', 'tp_target',
              'atr', 'confidence', 'pnl_pts', 'reason']
    exists = os.path.exists(log_file)
    with open(log_file, 'a', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        if not exists:
            w.writeheader()
        row = {k: '' for k in fields}
        row.update({'timestamp': datetime.now().isoformat(), 'event': event, 'symbol': symbol})
        row.update(kwargs)
        w.writerow(row)


def backtest(args):
    print(c("=" * 60, Colors.CYAN))
    print(c(f"Backtesting {args.model}", Colors.CYAN))
    print(c("=" * 60, Colors.CYAN))

    df    = load_data(args.input)
    m1_df = load_data(args.m1_input) if args.m1_input else pd.DataFrame()
    sess  = load_session(args.model)
    inp_name  = sess.get_inputs()[0].name
    prob_name = next(o.name for o in sess.get_outputs() if o.name == 'probabilities')

    print(f"Loaded {len(df)} bars from {args.input}")
    if not m1_df.empty:
        print(f"Loaded {len(m1_df)} M1 bars from {args.m1_input}")

    # Build features
    df = build_features(df, args.atr_period, args.adx_period, args.adx_min,
                        args.stoch_k, args.stoch_d, args.vol_window)
    print(f"After feature engineering: {len(df)} bars")

    # Compute EMA for filter
    df['ema'] = df['close'].ewm(span=args.ema_period, adjust=False).mean()

    position = None
    trades   = []
    equity   = args.initial_balance
    peak     = equity
    max_dd   = 0.0

    start_idx = args.window
    print(c(f"\nStarting backtest from bar {start_idx} to {len(df) - 1}...\n", Colors.WHITE))

    for i in range(start_idx, len(df)):
        bar = df.iloc[i]

        if position:
            # Manage open position
            move_sl_to_previous_candle(position, df, i)
            update_trailing(position, bar['close'])

            # Check SL hit
            if check_sl_hit(position, bar):
                exit_price = position.sl_price
                pnl_pts    = (exit_price - position.entry_price) * (1 if position.is_buy else -1)
                equity    += pnl_pts
                peak       = max(peak, equity)
                max_dd     = max(max_dd, peak - equity)
                log_trade(args.output, args.symbol, 'CLOSE',
                          direction='BUY' if position.is_buy else 'SELL',
                          price=exit_price, sl=position.sl_price, tp_target=position.tp_target,
                          pnl_pts=round(pnl_pts, 5), reason='sl_hit')
                trades.append({
                    'entry_bar': position.entry_bar, 'exit_bar': i,
                    'direction': 'BUY' if position.is_buy else 'SELL',
                    'entry_price': position.entry_price, 'exit_price': exit_price,
                    'pnl_pts': pnl_pts, 'reason': 'sl_hit'
                })
                print(c(f"[{i}] {bar['time']} CLOSE SL_HIT {trades[-1]['direction']} "
                        f"entry={position.entry_price:.5f} exit={exit_price:.5f} pnl={pnl_pts:+.2f}",
                        Colors.RED))
                position = None
                continue

            # Check trailing exit
            if not m1_df.empty and check_trailing_exit(position, m1_df, bar['time']):
                exit_price = bar['close']
                pnl_pts    = (exit_price - position.entry_price) * (1 if position.is_buy else -1)
                equity    += pnl_pts
                peak       = max(peak, equity)
                max_dd     = max(max_dd, peak - equity)
                log_trade(args.output, args.symbol, 'CLOSE',
                          direction='BUY' if position.is_buy else 'SELL',
                          price=exit_price, sl=position.sl_price, tp_target=position.tp_target,
                          pnl_pts=round(pnl_pts, 5), reason='trailing_exit')
                trades.append({
                    'entry_bar': position.entry_bar, 'exit_bar': i,
                    'direction': 'BUY' if position.is_buy else 'SELL',
                    'entry_price': position.entry_price, 'exit_price': exit_price,
                    'pnl_pts': pnl_pts, 'reason': 'trailing_exit'
                })
                print(c(f"[{i}] {bar['time']} CLOSE TRAILING {trades[-1]['direction']} "
                        f"entry={position.entry_price:.5f} exit={exit_price:.5f} pnl={pnl_pts:+.2f}",
                        Colors.MAGENTA))
                position = None
                continue

        else:
            # Run inference
            X = make_input(df, i, args.window)
            probs = sess.run([prob_name], {inp_name: X})[0][0]
            p_hold, p_buy, p_sell = probs

            # Check signals
            signal = None
            if p_buy >= args.confidence and p_buy > p_sell:
                signal = 'BUY'
            elif p_sell >= args.confidence and p_sell > p_buy:
                signal = 'SELL'

            if signal:
                # Apply EMA filter
                close_price = bar['close']
                ema_val     = bar['ema']
                if signal == 'BUY' and close_price < ema_val:
                    continue
                if signal == 'SELL' and close_price > ema_val:
                    continue

                # Open position at next bar's open (simulate 1-bar delay)
                if i + 1 >= len(df):
                    break
                next_bar    = df.iloc[i + 1]
                entry_price = next_bar['open']
                atr         = df.iloc[i]['atr'] if 'atr' in df.columns else 1.0
                is_buy      = (signal == 'BUY')
                sl_price    = entry_price - atr * args.sl_mult if is_buy else entry_price + atr * args.sl_mult
                tp_target   = entry_price + atr * args.tp_mult if is_buy else entry_price - atr * args.tp_mult

                position = Position(
                    is_buy=is_buy,
                    entry_price=entry_price,
                    entry_bar=i + 1,
                    sl_price=sl_price,
                    tp_target=tp_target,
                    confidence=p_buy if is_buy else p_sell,
                    atr=atr,
                )
                log_trade(args.output, args.symbol, 'OPEN',
                          direction=signal, price=entry_price, sl=sl_price, tp_target=tp_target,
                          atr=round(atr, 5), confidence=round(position.confidence, 3))
                print(c(f"[{i + 1}] {next_bar['time']} OPEN {signal} "
                        f"price={entry_price:.5f} sl={sl_price:.5f} iTP={tp_target:.5f} conf={position.confidence:.3f}",
                        Colors.GREEN if is_buy else Colors.RED))

    # Close any remaining position at last bar
    if position:
        exit_price = df.iloc[-1]['close']
        pnl_pts    = (exit_price - position.entry_price) * (1 if position.is_buy else -1)
        equity    += pnl_pts
        log_trade(args.output, args.symbol, 'CLOSE',
                  direction='BUY' if position.is_buy else 'SELL',
                  price=exit_price, sl=position.sl_price, tp_target=position.tp_target,
                  pnl_pts=round(pnl_pts, 5), reason='end_of_data')
        trades.append({
            'entry_bar': position.entry_bar, 'exit_bar': len(df) - 1,
            'direction': 'BUY' if position.is_buy else 'SELL',
            'entry_price': position.entry_price, 'exit_price': exit_price,
            'pnl_pts': pnl_pts, 'reason': 'end_of_data'
        })

    # Summary
    print(c("\n" + "=" * 60, Colors.CYAN))
    print(c("Backtest Summary", Colors.CYAN))
    print(c("=" * 60, Colors.CYAN))
    print(f"Total trades:     {len(trades)}")
    if trades:
        wins   = [t for t in trades if t['pnl_pts'] > 0]
        losses = [t for t in trades if t['pnl_pts'] <= 0]
        print(f"Wins:             {len(wins)} ({len(wins) / len(trades) * 100:.1f}%)")
        print(f"Losses:           {len(losses)} ({len(losses) / len(trades) * 100:.1f}%)")
        total_pnl = sum(t['pnl_pts'] for t in trades)
        avg_pnl   = total_pnl / len(trades)
        print(f"Total PnL (pts):  {total_pnl:+.2f}")
        print(f"Avg PnL (pts):    {avg_pnl:+.2f}")
        print(f"Max drawdown:     {max_dd:.2f}")
        print(f"Final equity:     {equity:.2f}")
    print(c(f"\nTrades logged to: {args.output}", Colors.GREEN))


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--model',          required=True,              help='ONNX model path')
    parser.add_argument('--input',          required=True,              help='CSV or Parquet input (trading timeframe)')
    parser.add_argument('--m1_input',       default=None,               help='M1 CSV or Parquet for trailing exit (optional)')
    parser.add_argument('--symbol',         required=True,              help='Symbol name')
    parser.add_argument('--timeframe',      required=True,              help='Timeframe: M1 M5 M15 M30 H1 H4 D1')
    parser.add_argument('--window',         type=int,   default=20,     help='Window size')
    parser.add_argument('--confidence',     type=float, default=0.60,   help='Minimum probability to open trade')
    parser.add_argument('--atr_period',     type=int,   default=14,     help='ATR period')
    parser.add_argument('--sl_mult',        type=float, default=1.5,    help='SL = ATR × sl_mult')
    parser.add_argument('--tp_mult',        type=float, default=2.0,    help='Imaginary TP = ATR × tp_mult')
    parser.add_argument('--ema_period',     type=int,   default=18,     help='EMA period for trend filter')
    parser.add_argument('--adx_period',     type=int,   default=8,      help='ADX period')
    parser.add_argument('--adx_min',        type=float, default=20.0,   help='ADX minimum threshold')
    parser.add_argument('--stoch_k',        type=int,   default=14,     help='Stochastic K period')
    parser.add_argument('--stoch_d',        type=int,   default=3,      help='Stochastic D period')
    parser.add_argument('--vol_window',     type=int,   default=20,     help='Volume rolling window')
    parser.add_argument('--initial_balance',type=float, default=10000.0,help='Starting equity for PnL tracking')
    parser.add_argument('--output',         default='backtest_trades.csv', help='Output CSV path')
    args = parser.parse_args()

    backtest(args)


if __name__ == '__main__':
    main()
