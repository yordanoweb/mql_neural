"""
Live inference loop: loads an ONNX model, polls MT5 for the latest candles,
runs feature engineering, and places buy/sell market orders based on model output.

Usage:
    python execute_onnx_adx_stoch_vol_on_mt5.py \
        --model onnx/ndx100_m5_12_feat_adx_stoch_vol.onnx \
        --symbol NAS100 --timeframe M5 \
        --window 20 --confidence 0.60 --lot 1.0 \
        --interval 60
"""

import argparse
import time

import numpy as np
import onnxruntime as rt
import pandas as pd

from utils.features import add_adx_features, add_stoch_features, add_volume_features

# MT5 timeframe constants (avoids importing MetaTrader5 at module level — Windows-only lib)
TIMEFRAME_MAP = {
    'M1': 1, 'M5': 5, 'M15': 15, 'M30': 30, 'H1': 16385, 'H4': 16388, 'D1': 16408,
}

FEATURE_COLS = [
    'adx_norm', 'dip_norm', 'din_norm',
    'stoch_k', 'stoch_d', 'stoch_diff', 'stoch_signal',
    'vol_norm', 'vol_change', 'vol_ma_ratio', 'obv_norm', 'vol_spike',
]


def load_session(model_path: str) -> rt.InferenceSession:
    sess = rt.InferenceSession(model_path)
    out_shape = sess.get_outputs()[0].shape
    assert out_shape[1] == 2, f"Expected output [*, 2], got {out_shape}"
    return sess


def fetch_candles(symbol: str, tf, n: int) -> pd.DataFrame:
    import MetaTrader5 as mt5
    rates = mt5.copy_rates_from_pos(symbol, tf, 0, n)
    if rates is None or len(rates) < n:
        raise RuntimeError(f"Not enough candles for {symbol}: {mt5.last_error()}")
    df = pd.DataFrame(rates)
    df.rename(columns={'tick_volume': 'tick_volume'}, inplace=True)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    return df


def build_input(df: pd.DataFrame, window: int,
                adx_period: int, stoch_k: int, stoch_d: int, vol_window: int) -> np.ndarray:
    df = add_adx_features(df.copy(), adx_period)
    df = add_stoch_features(df, stoch_k, stoch_d)
    df = add_volume_features(df, vol_window)
    df.dropna(inplace=True)
    if len(df) < window:
        raise RuntimeError("Not enough clean rows after feature computation")
    window_data = df[FEATURE_COLS].values[-window:].flatten().astype(np.float32)
    return window_data.reshape(1, -1)


def get_open_position(symbol: str):
    import MetaTrader5 as mt5
    positions = mt5.positions_get(symbol=symbol)
    return positions[0] if positions else None


def close_position(pos, lot: float) -> None:
    import MetaTrader5 as mt5
    direction = mt5.ORDER_TYPE_SELL if pos.type == mt5.POSITION_TYPE_BUY else mt5.ORDER_TYPE_BUY
    price     = mt5.symbol_info_tick(pos.symbol).bid if direction == mt5.ORDER_TYPE_SELL \
                else mt5.symbol_info_tick(pos.symbol).ask
    req = {
        'action':    mt5.TRADE_ACTION_DEAL,
        'symbol':    pos.symbol,
        'volume':    lot,
        'type':      direction,
        'position':  pos.ticket,
        'price':     price,
        'deviation': 20,
        'magic':     0,
        'comment':   'onnx_close',
        'type_time': mt5.ORDER_TIME_GTC,
        'type_filling': mt5.ORDER_FILLING_IOC,
    }
    mt5.order_send(req)


def open_position(symbol: str, order_type: int, lot: float) -> None:
    import MetaTrader5 as mt5
    tick  = mt5.symbol_info_tick(symbol)
    price = tick.ask if order_type == mt5.ORDER_TYPE_BUY else tick.bid
    req = {
        'action':    mt5.TRADE_ACTION_DEAL,
        'symbol':    symbol,
        'volume':    lot,
        'type':      order_type,
        'price':     price,
        'deviation': 20,
        'magic':     0,
        'comment':   'onnx_open',
        'type_time': mt5.ORDER_TIME_GTC,
        'type_filling': mt5.ORDER_FILLING_IOC,
    }
    result = mt5.order_send(req)
    label  = 'BUY' if order_type == mt5.ORDER_TYPE_BUY else 'SELL'
    print(f"  → {label} order: retcode={result.retcode}")


def run(args):
    import MetaTrader5 as mt5
    if not mt5.initialize():
        raise RuntimeError(f"MT5 init failed: {mt5.last_error()}")

    sess = load_session(args.model)
    inp_name = sess.get_inputs()[0].name
    tf = TIMEFRAME_MAP[args.timeframe.upper()]
    # fetch enough candles to cover window + indicator warm-up
    n_candles = args.window + 100

    print(f"Running inference on {args.symbol} {args.timeframe} every {args.interval}s")
    print(f"Model : {args.model}  |  confidence threshold: {args.confidence}")

    try:
        while True:
            df = fetch_candles(args.symbol, tf, n_candles)
            X  = build_input(df, args.window, args.adx_period,
                             args.stoch_k, args.stoch_d, args.vol_window)

            p_sell, p_buy = sess.run(None, {inp_name: X})[0][0]
            print(f"P(sell)={p_sell:.3f}  P(buy)={p_buy:.3f}", end='')

            pos = get_open_position(args.symbol)

            if p_buy >= args.confidence:
                print("  → signal: BUY")
                if pos and pos.type == mt5.POSITION_TYPE_SELL:
                    close_position(pos, args.lot)
                if not pos or pos.type == mt5.POSITION_TYPE_SELL:
                    open_position(args.symbol, mt5.ORDER_TYPE_BUY, args.lot)
            elif p_sell >= args.confidence:
                print("  → signal: SELL")
                if pos and pos.type == mt5.POSITION_TYPE_BUY:
                    close_position(pos, args.lot)
                if not pos or pos.type == mt5.POSITION_TYPE_BUY:
                    open_position(args.symbol, mt5.ORDER_TYPE_SELL, args.lot)
            else:
                print("  → no signal")

            time.sleep(args.interval)

    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        mt5.shutdown()


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--model',      required=True,  help='Path to ONNX model')
    parser.add_argument('--symbol',     required=True,  help='MT5 symbol, e.g. NAS100')
    parser.add_argument('--timeframe',  required=True,  help='M1 M5 M15 M30 H1 H4 D1')
    parser.add_argument('--window',     type=int,   default=20,   help='Window size (must match training)')
    parser.add_argument('--confidence', type=float, default=0.60, help='Min probability to place an order')
    parser.add_argument('--lot',        type=float, default=1.0,  help='Order lot size')
    parser.add_argument('--interval',   type=int,   default=60,   help='Seconds between inference cycles')
    parser.add_argument('--adx_period', type=int,   default=14,   help='ADX indicator period')
    parser.add_argument('--stoch_k',    type=int,   default=10,   help='Stochastic K period')
    parser.add_argument('--stoch_d',    type=int,   default=3,    help='Stochastic D period')
    parser.add_argument('--vol_window', type=int,   default=10,   help='Volume rolling window')
    args = parser.parse_args()
    run(args)


if __name__ == '__main__':
    main()
