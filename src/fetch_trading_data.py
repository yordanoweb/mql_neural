"""
Fetch and compose M5/H1 JSON trading data — ROBUST VERSION v3.
Handles yfinance multi-index columns, None returns, and mock fallback.
"""

import argparse
import json
import numpy as np
import pandas as pd
import time
import sys
from datetime import datetime

try:
    import yfinance as yf
    YF_AVAILABLE = True
except ImportError:
    YF_AVAILABLE = False


def calculate_rsi(prices, period=14):
    delta = prices.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def calculate_atr(df, period=14):
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = ranges.max(axis=1)
    return true_range.rolling(window=period).mean()


def parabolic_sar(df, af=0.02, max_af=0.2):
    sar = df['Close'].copy()
    ep = df['High'].iloc[0]
    trend = 1
    af_val = af
    for i in range(1, len(df)):
        if trend == 1:
            sar.iloc[i] = sar.iloc[i-1] + af_val * (ep - sar.iloc[i-1])
            if df['Low'].iloc[i] < sar.iloc[i]:
                trend, sar.iloc[i], ep, af_val = -1, ep, df['Low'].iloc[i], af
            elif df['High'].iloc[i] > ep:
                ep, af_val = df['High'].iloc[i], min(af_val + af, max_af)
        else:
            sar.iloc[i] = sar.iloc[i-1] + af_val * (ep - sar.iloc[i-1])
            if df['High'].iloc[i] > sar.iloc[i]:
                trend, sar.iloc[i], ep, af_val = 1, ep, df['High'].iloc[i], af
            elif df['Low'].iloc[i] < ep:
                ep, af_val = df['Low'].iloc[i], min(af_val + af, max_af)
    return sar


def compute_5m_indicators(df):
    df['vol_proxy'] = (df['High'] - df['Low']) * df['Volume'].replace(0, np.nan).fillna(1)
    df['price_change'] = df['Close'].diff()
    df['wvo'] = (df['price_change'] * df['vol_proxy']).rolling(10).sum() / df['vol_proxy'].rolling(10).sum()
    df['rsi_14'] = calculate_rsi(df['Close'], 14)
    df['arsi'] = df['rsi_14'].ewm(span=5, adjust=False).mean()
    df['vwap_num'] = (df['Close'] * df['vol_proxy']).cumsum()
    df['vwap_den'] = df['vol_proxy'].cumsum()
    df['vwap'] = df['vwap_num'] / df['vwap_den']
    df['atr_14'] = calculate_atr(df, 14)
    df['vwio'] = (df['Close'] - df['vwap']) / df['atr_14']
    return df


def compute_1h_indicators(df):
    df['vol_proxy'] = (df['High'] - df['Low']) * df['Volume'].replace(0, np.nan).fillna(1)
    recent_high = df['High'].rolling(24).max()
    recent_low = df['Low'].rolling(24).min()
    df['hourly_hl_percentile'] = (df['Close'] - recent_low) / (recent_high - recent_low)
    df['hourly_vol_momentum'] = df['vol_proxy'].rolling(5).mean() / df['vol_proxy'].rolling(20).mean()
    df['ema_12'] = df['Close'].ewm(span=12, adjust=False).mean()
    df['ema_26'] = df['Close'].ewm(span=26, adjust=False).mean()
    df['macd'] = df['ema_12'] - df['ema_26']
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    df['macd_hist'] = df['macd'] - df['macd_signal']
    df['parabolic_sar'] = parabolic_sar(df)
    df['ema_50'] = df['Close'].ewm(span=50, adjust=False).mean()
    df['ema_200'] = df['Close'].ewm(span=200, adjust=False).mean()
    df['ema_crossover'] = np.where(df['ema_50'] > df['ema_200'], 'Bullish', 'Bearish')
    return df


def flatten_columns(df):
    """Flatten yfinance multi-index columns like ('Close', 'NQ=F') -> 'Close'."""
    new_cols = []
    for c in df.columns:
        if isinstance(c, tuple):
            new_cols.append(c[0])  # Take first element
        else:
            new_cols.append(c)
    df.columns = new_cols
    return df


def fetch_data_yf(ticker, interval, period='5d', retries=3, delay=2):
    """Fetch from yfinance with retries and fallback to 1m->5m resampling."""
    if not YF_AVAILABLE:
        raise ImportError("yfinance not installed. Run: pip install yfinance")

    for attempt in range(retries):
        try:
            df = yf.download(ticker, period=period, interval=interval,
                             progress=False, auto_adjust=False, threads=False)
            if df is not None and not df.empty:
                return flatten_columns(df)
        except Exception as e:
            print(f"  Attempt {attempt + 1}/{retries} failed: {e}", file=sys.stderr)
            if attempt < retries - 1:
                time.sleep(delay)

    # FALLBACK: fetch 1m and resample to 5m
    if interval == '5m':
        print(f"  Falling back to 1m -> 5m resampling for {ticker}...", file=sys.stderr)
        try:
            df = yf.download(ticker, period='5d', interval='1m',
                             progress=False, auto_adjust=False, threads=False)
            if df is not None and not df.empty:
                df = flatten_columns(df)
                df.index = pd.to_datetime(df.index)
                return df.resample('5min').agg({
                    'Open': 'first', 'High': 'max', 'Low': 'min',
                    'Close': 'last', 'Volume': 'sum'
                }).dropna()
        except Exception as e:
            print(f"  Fallback failed: {e}", file=sys.stderr)

    raise ValueError(f"No data for {ticker} ({interval}) after {retries} attempts")


def generate_mock_data(symbol):
    """Generate realistic mock data when live data is unavailable."""
    np.random.seed(42)
    base = {
        'EURUSD': 1.1567, 'NQ': 30088, 'ES': 5600,
        'BTCUSDT': 64250, 'ETHUSDT': 3450, 'GBPUSD': 1.2750
    }.get(symbol, 100.0)

    # 5m: 288 candles
    n = 288
    dates = pd.date_range(end=pd.Timestamp.utcnow(), periods=n, freq='5min')
    noise = np.cumsum(np.random.randn(n) * 0.0003)
    closes = base * (1 + noise)
    df_5m = pd.DataFrame({
        'Date': dates, 'Open': closes * (1 + np.random.randn(n) * 0.0001),
        'High': closes * (1 + abs(np.random.randn(n)) * 0.0005),
        'Low': closes * (1 - abs(np.random.randn(n)) * 0.0005),
        'Close': closes, 'Volume': np.random.randint(100, 5000, n)
    })
    df_5m['High'] = df_5m[['Open', 'High', 'Close']].max(axis=1)
    df_5m['Low'] = df_5m[['Open', 'Low', 'Close']].min(axis=1)

    # 1h: 120 candles
    n = 120
    dates = pd.date_range(end=pd.Timestamp.utcnow(), periods=n, freq='1h')
    noise = np.cumsum(np.random.randn(n) * 0.001)
    closes = base * (1 + noise)
    df_1h = pd.DataFrame({
        'Date': dates, 'Open': closes * (1 + np.random.randn(n) * 0.0003),
        'High': closes * (1 + abs(np.random.randn(n)) * 0.001),
        'Low': closes * (1 - abs(np.random.randn(n)) * 0.001),
        'Close': closes, 'Volume': np.random.randint(1000, 50000, n)
    })
    df_1h['High'] = df_1h[['Open', 'High', 'Close']].max(axis=1)
    df_1h['Low'] = df_1h[['Open', 'Low', 'Close']].min(axis=1)

    return df_5m, df_1h


def standardize_df(df):
    """Fix yfinance's inconsistent column naming and index handling."""
    df = flatten_columns(df)

    if 'Date' not in df.columns and 'Datetime' not in df.columns:
        df = df.reset_index()

    # Safe rename using str() to handle any column type
    rename_map = {}
    for c in df.columns:
        col_str = str(c)
        lower = col_str.lower()
        if lower in ['open', 'high', 'low', 'close', 'volume']:
            rename_map[c] = col_str.title()
        elif lower in ['datetime', 'date']:
            rename_map[c] = 'Date'
        else:
            rename_map[c] = col_str
    df = df.rename(columns=rename_map)

    if 'Date' not in df.columns:
        for c in df.columns:
            if pd.api.types.is_datetime64_any_dtype(df[c]):
                df = df.rename(columns={c: 'Date'})
                break

    return df.sort_values('Date').reset_index(drop=True)


def build_timeframe_json(df, tf):
    last = df.iloc[-1]
    vol = float(last['Volume']) if last['Volume'] > 0 else float(last.get('vol_proxy', 1))
    close = float(last['Close'])
    decimals = 5 if close < 10 else (4 if close < 100 else 2)

    price = {
        "open": round(float(last['Open']), decimals),
        "high": round(float(last['High']), decimals),
        "low": round(float(last['Low']), decimals),
        "close": round(close, decimals),
        "volume": round(vol, 2)
    }

    if tf == '5m':
        ind = {
            "WVO": round(float(last['wvo']), 5),
            "ARSI": round(float(last['arsi']), 2),
            "VWIO": round(float(last['vwio']), 2)
        }
    else:
        ind = {
            "Hourly_High_Low_Percentile": round(float(last['hourly_hl_percentile']), 4),
            "Hourly_Volume_Momentum": round(float(last['hourly_vol_momentum']), 4),
            "MACD_Histogram": round(float(last['macd_hist']), 2),
            "Parabolic_SAR": round(float(last['parabolic_sar']), 2),
            "EMA_50_200_Crossover": str(last['ema_crossover'])
        }

    return {"price": price, "indicators": ind}


def build_full_json(symbol, ticker, timestamp=None, mock=False):
    if timestamp is None:
        timestamp = int(datetime.utcnow().timestamp())

    if mock:
        df_5m, df_1h = generate_mock_data(symbol)
    else:
        df_5m = fetch_data_yf(ticker, '5m', period='5d')
        df_1h = fetch_data_yf(ticker, '60m', period='5d')

    df_5m = standardize_df(df_5m)
    df_1h = standardize_df(df_1h)

    # Validate required columns
    required = ['Open', 'High', 'Low', 'Close', 'Volume']
    for df_name, df in [('5m', df_5m), ('1h', df_1h)]:
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"{df_name} missing columns: {missing}. Got: {list(df.columns)}")

    df_5m = compute_5m_indicators(df_5m)
    df_1h = compute_1h_indicators(df_1h)

    return {
        "symbol": symbol,
        "timestamp": timestamp,
        "timeframes": {
            "5m": build_timeframe_json(df_5m, '5m'),
            "1h": build_timeframe_json(df_1h, '1h')
        },
        "order_book": {"bid_ask_spread": 0.25, "order_imbalance": 0.52},
        "sentiment": {
            "funding_rate": 0.0,
            "fear_greed_index": 50,
            "open_interest": 0,
            "whale_activity": {"inflow": 0.0, "outflow": 0.0}
        },
        "macro_factors": {
            "exchange_reserves": 0.0,
            "btc_hash_rate": 0.0,
            "fomc_event_impact": "Neutral"
        }
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--symbol', required=True)
    parser.add_argument('--ticker', help='Yahoo ticker override')
    parser.add_argument('--timestamp', type=int)
    parser.add_argument('--output')
    parser.add_argument('--mock', action='store_true', help='Use synthetic test data')
    args = parser.parse_args()

    ticker_map = {
        'EURUSD': 'EURUSD=X', 'NQ': 'NQ=F', 'ES': 'ES=F',
        'BTCUSDT': 'BTC-USD', 'ETHUSDT': 'ETH-USD',
        'GBPUSD': 'GBPUSD=X', 'USDJPY': 'USDJPY=X'
    }
    ticker = args.ticker or ticker_map.get(args.symbol, args.symbol)

    try:
        data = build_full_json(args.symbol, ticker, args.timestamp, mock=args.mock)
        out = json.dumps(data, indent=2)
        print(out)
        if args.output:
            with open(args.output, 'w') as f:
                f.write(out)
            print(f"Saved to {args.output}", file=sys.stderr)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        print(f"\nTry: python fetch_trading_data.py --symbol {args.symbol} --mock", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()
