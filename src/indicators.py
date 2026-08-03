
import csv
import math
from datetime import datetime

def read_csv_data(filepath):
    """Read CSV file and return lists of dates, opens, highs, lows, closes, volumes"""
    dates = []
    opens = []
    highs = []
    lows = []
    closes = []
    volumes = []
    
    with open(filepath, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Handle different possible column names
            date_key = next((k for k in row.keys() if 'date' in k.lower()), None)
            open_key = next((k for k in row.keys() if k.lower() in ['open', 'o']), None)
            high_key = next((k for k in row.keys() if k.lower() in ['high', 'h']), None)
            low_key = next((k for k in row.keys() if k.lower() in ['low', 'l']), None)
            close_key = next((k for k in row.keys() if k.lower() in ['close', 'c']), None)
            vol_key = next((k for k in row.keys() if 'vol' in k.lower()), None)
            
            if date_key:
                dates.append(row[date_key])
            if open_key:
                opens.append(float(row[open_key]))
            if high_key:
                highs.append(float(row[high_key]))
            if low_key:
                lows.append(float(row[low_key]))
            if close_key:
                closes.append(float(row[close_key]))
            if vol_key:
                volumes.append(float(row[vol_key]))
    
    return dates, opens, highs, lows, closes, volumes

def calculate_sma(data, period):
    """Calculate Simple Moving Average"""
    result = []
    for i in range(len(data)):
        if i < period - 1:
            result.append(None)
        else:
            sma = sum(data[i - period + 1:i + 1]) / period
            result.append(sma)
    return result

def calculate_ema(data, period, smoothing=2):
    """Calculate Exponential Moving Average"""
    result = []
    multiplier = smoothing / (period + 1)
    
    for i in range(len(data)):
        if i < period - 1:
            result.append(None)
        elif i == period - 1:
            # First EMA is SMA
            ema = sum(data[:period]) / period
            result.append(ema)
        else:
            ema = (data[i] - result[i-1]) * multiplier + result[i-1]
            result.append(ema)
    return result


def calculate_atr(highs, lows, closes, period=14, method='sma'):
    """Calculate Average True Range (ATR).

    Parameters
    ----------
    highs : list of float
    lows : list of float
    closes : list of float
    period : int, default 14
        Look‑back period for smoothing the true range.
    method : {'sma','ema'}
        Smoothing method to apply to the true range.  'sma' returns a
        simple moving average of the TR; 'ema' applies an exponential
        moving average (Wilder's smoothing).

    Returns
    -------
    atr : list of float or None
        ATR values aligned with input data; entries before enough data
        are ``None``.
    """
    tr = calculate_true_range(highs, lows, closes)
    if method.lower() == 'sma':
        return calculate_sma(tr, period)
    elif method.lower() == 'ema':
        return calculate_ema(tr, period)
    else:
        raise ValueError(f"Unknown method {method!r}, expected 'sma' or 'ema'.")


def calculate_rsi(closes, period=14):
    """Calculate the Relative Strength Index (RSI).

    This implementation follows Wilder's original formula using smoothed
    average gains and losses.  The returned list contains ``None`` until
    enough data points are available to compute the first RSI value.

    Parameters
    ----------
    closes : list of float
    period : int, default 14
        Look‑back period for the RSI calculation.

    Returns
    -------
    rsi : list of float or None
        RSI values in the range ``0..100``.
    """
    n = len(closes)
    if n == 0:
        return []

    # compute gains and losses
    gains = [0.0] * n
    losses = [0.0] * n
    for i in range(1, n):
        delta = closes[i] - closes[i-1]
        gains[i] = max(delta, 0.0)
        losses[i] = max(-delta, 0.0)

    avg_gain = [None] * n
    avg_loss = [None] * n
    rsi = [None] * n

    # first average is simple average of initial period
    if n >= period + 1:
        first_avg_gain = sum(gains[1:period+1]) / period
        first_avg_loss = sum(losses[1:period+1]) / period
        avg_gain[period] = first_avg_gain
        avg_loss[period] = first_avg_loss
        # first RSI value
        if first_avg_loss == 0:
            rsi[period] = 100.0
        else:
            rs = first_avg_gain / first_avg_loss
            rsi[period] = 100 - (100 / (1 + rs))

        # subsequent values use Wilder's smoothing
        for i in range(period + 1, n):
            avg_gain[i] = (avg_gain[i-1] * (period - 1) + gains[i]) / period
            avg_loss[i] = (avg_loss[i-1] * (period - 1) + losses[i]) / period
            if avg_loss[i] == 0:
                rsi[i] = 100.0
            else:
                rs = avg_gain[i] / avg_loss[i]
                rsi[i] = 100 - (100 / (1 + rs))

    return rsi

def calculate_true_range(highs, lows, closes):
    """Calculate True Range"""
    tr_values = []
    for i in range(len(highs)):
        if i == 0:
            tr = highs[i] - lows[i]
        else:
            tr1 = highs[i] - lows[i]
            tr2 = abs(highs[i] - closes[i-1])
            tr3 = abs(lows[i] - closes[i-1])
            tr = max(tr1, tr2, tr3)
        tr_values.append(tr)
    return tr_values

def calculate_dm(highs, lows):
    """Calculate +DM and -DM"""
    plus_dm = []
    minus_dm = []
    
    for i in range(len(highs)):
        if i == 0:
            plus_dm.append(0)
            minus_dm.append(0)
        else:
            up_move = highs[i] - highs[i-1]
            down_move = lows[i-1] - lows[i]
            
            if up_move > down_move and up_move > 0:
                plus_dm.append(up_move)
            else:
                plus_dm.append(0)
                
            if down_move > up_move and down_move > 0:
                minus_dm.append(down_move)
            else:
                minus_dm.append(0)
    
    return plus_dm, minus_dm

def calculate_adx(highs, lows, closes, period=14):
    """
    Calculate ADX (Average Directional Index) from scratch
    Returns: +DI, -DI, DX, ADX
    """
    n = len(closes)
    
    # Step 1: Calculate True Range
    tr = calculate_true_range(highs, lows, closes)
    
    # Step 2: Calculate +DM and -DM
    plus_dm, minus_dm = calculate_dm(highs, lows)
    
    # Step 3: Calculate smoothed values (Wilder's smoothing)
    tr_smooth = [None] * n
    plus_dm_smooth = [None] * n
    minus_dm_smooth = [None] * n
    
    # First smoothed values are sums
    if n >= period:
        tr_smooth[period-1] = sum(tr[:period])
        plus_dm_smooth[period-1] = sum(plus_dm[:period])
        minus_dm_smooth[period-1] = sum(minus_dm[:period])
        
        # Subsequent values using Wilder's formula
        for i in range(period, n):
            tr_smooth[i] = tr_smooth[i-1] - (tr_smooth[i-1] / period) + tr[i]
            plus_dm_smooth[i] = plus_dm_smooth[i-1] - (plus_dm_smooth[i-1] / period) + plus_dm[i]
            minus_dm_smooth[i] = minus_dm_smooth[i-1] - (minus_dm_smooth[i-1] / period) + minus_dm[i]
    
    # Step 4: Calculate +DI and -DI
    plus_di = [None] * n
    minus_di = [None] * n
    
    for i in range(n):
        if tr_smooth[i] is not None and tr_smooth[i] != 0:
            plus_di[i] = 100 * (plus_dm_smooth[i] / tr_smooth[i])
            minus_di[i] = 100 * (minus_dm_smooth[i] / tr_smooth[i])
        else:
            plus_di[i] = None
            minus_di[i] = None
    
    # Step 5: Calculate DX (Directional Movement Index)
    dx = [None] * n
    for i in range(n):
        if plus_di[i] is not None and minus_di[i] is not None:
            sum_di = plus_di[i] + minus_di[i]
            if sum_di != 0:
                dx[i] = 100 * abs(plus_di[i] - minus_di[i]) / sum_di
            else:
                dx[i] = 0
    
    # Step 6: Calculate ADX (smoothed DX)
    adx = [None] * n
    
    # First ADX is average of first 'period' DX values
    valid_dx = [x for x in dx if x is not None]
    if len(valid_dx) >= period:
        first_adx_idx = period - 1 + period - 1  # Index where we have enough DX values
        if first_adx_idx < n:
            adx[first_adx_idx] = sum(valid_dx[:period]) / period
            
            # Subsequent ADX values
            for i in range(first_adx_idx + 1, n):
                if dx[i] is not None:
                    adx[i] = ((adx[i-1] * (period - 1)) + dx[i]) / period
    
    return plus_di, minus_di, dx, adx

def calculate_stochastic(highs, lows, closes, k_period=14, d_period=3, slowing=3):
    """
    Calculate Stochastic Oscillator from scratch
    Returns: %K (fast), %D (slow), Slow %K, Slow %D
    """
    n = len(closes)
    
    # Step 1: Calculate %K (fast)
    k_fast = []
    for i in range(n):
        if i < k_period - 1:
            k_fast.append(None)
        else:
            lowest_low = min(lows[i - k_period + 1:i + 1])
            highest_high = max(highs[i - k_period + 1:i + 1])
            
            range_val = highest_high - lowest_low
            if range_val == 0:
                k_fast.append(50)  # Default to middle if no range
            else:
                k = 100 * ((closes[i] - lowest_low) / range_val)
                k_fast.append(k)
    
    # Step 2: Calculate %D (SMA of %K fast)
    d_fast = []
    for i in range(n):
        if i < k_period - 1 + d_period - 1:
            d_fast.append(None)
        else:
            valid_k = [x for x in k_fast[i - d_period + 1:i + 1] if x is not None]
            if valid_k:
                d_fast.append(sum(valid_k) / len(valid_k))
            else:
                d_fast.append(None)
    
    # Step 3: Calculate Slow %K (SMA of %K fast with slowing period)
    k_slow = []
    for i in range(n):
        if i < k_period - 1 + slowing - 1:
            k_slow.append(None)
        else:
            valid_k = [x for x in k_fast[i - slowing + 1:i + 1] if x is not None]
            if valid_k:
                k_slow.append(sum(valid_k) / len(valid_k))
            else:
                k_slow.append(None)
    
    # Step 4: Calculate Slow %D (SMA of Slow %K)
    d_slow = []
    for i in range(n):
        if i < k_period - 1 + slowing - 1 + d_period - 1:
            d_slow.append(None)
        else:
            valid_k_slow = [x for x in k_slow[i - d_period + 1:i + 1] if x is not None]
            if valid_k_slow:
                d_slow.append(sum(valid_k_slow) / len(valid_k_slow))
            else:
                d_slow.append(None)
    
    return k_fast, d_fast, k_slow, d_slow

# Create sample CSV data for demonstration
sample_data = """Date,Open,High,Low,Close,Volume
2024-01-01,100,105,98,102,1000000
2024-01-02,102,108,101,107,1200000
2024-01-03,107,110,105,106,1100000
2024-01-04,106,112,104,111,1300000
2024-01-05,111,115,109,113,1400000
2024-01-06,113,118,112,117,1500000
2024-01-07,117,119,114,115,1200000
2024-01-08,115,120,113,119,1600000
2024-01-09,119,125,118,124,1800000
2024-01-10,124,128,122,126,1700000
2024-01-11,126,130,124,129,1900000
2024-01-12,129,135,127,134,2000000
2024-01-13,134,138,132,136,2100000
2024-01-14,136,142,135,141,2200000
2024-01-15,141,145,139,143,2300000
2024-01-16,143,148,142,147,2400000
2024-01-17,147,152,146,151,2500000
2024-01-18,151,155,149,153,2600000
2024-01-19,153,158,152,157,2700000
2024-01-20,157,162,156,161,2800000
2024-01-21,161,165,159,163,2900000
2024-01-22,163,168,162,167,3000000
2024-01-23,167,172,166,171,3100000
2024-01-24,171,175,170,174,3200000
2024-01-25,174,179,173,178,3300000
2024-01-26,178,182,177,181,3400000
2024-01-27,181,186,180,185,3500000
2024-01-28,185,189,184,188,3600000
2024-01-29,188,192,187,191,3700000
2024-01-30,191,195,190,194,3800000"""

if __name__ == "__main__":
    # Write sample data to file
    with open('/tmp/sample_stock_data.csv', 'w') as f:
        f.write(sample_data)

    print("✅ Sample CSV created successfully!")
    print("\n" + "="*60)
    print("CALCULATING ADX AND STOCHASTIC FROM SCRATCH")
    print("="*60)

    # Read the data
    dates, opens, highs, lows, closes, volumes = read_csv_data('/tmp/sample_stock_data.csv')

    print(f"\n📊 Loaded {len(closes)} price points")
    print(f"Date range: {dates[0]} to {dates[-1]}")

    # Calculate ADX
    plus_di, minus_di, dx, adx = calculate_adx(highs, lows, closes, period=14)

    # Calculate Stochastic
    k_fast, d_fast, k_slow, d_slow = calculate_stochastic(highs, lows, closes, k_period=14, d_period=3, slowing=3)

    # Calculate ATR and RSI using new helpers
    atr = calculate_atr(highs, lows, closes, period=14, method='ema')
    rsi = calculate_rsi(closes, period=14)

    # Display results
    print("\n" + "="*60)
    print("ADX (Average Directional Index) Results")
    print("="*60)
    print(f"{'Date':<12} {'+DI':<8} {'-DI':<8} {'DX':<8} {'ADX':<8}")
    print("-"*60)

    for i in range(len(dates)):
        if adx[i] is not None:
            print(f"{dates[i]:<12} {plus_di[i]:<8.2f} {minus_di[i]:<8.2f} {dx[i]:<8.2f} {adx[i]:<8.2f}")

    print("\n" + "="*60)
    print("Stochastic Oscillator Results")
    print("="*60)
    print(f"{'Date':<12} {'%K Fast':<10} {'%D Fast':<10} {'Slow %K':<10} {'Slow %D':<10}")
    print("-"*60)

    for i in range(len(dates)):
        if k_slow[i] is not None:
            kf = f"{k_fast[i]:.2f}" if k_fast[i] is not None else "N/A"
            df = f"{d_fast[i]:.2f}" if d_fast[i] is not None else "N/A"
            ks = f"{k_slow[i]:.2f}" if k_slow[i] is not None else "N/A"
            ds = f"{d_slow[i]:.2f}" if d_slow[i] is not None else "N/A"
            print(f"{dates[i]:<12} {kf:<10} {df:<10} {ks:<10} {ds:<10}")

    # ATR results
    print("\n" + "="*60)
    print("ATR (Average True Range) Results")
    print("="*60)
    print(f"{'Date':<12} {'ATR':<8}")
    print("-"*60)
    for i in range(len(dates)):
        if atr[i] is not None:
            print(f"{dates[i]:<12} {atr[i]:<8.4f}")

    # RSI results
    print("\n" + "="*60)
    print("RSI (Relative Strength Index) Results")
    print("="*60)
    print(f"{'Date':<12} {'RSI':<8}")
    print("-"*60)
    for i in range(len(dates)):
        if rsi[i] is not None:
            print(f"{dates[i]:<12} {rsi[i]:<8.2f}")

    print("\n" + "="*60)
    print("INTERPRETATION GUIDE")
    print("="*60)
    print("\n📈 ADX (Average Directional Index):")
    print("   • ADX < 20: Weak trend (consolidation/ranging)")
    print("   • ADX 20-40: Trend developing")
    print("   • ADX 40+: Strong trend")
    print("   • +DI > -DI: Bullish momentum")
    print("   • -DI > +DI: Bearish momentum")

    print("\n📊 Stochastic Oscillator:")
    print("   • %K > 80: Overbought conditions")
    print("   • %K < 20: Oversold conditions")
    print("   • %K crosses above %D: Buy signal")
    print("   • %K crosses below %D: Sell signal")

    print("\n🛠️ ATR (Average True Range):")
    print("   • Measures volatility; higher ATR means higher price swings")
    print("   • Often used to set stop‑loss distance (e.g. 1–2×ATR)")

    print("\n📈 RSI (Relative Strength Index):")
    print("   • RSI > 70: Overbought, potential reversal down")
    print("   • RSI < 30: Oversold, potential reversal up")
    print("   • Crosses of 50 can indicate momentum shifts")
