"""
Feature engineering utilities for ONNX training scripts.

All functions take a DataFrame with columns: time, open, high, low, close, tick_volume
and return the same DataFrame with new feature columns appended.

Call df.dropna(inplace=True) after building all features, before windowing.
"""

import numpy as np
import pandas as pd
import ta


def safe_series(s: pd.Series, fill: float = 0.0, clip: float = 10.0) -> pd.Series:
    """Replace inf/-inf with NaN, fill NaN, then clip to [-clip, clip]."""
    return s.replace([np.inf, -np.inf], np.nan).fillna(fill).clip(-clip, clip)


def add_adx_features(df: pd.DataFrame, adx_period: int = 14, adx_min: float = 20.0) -> pd.DataFrame:
    """5 ADX features: strength, di_signal, di_separation, momentum, regime."""
    ind      = ta.trend.ADXIndicator(df['high'], df['low'], df['close'], window=adx_period)
    adx      = ind.adx()
    di_plus  = ind.adx_pos()
    di_minus = ind.adx_neg()

    df['adx_strength']  = safe_series((adx - adx_min) / (100.0 - adx_min), clip=1.0)
    di_signal           = np.where(di_plus > di_minus, 1.0, np.where(di_minus > di_plus, -1.0, 0.0))
    df['adx_di_signal'] = safe_series(pd.Series(di_signal, index=adx.index))
    di_sum              = (di_plus + di_minus).replace(0, np.nan).fillna(1.0)
    df['adx_di_sep']    = safe_series((di_plus - di_minus) / di_sum, clip=1.0)
    df['adx_momentum']  = safe_series(adx.diff() / 100.0, clip=1.0)
    regime              = np.where(adx < adx_min, 0.0, np.where(adx < 40.0, 0.5, 1.0))
    df['adx_regime']    = safe_series(pd.Series(regime, index=adx.index), clip=1.0)
    return df


def add_stoch_features(df: pd.DataFrame, k_period: int = 14, d_period: int = 3) -> pd.DataFrame:
    """4 stochastic features: momentum, position, velocity, divergence."""
    stoch = ta.momentum.StochasticOscillator(df['high'], df['low'], df['close'],
                                             window=k_period, smooth_window=d_period)
    k = stoch.stoch()
    d = stoch.stoch_signal()

    df['stoch_momentum']   = safe_series((k - d) / 100.0)
    df['stoch_position']   = safe_series((k - 50.0) / 50.0)
    df['stoch_velocity']   = safe_series(k.diff() / 100.0)
    ob = np.where(k > 80, -(k - 80) / 20.0, 0)
    os_ = np.where(k < 20, (20 - k) / 20.0, 0)
    df['stoch_divergence'] = safe_series(pd.Series(ob + os_, index=k.index))
    return df


def add_volume_features(df: pd.DataFrame, vol_window: int = 20) -> pd.DataFrame:
    """5 volume features: ratio, momentum, price_div, percentile, zscore."""
    vol    = df['tick_volume']
    close  = df['close']
    vol_ma = vol.rolling(vol_window).mean()
    vol_std= vol.rolling(vol_window).std()

    df['vol_ratio']     = safe_series(vol / vol_ma.replace(0, np.nan).fillna(1.0), clip=5.0)
    fast                = vol.ewm(span=5,  adjust=False).mean()
    slow                = vol.ewm(span=20, adjust=False).mean()
    df['vol_momentum']  = safe_series((fast - slow) / slow.replace(0, np.nan).fillna(1.0), clip=5.0)
    pc                  = safe_series(close.pct_change().abs(), clip=5.0)
    vc                  = safe_series(vol.pct_change().abs(), clip=5.0)
    df['vol_price_div'] = safe_series(vc - pc, clip=5.0)
    df['vol_percentile']= safe_series(
        (vol.rolling(vol_window).apply(
            lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False
        ) - 0.5) * 2
    )
    df['vol_zscore']    = safe_series(
        np.clip((vol - vol_ma) / vol_std.replace(0, np.nan).fillna(1.0), -3, 3) / 3.0
    )
    return df


def add_price_features(df: pd.DataFrame, atr_period: int = 14) -> pd.DataFrame:
    """2 basic price features (body, range) normalised by ATR. Also stores raw ATR column."""
    atr      = ta.volatility.AverageTrueRange(
        df['high'], df['low'], df['close'], window=atr_period
    ).average_true_range()
    atr_safe        = atr.replace(0, np.nan).ffill().fillna(1.0)
    df['atr']       = atr
    df['feat_body'] = safe_series((df['close'] - df['open']) / atr_safe)
    df['feat_range']= safe_series((df['high']  - df['low'])  / atr_safe)
    return df


def add_label(df: pd.DataFrame, forward_bars: int = 10, min_profit_atr: float = 1.5) -> pd.DataFrame:
    """
    ATR-based binary label over a future window:
      1 (buy)  — max upside   >= min_profit_atr AND upside > downside
      0 (sell) — otherwise
    Requires 'atr' column (added by add_price_features).
    Last forward_bars rows are dropped (no future data available).
    """
    labels  = np.zeros(len(df), dtype=np.int64)
    atr_arr = df['atr'].values
    close   = df['close'].values
    high    = df['high'].values
    low     = df['low'].values

    for i in range(len(df) - forward_bars):
        if np.isnan(atr_arr[i]) or atr_arr[i] == 0:
            continue
        entry       = close[i]
        future_high = high[i+1 : i+forward_bars+1].max()
        future_low  = low[i+1  : i+forward_bars+1].min()
        upside      = (future_high - entry) / atr_arr[i]
        downside    = (entry - future_low)  / atr_arr[i]
        if upside >= min_profit_atr and upside > downside:
            labels[i] = 1

    df = df.copy()
    df['label'] = labels
    return df.iloc[:-forward_bars].reset_index(drop=True)


def make_windows(
    df: pd.DataFrame, feature_cols: list[str], window: int
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build (X, y) arrays from a labelled DataFrame.
    X shape: (n_samples, window * n_features) — row-major flatten
    y shape: (n_samples,)
    """
    arr    = df[feature_cols].values.astype(np.float32)
    labels = df['label'].values.astype(np.int64)
    X, y   = [], []
    for i in range(window, len(arr)):
        X.append(arr[i - window:i].flatten())
        y.append(labels[i])
    X = np.array(X, dtype=np.float32)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    return X, np.array(y, dtype=np.int64)
