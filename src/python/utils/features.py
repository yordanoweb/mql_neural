"""
Feature engineering utilities for ONNX training scripts.

All functions take a DataFrame with columns: time, open, high, low, close, tick_volume
and return the same DataFrame with new feature columns appended.

Call df.dropna(inplace=True) after building all features, before windowing.
"""

import numpy as np
import pandas as pd
import ta


def add_base_features(df: pd.DataFrame, atr_period: int = 14, rsi_period: int = 14) -> pd.DataFrame:
    """3 base features: return, atr_norm, rsi_norm."""
    df['return']   = df['close'].pct_change()
    df['atr_norm'] = ta.volatility.AverageTrueRange(
        df['high'], df['low'], df['close'], window=atr_period
    ).average_true_range() / df['close']
    df['rsi_norm'] = ta.momentum.RSIIndicator(df['close'], window=rsi_period).rsi() / 100
    return df


def add_adx_features(df: pd.DataFrame, adx_period: int = 14) -> pd.DataFrame:
    """3 ADX features: adx_norm, dip_norm, din_norm."""
    adx = ta.trend.ADXIndicator(df['high'], df['low'], df['close'], window=adx_period)
    df['adx_norm'] = adx.adx() / 100
    df['dip_norm'] = adx.adx_pos() / 100
    df['din_norm'] = adx.adx_neg() / 100
    return df


def add_stoch_features(df: pd.DataFrame, k_period: int = 10, d_period: int = 3) -> pd.DataFrame:
    """4 stochastic features: stoch_k, stoch_d, stoch_diff, stoch_signal."""
    stoch = ta.momentum.StochasticOscillator(
        df['high'], df['low'], df['close'], window=k_period, smooth_window=d_period
    )
    df['stoch_k']      = stoch.stoch() / 100
    df['stoch_d']      = stoch.stoch_signal() / 100
    df['stoch_diff']   = df['stoch_k'] - df['stoch_d']
    df['stoch_signal'] = (df['stoch_k'] > df['stoch_d']).astype(float)
    return df


def add_volume_features(df: pd.DataFrame, vol_window: int = 10) -> pd.DataFrame:
    """5 volume features: vol_norm, vol_change, vol_ma_ratio, obv_norm, vol_spike."""
    df['vol_norm']     = df['tick_volume'] / df['tick_volume'].rolling(vol_window).mean()
    df['vol_change']   = df['tick_volume'].pct_change()
    df['vol_ma_ratio'] = df['tick_volume'] / df['tick_volume'].rolling(vol_window * 2).mean()
    df['obv_norm']     = ta.volume.OnBalanceVolumeIndicator(
        df['close'], df['tick_volume']
    ).on_balance_volume().pct_change()
    df['vol_spike']    = (
        df['tick_volume'] > df['tick_volume'].rolling(vol_window).mean() * 2
    ).astype(float)
    return df


def add_label(df: pd.DataFrame, forward_bars: int = 1, min_pct: float = 0.0) -> pd.DataFrame:
    """Binary label: 1 = price rises by min_pct in next forward_bars candles."""
    future = df['close'].shift(-forward_bars)
    df['label'] = ((future - df['close']) / df['close'] > min_pct).astype(int)
    df.dropna(subset=['label'], inplace=True)
    return df


def make_windows(
    df: pd.DataFrame, feature_cols: list[str], window: int
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build (X, y) arrays from a labelled DataFrame.

    X shape: (n_samples, window * n_features)  — row-major flatten
    y shape: (n_samples,)
    """
    arr    = df[feature_cols].values.astype(np.float32)
    labels = df['label'].values.astype(np.int64)
    X, y   = [], []
    for i in range(window, len(arr)):
        X.append(arr[i - window:i].flatten())
        y.append(labels[i])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int64)
