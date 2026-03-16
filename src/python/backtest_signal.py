import argparse
import pandas as pd
import numpy as np

from ta.momentum import RSIIndicator, StochasticOscillator
from ta.trend import EMAIndicator, MACD
from ta.volatility import BollingerBands
from ta.volume import OnBalanceVolumeIndicator


# -----------------------------
# ARGUMENTS
# -----------------------------

parser = argparse.ArgumentParser(description="Backtest discovered signal")

parser.add_argument("csvfile")
parser.add_argument("signal")

args = parser.parse_args()

csv_path = args.csvfile
signal_name = args.signal


# -----------------------------
# LOAD DATA
# -----------------------------

df = pd.read_csv(csv_path)

df["timestamp"] = pd.to_datetime(df["time"])
df = df.sort_values("timestamp")
df.set_index("timestamp", inplace=True)


# -----------------------------
# INDICATORS
# -----------------------------

df["rsi"] = RSIIndicator(df["close"], window=14).rsi()

df["ema10"] = EMAIndicator(df["close"], window=10).ema_indicator()
df["ema20"] = EMAIndicator(df["close"], window=20).ema_indicator()
df["ema50"] = EMAIndicator(df["close"], window=50).ema_indicator()
df["ema200"] = EMAIndicator(df["close"], window=200).ema_indicator()

macd = MACD(df["close"])
df["macd"] = macd.macd()
df["macd_signal"] = macd.macd_signal()

stoch = StochasticOscillator(df["high"], df["low"], df["close"])
df["stoch"] = stoch.stoch()

bb = BollingerBands(df["close"])
df["bb_low"] = bb.bollinger_lband()
df["bb_high"] = bb.bollinger_hband()

df["volatility"] = df["close"].pct_change().rolling(24).std()
df["high_vol"] = df["volatility"] > df["volatility"].rolling(200).mean()

df["trend_up"] = df["ema50"] > df["ema200"]


# -----------------------------
# SIGNAL DEFINITIONS
# -----------------------------

signals = {}

signals["rsi_sell_75"] = df["rsi"] > 75
signals["macd_bull"] = df["macd"] > df["macd_signal"]

signals["ema_bear_50_200"] = df["ema50"] < df["ema200"]
signals["ema_bear_10_20"] = df["ema10"] < df["ema20"]

signals["stoch_sell_80"] = df["stoch"] > 80

signals["bb_low"] = df["close"] < df["bb_low"]

signals["highvol"] = df["high_vol"]


# -----------------------------
# BUILD SIGNAL FROM NAME
# -----------------------------

parts = signal_name.split("__")

condition = pd.Series(True, index=df.index)

for p in parts:

    if p == "highvol":
        condition &= df["high_vol"]

    elif p in signals:
        condition &= signals[p]

    else:
        print("Unknown component:", p)


df["signal"] = condition


# -----------------------------
# TRADE SIMULATION
# -----------------------------

hold_bars = 4
capital = 10000

equity = [capital]

trade_returns = []

for i in range(len(df)-hold_bars):

    if df["signal"].iloc[i]:

        entry = df["close"].iloc[i]
        exit_price = df["close"].iloc[i+hold_bars]

        r = (exit_price - entry) / entry

        trade_returns.append(r)

        capital *= (1+r)

        equity.append(capital)


equity_curve = pd.Series(equity)


# -----------------------------
# METRICS
# -----------------------------

returns = np.array(trade_returns)

if len(returns) == 0:
    print("No trades")
    exit()

winrate = (returns > 0).mean()

avg_return = returns.mean()

profit_factor = returns[returns>0].sum() / abs(returns[returns<0].sum())

sharpe = returns.mean() / returns.std() * np.sqrt(252*6)

peak = equity_curve.cummax()

drawdown = (equity_curve - peak) / peak

max_dd = drawdown.min()


# -----------------------------
# OUTPUT
# -----------------------------

print("\nTrades:", len(returns))

print("Winrate:", round(winrate,3))

print("Average return:", round(avg_return,5))

print("Profit factor:", round(profit_factor,3))

print("Sharpe (approx):", round(sharpe,3))

print("Max drawdown:", round(max_dd,3))
