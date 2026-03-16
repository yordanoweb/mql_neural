import argparse
import pandas as pd
import numpy as np

from ta.momentum import RSIIndicator, StochasticOscillator
from ta.trend import EMAIndicator, MACD
from ta.volatility import BollingerBands
from ta.volume import OnBalanceVolumeIndicator


# -----------------------------
# ARGUMENT PARSER
# -----------------------------

parser = argparse.ArgumentParser(description="Quant-style signal discovery from OHLCV CSV")
parser.add_argument("csvfile", help="Path to OHLCV CSV file")

args = parser.parse_args()
csv_path = args.csvfile


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

df["ema20"] = EMAIndicator(df["close"], window=20).ema_indicator()
df["ema50"] = EMAIndicator(df["close"], window=50).ema_indicator()
df["ema200"] = EMAIndicator(df["close"], window=200).ema_indicator()

macd = MACD(df["close"])
df["macd"] = macd.macd()
df["macd_signal"] = macd.macd_signal()

stoch = StochasticOscillator(df["high"], df["low"], df["close"])
df["stoch"] = stoch.stoch()

bb = BollingerBands(df["close"])
df["bb_high"] = bb.bollinger_hband()
df["bb_low"] = bb.bollinger_lband()

df["obv"] = OnBalanceVolumeIndicator(df["close"], df["tick_volume"]).on_balance_volume()


# -----------------------------
# MARKET REGIMES
# -----------------------------

df["volatility"] = df["close"].pct_change().rolling(24).std()

df["high_vol"] = df["volatility"] > df["volatility"].rolling(200).mean()
df["low_vol"] = df["volatility"] < df["volatility"].rolling(200).mean()

df["trend_up"] = df["ema50"] > df["ema200"]
df["trend_down"] = df["ema50"] < df["ema200"]


# -----------------------------
# SIGNAL GENERATION
# -----------------------------

signals = {}

# RSI sweeps
for level in [20,25,30,35,40]:
    signals[f"rsi_buy_{level}"] = df["rsi"] < level

for level in [60,65,70,75,80]:
    signals[f"rsi_sell_{level}"] = df["rsi"] > level


# stochastic sweeps
for level in [15,20,25]:
    signals[f"stoch_buy_{level}"] = df["stoch"] < level

for level in [75,80,85]:
    signals[f"stoch_sell_{level}"] = df["stoch"] > level


# EMA relationships
ema_pairs = [(10,20),(20,50),(20,100),(50,200)]

for fast, slow in ema_pairs:

    fast_col = f"ema{fast}"
    slow_col = f"ema{slow}"

    if fast_col not in df:
        df[fast_col] = EMAIndicator(df["close"], window=fast).ema_indicator()

    if slow_col not in df:
        df[slow_col] = EMAIndicator(df["close"], window=slow).ema_indicator()

    signals[f"ema_bull_{fast}_{slow}"] = df[fast_col] > df[slow_col]
    signals[f"ema_bear_{fast}_{slow}"] = df[fast_col] < df[slow_col]


signals["bb_low"] = df["close"] < df["bb_low"]
signals["bb_high"] = df["close"] > df["bb_high"]

signals["macd_bull"] = df["macd"] > df["macd_signal"]
signals["macd_bear"] = df["macd"] < df["macd_signal"]


# -----------------------------
# SIGNAL COMBINATIONS
# -----------------------------

combo_signals = {}

names = list(signals.keys())

for i in range(len(names)):
    for j in range(i+1, len(names)):

        s1 = names[i]
        s2 = names[j]

        combo_name = f"{s1}__{s2}"
        combo_signals[combo_name] = signals[s1] & signals[s2]

signals.update(combo_signals)


# -----------------------------
# ADD REGIME FILTERS
# -----------------------------

filtered = {}

for name, cond in signals.items():

    filtered[name+"_trend"] = cond & df["trend_up"]
    filtered[name+"_highvol"] = cond & df["high_vol"]

signals.update(filtered)


# -----------------------------
# FUTURE RETURNS
# -----------------------------

horizon = 4
df["future_return"] = df["close"].shift(-horizon) / df["close"] - 1


# -----------------------------
# WALK-FORWARD SPLIT
# -----------------------------

split = int(len(df)*0.7)

train = df.iloc[:split]
test = df.iloc[split:]


# -----------------------------
# SIGNAL EVALUATION
# -----------------------------

results = []

for name, cond in signals.items():

    train_subset = train[cond[:split]]
    test_subset = test[cond[split:]]

    if len(train_subset) < 80 or len(test_subset) < 40:
        continue

    train_ret = train_subset["future_return"].dropna()
    test_ret = test_subset["future_return"].dropna()

    if len(train_ret) < 20 or len(test_ret) < 20:
        continue

    train_mean = train_ret.mean()
    test_mean = test_ret.mean()

    train_std = train_ret.std()

    sharpe_like = train_mean/train_std if train_std != 0 else 0

    stats = {
        "signal": name,
        "train_count": len(train_ret),
        "test_count": len(test_ret),
        "train_mean": train_mean,
        "test_mean": test_mean,
        "test_winrate": (test_ret > 0).mean(),
        "train_sharpe_like": sharpe_like
    }

    results.append(stats)


results = pd.DataFrame(results)

results = results.sort_values("test_mean", ascending=False)


# -----------------------------
# OUTPUT
# -----------------------------

print("\nTop discovered signals (walk-forward validated):\n")

print(results.head(30).to_string(index=False))
