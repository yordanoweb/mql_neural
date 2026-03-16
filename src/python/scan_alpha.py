import argparse
import pandas as pd
import numpy as np
import itertools

from ta.momentum import RSIIndicator, StochasticOscillator
from ta.trend import EMAIndicator, MACD
from ta.volatility import BollingerBands

# -------------------------
# CONFIG
# -------------------------

HOLD_BARS = 4
MIN_TRADES = 80

# -------------------------
# LOAD CSV
# -------------------------

parser = argparse.ArgumentParser()
parser.add_argument("csvfile")

args = parser.parse_args()

df = pd.read_csv(args.csvfile)

df["timestamp"] = pd.to_datetime(df["time"])
df = df.sort_values("timestamp")
df.set_index("timestamp", inplace=True)

# -------------------------
# INDICATORS
# -------------------------

df["rsi"] = RSIIndicator(df["close"], 14).rsi()

df["ema10"] = EMAIndicator(df["close"], 10).ema_indicator()
df["ema20"] = EMAIndicator(df["close"], 20).ema_indicator()
df["ema50"] = EMAIndicator(df["close"], 50).ema_indicator()
df["ema200"] = EMAIndicator(df["close"], 200).ema_indicator()

macd = MACD(df["close"])
df["macd"] = macd.macd()
df["macd_signal"] = macd.macd_signal()

stoch = StochasticOscillator(df["high"], df["low"], df["close"])
df["stoch"] = stoch.stoch()

bb = BollingerBands(df["close"])
df["bb_low"] = bb.bollinger_lband()
df["bb_high"] = bb.bollinger_hband()

# volatility regime
df["vol"] = df["close"].pct_change().rolling(24).std()
df["high_vol"] = df["vol"] > df["vol"].rolling(200).mean()

# future return
df["future_return"] = df["close"].shift(-HOLD_BARS) / df["close"] - 1

# -------------------------
# BASE SIGNAL BUILDERS
# -------------------------

signals = {}

signals["rsi_buy_30"] = df["rsi"] < 30
signals["rsi_buy_35"] = df["rsi"] < 35

signals["rsi_sell_70"] = df["rsi"] > 70
signals["rsi_sell_75"] = df["rsi"] > 75

signals["stoch_buy_20"] = df["stoch"] < 20
signals["stoch_sell_80"] = df["stoch"] > 80

signals["ema_bull_20_50"] = df["ema20"] > df["ema50"]
signals["ema_bear_20_50"] = df["ema20"] < df["ema50"]

signals["ema_bull_50_200"] = df["ema50"] > df["ema200"]
signals["ema_bear_50_200"] = df["ema50"] < df["ema200"]

signals["macd_bull"] = df["macd"] > df["macd_signal"]
signals["macd_bear"] = df["macd"] < df["macd_signal"]

signals["bb_low"] = df["close"] < df["bb_low"]
signals["bb_high"] = df["close"] > df["bb_high"]

signals["high_vol"] = df["high_vol"]

names = list(signals.keys())

# -------------------------
# GENERATE COMBINATIONS
# -------------------------

combos = []

for r in range(1,4):
    combos.extend(itertools.combinations(names,r))

results = []

# -------------------------
# WALK FORWARD SPLIT
# -------------------------

split = int(len(df)*0.7)

train = df.iloc[:split]
test = df.iloc[split:]

# -------------------------
# TEST SIGNALS
# -------------------------

for combo in combos:

    cond_train = pd.Series(True,index=train.index)
    cond_test = pd.Series(True,index=test.index)

    for c in combo:

        cond_train &= signals[c].loc[train.index]
        cond_test &= signals[c].loc[test.index]

    r_train = train.loc[cond_train,"future_return"]
    r_test = test.loc[cond_test,"future_return"]

    if len(r_train) < MIN_TRADES:
        continue

    if len(r_test) < MIN_TRADES/2:
        continue

    train_mean = r_train.mean()
    test_mean = r_test.mean()

    winrate = (r_test > 0).mean()

    sharpe = r_train.mean()/r_train.std()

    results.append({
        "signal":"__".join(combo),
        "train_count":len(r_train),
        "test_count":len(r_test),
        "train_mean":train_mean,
        "test_mean":test_mean,
        "test_winrate":winrate,
        "train_sharpe_like":sharpe
    })

# -------------------------
# OUTPUT
# -------------------------

res = pd.DataFrame(results)

res = res.sort_values("test_mean",ascending=False)

print("\nTop discovered signals:\n")

print(res.head(30).to_string(index=False))
