import argparse
import pandas as pd
import numpy as np
import itertools

from ta.momentum import RSIIndicator, StochasticOscillator
from ta.trend import EMAIndicator, MACD
from ta.volatility import BollingerBands

# --------------------------------
# PARAMETERS
# --------------------------------

HOLD_BARS = 4
TRAIN_RATIO = 0.7
MIN_TRADES = 60

# --------------------------------
# LOAD DATA
# --------------------------------

parser = argparse.ArgumentParser()
parser.add_argument("csvfile")
args = parser.parse_args()

df = pd.read_csv(args.csvfile)

df["timestamp"] = pd.to_datetime(df["time"])
df = df.sort_values("timestamp")
df.set_index("timestamp", inplace=True)

# --------------------------------
# INDICATORS
# --------------------------------

close = df["close"]

df["rsi"] = RSIIndicator(close,14).rsi()

df["ema10"] = EMAIndicator(close,10).ema_indicator()
df["ema20"] = EMAIndicator(close,20).ema_indicator()
df["ema50"] = EMAIndicator(close,50).ema_indicator()
df["ema200"] = EMAIndicator(close,200).ema_indicator()

macd = MACD(close)
df["macd"] = macd.macd()
df["macd_signal"] = macd.macd_signal()

stoch = StochasticOscillator(df["high"],df["low"],close)
df["stoch"] = stoch.stoch()

bb = BollingerBands(close)
df["bb_low"] = bb.bollinger_lband()
df["bb_high"] = bb.bollinger_hband()

# volatility regime

df["vol"] = close.pct_change().rolling(24).std()
df["high_vol"] = df["vol"] > df["vol"].rolling(200).mean()

# --------------------------------
# FUTURE RETURNS
# --------------------------------

df["future"] = close.shift(-HOLD_BARS) / close - 1

# --------------------------------
# FEATURE LIBRARY
# --------------------------------

features = {}

features["rsi_lt_25"] = df["rsi"] < 25
features["rsi_lt_30"] = df["rsi"] < 30
features["rsi_lt_35"] = df["rsi"] < 35

features["rsi_gt_65"] = df["rsi"] > 65
features["rsi_gt_70"] = df["rsi"] > 70
features["rsi_gt_75"] = df["rsi"] > 75

features["stoch_lt_20"] = df["stoch"] < 20
features["stoch_gt_80"] = df["stoch"] > 80

features["ema10_gt_20"] = df["ema10"] > df["ema20"]
features["ema10_lt_20"] = df["ema10"] < df["ema20"]

features["ema20_gt_50"] = df["ema20"] > df["ema50"]
features["ema20_lt_50"] = df["ema20"] < df["ema50"]

features["ema50_gt_200"] = df["ema50"] > df["ema200"]
features["ema50_lt_200"] = df["ema50"] < df["ema200"]

features["macd_bull"] = df["macd"] > df["macd_signal"]
features["macd_bear"] = df["macd"] < df["macd_signal"]

features["bb_low"] = df["close"] < df["bb_low"]
features["bb_high"] = df["close"] > df["bb_high"]

features["high_vol"] = df["high_vol"]

feature_names = list(features.keys())

# --------------------------------
# DATA SPLIT
# --------------------------------

split = int(len(df)*TRAIN_RATIO)

train_idx = df.index[:split]
test_idx = df.index[split:]

# --------------------------------
# ALPHA GENERATION
# --------------------------------

combos = []

for k in range(1,4):
    combos.extend(itertools.combinations(feature_names,k))

results = []

# --------------------------------
# EVALUATION
# --------------------------------

for combo in combos:

    mask = pd.Series(True,index=df.index)

    for c in combo:
        mask &= features[c]

    trades = df.loc[mask]

    if len(trades) < MIN_TRADES:
        continue

    train = trades[trades.index < df.index[split]]
    test = trades[trades.index >= df.index[split]]

    # train = trades.loc[train_idx]
    # test = trades.loc[test_idx]

    if len(test) < MIN_TRADES/2:
        continue

    r_train = train["future"]
    r_test = test["future"]

    if r_train.std() == 0:
        continue

    sharpe = r_train.mean()/r_train.std()

    results.append({
        "signal":"__".join(combo),
        "train_count":len(r_train),
        "test_count":len(r_test),
        "train_mean":r_train.mean(),
        "test_mean":r_test.mean(),
        "test_winrate":(r_test>0).mean(),
        "train_sharpe":sharpe
    })

# --------------------------------
# RANKING
# --------------------------------

res = pd.DataFrame(results)

res = res.sort_values(
    ["test_mean","train_sharpe"],
    ascending=False
)

print("\nTop Alpha Signals\n")

print(res.head(30).to_string(index=False))
