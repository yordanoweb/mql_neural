import argparse
import pandas as pd
import numpy as np
import itertools

from ta.momentum import RSIIndicator, StochasticOscillator
from ta.trend import EMAIndicator, MACD
from ta.volatility import BollingerBands

HOLD = 4
TRAIN_SPLIT = 0.7
MIN_TRADES = 60


parser = argparse.ArgumentParser()
parser.add_argument("csvfile")
parser.add_argument("output_csv")
args = parser.parse_args()

df = pd.read_csv(args.csvfile)

df["timestamp"] = pd.to_datetime(df["time"])
df = df.sort_values("timestamp")
df.set_index("timestamp", inplace=True)

close = df["close"]


# -------------------------
# INDICATORS
# -------------------------

df["rsi"] = RSIIndicator(close,14).rsi()

stoch = StochasticOscillator(df["high"],df["low"],close)
df["stoch"] = stoch.stoch()

df["ema5"] = EMAIndicator(close,5).ema_indicator()
df["ema10"] = EMAIndicator(close,10).ema_indicator()
df["ema20"] = EMAIndicator(close,20).ema_indicator()
df["ema50"] = EMAIndicator(close,50).ema_indicator()
df["ema100"] = EMAIndicator(close,100).ema_indicator()
df["ema200"] = EMAIndicator(close,200).ema_indicator()

macd = MACD(close)
df["macd"] = macd.macd()
df["macd_signal"] = macd.macd_signal()

bb = BollingerBands(close)
df["bb_low"] = bb.bollinger_lband()
df["bb_high"] = bb.bollinger_hband()


df["vol"] = close.pct_change().rolling(24).std()
df["high_vol"] = df["vol"] > df["vol"].rolling(200).mean()

df["future"] = close.shift(-HOLD)/close - 1


# -------------------------
# FEATURE LIBRARY
# -------------------------

features = {}


# RSI grid
for t in [20,25,30,35,40]:
    features[f"rsi_lt_{t}"] = df["rsi"] < t

for t in [60,65,70,75,80]:
    features[f"rsi_gt_{t}"] = df["rsi"] > t


# STOCH grid
for t in [10,15,20,25]:
    features[f"stoch_lt_{t}"] = df["stoch"] < t

for t in [75,80,85,90]:
    features[f"stoch_gt_{t}"] = df["stoch"] > t


# EMA relationships
ema_pairs = [
    ("ema5","ema10"),
    ("ema10","ema20"),
    ("ema20","ema50"),
    ("ema50","ema100"),
    ("ema50","ema200"),
]

for a,b in ema_pairs:

    features[f"{a}_gt_{b}"] = df[a] > df[b]
    features[f"{a}_lt_{b}"] = df[a] < df[b]


# MACD
features["macd_bull"] = df["macd"] > df["macd_signal"]
features["macd_bear"] = df["macd"] < df["macd_signal"]


# Bollinger
features["bb_low"] = df["close"] < df["bb_low"]
features["bb_high"] = df["close"] > df["bb_high"]


# volatility regime
features["high_vol"] = df["high_vol"]


feature_names = list(features.keys())

print("Total base features:",len(feature_names))


# -------------------------
# TRAIN / TEST SPLIT
# -------------------------

split_idx = int(len(df)*TRAIN_SPLIT)
split_time = df.index[split_idx]


# -------------------------
# GENERATE SIGNALS
# -------------------------

combos = []

for k in range(1,4):
    combos.extend(itertools.combinations(feature_names,k))

print("Total candidate signals:",len(combos))


# -------------------------
# EVALUATION
# -------------------------

results = []

for combo in combos:

    mask = np.ones(len(df),dtype=bool)

    for c in combo:
        mask &= features[c].values

    idx = np.where(mask)[0]

    if len(idx) < MIN_TRADES:
        continue

    trades = df.iloc[idx]

    train = trades[trades.index < split_time]
    test = trades[trades.index >= split_time]

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
        "winrate":(r_test>0).mean(),
        "train_sharpe":sharpe
    })


# -------------------------
# RESULTS
# -------------------------

res = pd.DataFrame(results)

res = res.sort_values(
    ["test_mean","train_sharpe"],
    ascending=False
)

print("\nTop Signals\n")

_out = res.head(40)
print(_out.to_string(index=False))

# Exporting the DataFrame to a CSV file
_out.to_csv(args.output_csv, index=False)

