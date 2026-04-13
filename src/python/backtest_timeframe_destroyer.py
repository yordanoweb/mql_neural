import numpy as np
import pandas as pd
import onnxruntime as ort
import matplotlib.pyplot as plt
import argparse
import ta

# ---------------- ARGUMENTS ----------------
parser = argparse.ArgumentParser()
parser.add_argument("--onnx_model", type=str, required=True)
parser.add_argument("--input_csv", type=str, required=True)
parser.add_argument("--window", type=int, default=20)
parser.add_argument("--start_hour", type=int, default=20)
parser.add_argument("--end_hour", type=int, default=21)
parser.add_argument("--atr_period", type=int, default=14)

args = parser.parse_args()

# ATR-based thresholds (target definition)
ATR_THRESHOLD = 1.4
DOMINANCE_THRESHOLD = 0.6
CLOSE_CONFIRM = 0.3

# ---------------- LOAD MODEL ----------------
session = ort.InferenceSession(args.onnx_model)
input_name = session.get_inputs()[0].name

# ---------------- LOAD DATA ----------------
df = pd.read_csv(args.input_csv)

if 'time' in df.columns:
    df['datetime'] = pd.to_datetime(df['time'])
elif 'datetime' in df.columns:
    df['datetime'] = pd.to_datetime(df['datetime'])
else:
    raise ValueError("CSV must have 'time' or 'datetime'")

df.sort_values('datetime', inplace=True)

# ---------------- ATR ----------------
atr_indicator = ta.volatility.AverageTrueRange(
    high=df['high'], low=df['low'], close=df['close'], window=args.atr_period
)
df['atr'] = atr_indicator.average_true_range()

# ---------------- FEATURES ----------------
df['feat_body'] = (df['close'] - df['open']) / df['atr']
df['feat_range'] = (df['high'] - df['low']) / df['atr']

df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.dropna(inplace=True)

# ---------------- BUILD SAMPLES ----------------
X, y = [], []
pnl_targets = []  # guardamos movimiento real en ATR

df['date'] = df['datetime'].dt.date

for date, day_df in df.groupby('date'):

    window_df = day_df[
        (day_df['datetime'].dt.hour >= args.start_hour) &
        (day_df['datetime'].dt.hour < args.end_hour)
    ]

    if len(window_df) < 2:
        continue

    pre_window = day_df[day_df['datetime'] < window_df['datetime'].iloc[0]]

    if len(pre_window) < args.window:
        continue

    entry_idx = pre_window.index[-1]
    entry_price = df.loc[entry_idx, 'close']
    atr = df.loc[entry_idx, 'atr']

    if atr == 0 or np.isnan(atr):
        continue

    # -------- TARGET --------
    max_high = window_df['high'].max()
    min_low = window_df['low'].min()
    close_price = window_df['close'].iloc[-1]

    max_up = (max_high - entry_price) / atr
    max_down = (entry_price - min_low) / atr
    close_move = (close_price - entry_price) / atr

    label = 0

    if (
        max_up >= ATR_THRESHOLD and
        max_up > max_down and
        abs(max_up - max_down) >= DOMINANCE_THRESHOLD and
        close_move >= CLOSE_CONFIRM
    ):
        label = 1

    elif (
        max_down >= ATR_THRESHOLD and
        max_down > max_up and
        abs(max_up - max_down) >= DOMINANCE_THRESHOLD and
        -close_move >= CLOSE_CONFIRM
    ):
        label = 2

    elif (
        max_up >= ATR_THRESHOLD and
        max_down >= ATR_THRESHOLD and
        abs(max_up - max_down) < DOMINANCE_THRESHOLD
    ):
        continue

    # -------- FEATURES --------
    window_data = df.loc[entry_idx - args.window + 1: entry_idx][
        ['feat_body', 'feat_range']
    ].values.flatten()

    if len(window_data) != args.window * 2:
        continue

    X.append(window_data)
    y.append(label)

    # Guardamos movimiento real para PnL
    pnl_targets.append((max_up, max_down))

X = np.array(X, dtype=np.float32)
y = np.array(y)

X = X.reshape(1, -1)

# ---------------- INFERENCE ----------------
outputs = session.run(None, {input_name: X})
probs = outputs[1] if len(outputs) == 2 else outputs[0]

# ---------------- THRESHOLD ANALYSIS ----------------
thresholds = np.arange(0.50, 0.91, 0.02)

results = []

for t in thresholds:
    trades = 0
    correct = 0
    pnl_list = []

    for i in range(len(probs)):
        p = probs[i]
        pred_class = np.argmax(p)

        if pred_class == 0:
            continue

        confidence = p[pred_class]

        if confidence < t:
            continue

        trades += 1

        # -------- PnL --------
        max_up, max_down = pnl_targets[i]

        if pred_class == 1:  # LONG
            pnl = max_up
        else:  # SHORT
            pnl = max_down

        # penalización si fallas dirección
        if pred_class != y[i]:
            pnl = -pnl

        pnl_list.append(pnl)

        if pred_class == y[i]:
            correct += 1

    if trades > 0:
        acc = correct / trades
        total_pnl = np.sum(pnl_list)
        avg_pnl = np.mean(pnl_list)
        sharpe = np.mean(pnl_list) / (np.std(pnl_list) + 1e-9)
    else:
        acc = total_pnl = avg_pnl = sharpe = 0

    results.append((t, trades, acc, total_pnl, avg_pnl, sharpe))

# ---------------- PRINT ----------------
print("\nThreshold | Trades | Acc | TotalPnL | AvgPnL | Sharpe")
for r in results:
    print(f"{r[0]:.2f} | {r[1]:5d} | {r[2]:.3f} | {r[3]:7.2f} | {r[4]:.3f} | {r[5]:.3f}")

# ---------------- BEST THRESHOLD ----------------
best_pnl = max(results, key=lambda x: x[3])
best_sharpe = max(results, key=lambda x: x[5])

print("\nBEST BY TOTAL PnL:", best_pnl)
print("BEST BY SHARPE:", best_sharpe)

# ---------------- PLOT ----------------
threshold_vals = [r[0] for r in results]
pnl_vals = [r[3] for r in results]
acc_vals = [r[2] for r in results]

plt.figure()
plt.plot(threshold_vals, pnl_vals, label="Total PnL (ATR)")
plt.plot(threshold_vals, acc_vals, label="Accuracy")
plt.xlabel("Threshold")
plt.ylabel("Value")
plt.title("Profit vs Threshold")
plt.legend()
plt.grid()
plt.show()
