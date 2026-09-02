# mql_neural

## Scripts

### `src/train_buy_only.py`
Train a buy-only RandomForest classifier and export it as an ONNX model.

**Output contract:** binary `float32[1, 2]` → `[P(no_buy), P(buy)]`. The second probability is the buy signal.

**Target label:** `close[t + forward] > close[t]` → `1 (buy)`, otherwise `0 (no_buy)`.

**Features (raw price, no pip-unit normalization):**
- `feat_body` = `close - open`
- `feat_range` = `high - low`
- `feat_rsi` = `RSI(close, rsi_period) / 100.0`

**Usage example:**
```bash
python src/train_buy_only.py \
  --input-csv csv/SYMBOL_M15.csv \
  --output-filename onnx/SYMBOL_M15_buy_only.onnx \
  --window 20 \
  --forward 10 \
  --rsi-period 14
```

### `src/train_ensemble_buy_only.py`
Train a buy-only ensemble of 5 RandomForest models with out-of-sample (OOS) validation and export each as a separate ONNX model. Each model uses a different window/feature-set perspective.

**Output contract:** 5 ONNX files + ensemble report + threshold sweep. Each ONNX is `float32[1, 2]` → `[P(no_buy), P(buy)]`.

**Models:**
| ID | Alias | Window | Feature Set | Perspective | inc_percent |
|----|-------|--------|-------------|-------------|-------------|
| A | impulse | 5 | standard | microstructure | 0.3 |
| B | swing | 15 | standard | short_term | 0.5 |
| C | trend | 30 | standard | medium_term | 1.0 |
| D | structure | 20 | structure | scale_invariant | 0.5 |
| E | volatility | 20 | volatility | volatility_regime | 0.5 |

**Feature sets:**
- `standard`: `feat_body`, `feat_range`, `feat_rsi`
- `structure`: `feat_body_ratio`, `feat_range_norm` (Wilder ATR / MT5 `iATR`), `feat_rsi`
- `volatility`: `feat_range`, `feat_range_expansion`, `feat_rsi`

**Target label:** `max(high[t+1..t+forward]) > close[t] * (1 + inc_percent/100)` → `1 (buy)`, otherwise `0 (no_buy)`.

**Auto-scoring:** If minority class < 5%, scoring switches to `average_precision` (PR-AUC); otherwise `balanced_accuracy`.

**Usage example (full ensemble):**
```bash
python src/train_ensemble_buy_only.py \
  --input-csv csv/SYMBOL_M15.csv \
  --output-dir ./onnx \
  --model-id all \
  --forward 10 \
  --inc-percent 0.5 \
  --n-splits 5
```

**Usage example (single model):**
```bash
python src/train_ensemble_buy_only.py \
  --input-csv csv/SYMBOL_M15.csv \
  --output-dir ./onnx \
  --model-id D \
  --feature-set structure \
  --window 20
```

**Outputs:**
- `model_A_impulse_SYMBOL.onnx` ... `model_E_volatility_SYMBOL.onnx`
- `ensemble_report_buy_only.json`
- `ensemble_threshold_report_buy_only.json`
- `ensemble_thresholds_buy_only.csv`

### `src/BuyOnly_ONNX_EA.mq5`
MetaTrader 5 Expert Advisor that trades only long positions using the ONNX model from `train_buy_only.py`.

**Input contract:** `float32[1, window * 3]` with raw-price body/range + RSI/100.

**Output contract:** `float32[1, 2]` → `[P(no_buy), P(buy)]`. The EA enters a **BUY** when `P(buy) >= InpMinConf`.

**Key inputs:**
- `InpModelFile` — ONNX filename in `MQL5/Files/`
- `InpMinConf` — minimum buy probability to enter (default `0.62`)
- `InpWindow` — must match the training `--window`
- `InpATR` / `InpMultiplier` — SL/TP distance = ATR * multiplier
- `InpStartHour` / `InpEndHour` — trading time window
- `InpCooldownBars` — bars to wait after a close
- `InpMinBodyATR` / `InpMinRangeATR` / `InpMinBodyRatio` — movement-strength filters
- `InpMaxSpreadATRRatio` — spread/ATR filter
- `InpTestMode` — when `true`, bypasses confidence/spread/movement filters so you can verify order execution (still respects time window and one open position)

**Behavior:**
- Uses ONNX inference on `OnTimer` and keeps time/cooldown/candle-direction/volatility entry protections.
- Supports mirrored execution side through mirror mode (normal BUY vs mirrored SELL for buy-signal entries).
- Keeps optional early close on `OnTick` when open-position net profit reaches `InpMinDollars`.
- Before sending each order, SL/TP are taken directly from ATR-based price levels in the EA (`GetStopLoss` / `GetTakeProfit`) and sent in the entry order request (no retry loop / open-then-modify path).

### Mirror failover behavior (`BuyOnly_ONNX_EA.mq5` and `SellOnly_ONNX_EA.mq5`)
- `InpMirrorEntryOperation` is used as the startup mirror mode.
- Each EA keeps a runtime mirror state and uses it to decide BUY/SELL execution side.
- On every failed entry order (`m_trade.Buy/Sell` returns `false`), the runtime mirror state is flipped and remains active for subsequent entries until another failed entry flips it again.
- The chart comment displays both configured input mirror mode and current runtime mirror mode.

For backtesting, embed the ONNX model as a resource and rename the `#resource` directive at the top of the file.

### `src/EnsembleBuyEA.mq5`
MetaTrader 5 Expert Advisor that trades using the 5-model ONNX ensemble produced by `train_ensemble_buy_only.py`.

**Input contract:** same as training — `float32[1, window * 3]` per model, with model-specific feature sets.

**Output contract:** `float32[1, 2]` per model → `[P(no_buy), P(buy)]`.

**Aggregation modes:**
- `ENSEMBLE_MEAN` — simple mean of buy probabilities
- `ENSEMBLE_WEIGHTED` — weighted mean (configurable per-model weights)
- `ENSEMBLE_MEDIAN` — median of buy probabilities
- `ENSEMBLE_MAJORITY` — majority vote (prob > 0.5 counts as buy)
- `ENSEMBLE_TRIMMEAN` — trimmed mean (drop min/max)

**Key inputs:**
- `InpModelA_Path` ... `InpModelE_Path` — ONNX filenames in `MQL5/Files/`
- `InpEnsembleMode` — aggregation mode
- `InpWeightA` ... `InpWeightE` — per-model weights for weighted mode
- `InpConfidenceThreshold` — minimum ensemble buy probability to enter
- `InpMinConfidenceDiff` — minimum difference between buy and sell probabilities
- `InpMaximumRisk` / `InpDecreaseFactor` — position sizing
- `InpSL_ATR_Mult` / `InpTP_ATR_Mult` — SL/TP = ATR * multiplier
- `InpMaxPositions` — max concurrent positions
- `InpRSI_Period` / `InpATR_Period` — indicator periods (ATR used by model D)

**Behavior:**
- Loads all 5 ONNX models on init. On each new bar, computes features for each model independently and runs inference.
- Aggregates probabilities according to selected mode and decides whether to open a BUY.
- Telegram notifications on trade open/close/SL/TP.
- Saves EA inputs to timestamped `.set` files.
