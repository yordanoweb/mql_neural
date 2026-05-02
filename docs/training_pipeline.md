# Training Pipeline Spec

## Goal
Take raw OHLCV data → engineer features → train a binary classifier → export a valid ONNX model.

## Pipeline Steps

### 1. Data Ingestion
- Accept CSV or Parquet (`--input`)
- Required columns: `time, open, high, low, close, tick_volume`
- Parse `time` as datetime, sort ascending, drop NaNs

### 2. Feature Engineering  (`src/python/utils/features.py`)
- All features sanitized with `safe_series()` — replaces inf/NaN, clips to range
- Call `df.dropna(inplace=True)` after all indicators are computed
- Build sliding windows of shape `(n_samples, WINDOW_SIZE * N_FEATURES)` — row-major flatten
- Final `np.nan_to_num()` safety pass before fitting

### 3. Label Generation (ATR-based)
```python
for each bar i:
    upside   = (max(high[i+1..i+forward]) - close[i]) / ATR[i]
    downside = (close[i] - min(low[i+1..i+forward]))  / ATR[i]
    if upside >= min_profit_atr AND upside > downside:
        label[i] = 1   # buy
    elif downside >= min_profit_atr AND downside > upside:
        label[i] = 2   # sell
    else:
        label[i] = 0   # hold
```
- `--min_profit_atr` is in ATR units (e.g. 1.5 = price must move 1.5× ATR upward)
- Last `forward` rows are dropped (no future data)
- Print class distribution before training

### 4. Train / Test Split
- 80 / 20, **no shuffle** — preserve time order

### 5. Model Training
- RF: `RandomizedSearchCV` with `TimeSeriesSplit(n_splits=3)`, `class_weight='balanced'`
  - Scored on `balanced_accuracy`
  - `--n_iter` controls search budget, `--jobs` controls parallelism
- MLP: `Pipeline([StandardScaler, MLPClassifier])`, no hyperparam search
- Print classification report on test set

### 6. ONNX Export  (`src/python/utils/onnx_export.py`)
- Pass `best_estimator_` when exporting from `RandomizedSearchCV`
- Input type: `FloatTensorType([None, WINDOW_SIZE * N_FEATURES])`
- `zipmap=False`, `target_opset=17`
- `onnx.checker.check_model()` before saving
- Required metadata keys (queryable via `query_onnx_model.py`):
  - `feature_names` — comma-separated, same order as window columns
  - `window_size` — integer as string
  - `n_features` — integer as string
- Save to `onnx/<symbol>_<timeframe>_<n>_feat[_<tag>].onnx`

### 7. Verification
- Run `onnxruntime` inference on one sample
- Assert `probabilities` output exists with shape `[*, 2]`
- sklearn exports two outputs: `label [None]` and `probabilities [None, 2]` — always read by name

## Classification Target
- **3 classes**: `0 = hold`, `1 = buy`, `2 = sell`
- Label: ATR-based — over the next `forward` bars:
  - `1 (buy)`  — upside   >= `min_profit_atr × ATR` AND upside > downside
  - `2 (sell)` — downside >= `min_profit_atr × ATR` AND downside > upside
  - `0 (hold)` — otherwise
- Output: `float32[1, 3]` — `[P(hold), P(buy), P(sell)]`

## Implemented Scripts
| Script | Features (n) | Groups |
|---|---|---|
| `train_adx_stoch_vol.py` | 16 | Price (2) + ADX (5) + Stochastic (4) + Volume (5) |

### Price (2): `feat_body`, `feat_range` — normalised by ATR
### ADX (5): `adx_strength`, `adx_di_signal`, `adx_di_sep`, `adx_momentum`, `adx_regime`
### Stochastic (4): `stoch_momentum`, `stoch_position`, `stoch_velocity`, `stoch_divergence`
### Volume (5): `vol_ratio`, `vol_momentum`, `vol_price_div`, `vol_percentile`, `vol_zscore`

## CLI Contract
Every `train_*.py` must accept:
```
--input          CSV or Parquet path
--symbol         symbol name (used in output filename)
--timeframe      M1 M5 M15 M30 H1 H4 D1
--model          mlp | rf  (default: rf)
--window         window size (default: 20)
--forward        forward bars for label (default: 10)
--min_profit_atr minimum upside in ATR units to label as buy (default: 1.5)
--output         ONNX output path (auto-generated if omitted)
--date_col       column name for date (if separate from time)
--time_col       column name for time (or datetime if combined)
--open_col       column name for open price (default: open)
--high_col       column name for high price (default: high)
--low_col        column name for low price (default: low)
--close_col      column name for close price (default: close)
--volume_col     column name for volume (default: tick_volume)
```
Indicator period args: `--atr_period`, `--adx_period`, `--adx_min`, `--stoch_k`, `--stoch_d`, `--vol_window`
RF-only args: `--n_iter`, `--jobs`

## ONNX Contract
| Property | Value |
|---|---|
| Input  | `float32[1, WINDOW_SIZE * N_FEATURES]` |
| Output | `float32[1, 3]` — `[P(hold), P(buy), P(sell)]` |
| Metadata | `feature_names`, `window_size`, `n_features` |

## Querying a Model
```bash
python src/python/query_onnx_model.py onnx/ustec_m5_16_feat_adx_stoch_vol.onnx
```
Prints: input/output tensor shapes, all metadata keys, and the numbered feature list.

## See Also
- `docs/execution_script.md` — live inference + MT5 order execution

## Maintenance Rule
After every implementation, feature addition, bug fix, or test: update this doc and `docs/execution_script.md` to reflect the current behaviour before committing.
