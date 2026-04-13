# Training Pipeline Spec

## Goal
Take raw OHLCV data → engineer features → train a binary classifier → export a valid ONNX model.

## Pipeline Steps

### 1. Data Ingestion
- Accept CSV or Parquet (`--input`)
- Required columns: `time, open, high, low, close, tick_volume`
- Parse `time` as datetime, sort ascending, drop NaNs

### 2. Feature Engineering  (`src/python/utils/features.py`)
- Compute indicators with the `ta` library
- Normalize to roughly `[0, 1]` or `[-1, 1]`
- Call `df.dropna(inplace=True)` after all indicators are computed
- Build sliding windows of shape `(n_samples, WINDOW_SIZE * N_FEATURES)` — row-major flatten

### 3. Label Generation
```python
future_close = df['close'].shift(-forward_bars)
df['label'] = ((future_close - df['close']) / df['close'] > min_pct).astype(int)
df.dropna(subset=['label'], inplace=True)
```
- Print class distribution before training
- If imbalance > 60/40: use `class_weight='balanced'`

### 4. Train / Test Split
- 80 / 20, **no shuffle** — preserve time order

### 5. Model Training
- Supported: `sklearn` (MLPClassifier, RandomForestClassifier), `xgboost.XGBClassifier`
- Print classification report on test set

### 6. ONNX Export  (`src/python/utils/onnx_export.py`)
- Input type: `FloatTensorType([None, WINDOW_SIZE * N_FEATURES])`
- Set `zipmap=False` for sklearn classifiers
- Required metadata keys (stored in every exported model, queryable via `query_onnx_model.py`):
  - `feature_names` — comma-separated, same order as window columns
  - `window_size` — integer as string
  - `n_features` — integer as string
- Save to `onnx/<symbol>_<timeframe>_<n>_feat[_<tag>].onnx`

### 7. Verification
- Run `onnxruntime` inference on one sample
- Assert `probabilities` output exists with shape `[*, 2]`
- sklearn exports two outputs: `label [None]` and `probabilities [None, 2]` — always read by name
- If only one class is present in labels, abort before export with a clear message

## Querying a Model
```bash
python src/python/query_onnx_model.py onnx/ndx100_m5_12_feat_adx_stoch_vol.onnx
```
Prints: input/output tensor shapes, all metadata keys, and the numbered feature list.

## Classification Target
- **2 classes**: `0 = sell`, `1 = buy`
- Label: price rises by `> min_pct` in the next `forward` bars
- Output: `float32[1, 2]` — `[P(sell), P(buy)]`

## Implemented Scripts
| Script | Features (n) | Groups |
|---|---|---|
| `train_adx_stoch_vol.py` | 12 | ADX (3) + Stochastic (4) + Volume (5) |

### ADX (3): `adx_norm`, `dip_norm`, `din_norm`
### Stochastic (4): `stoch_k`, `stoch_d`, `stoch_diff`, `stoch_signal`
### Volume (5): `vol_norm`, `vol_change`, `vol_ma_ratio`, `obv_norm`, `vol_spike`

## CLI Contract
Every `train_*.py` must accept:
```
--input        CSV or Parquet path
--symbol       symbol name (used in output filename)
--timeframe    M1 M5 M15 M30 H1 H4 D1
--model        mlp | rf  (default: mlp)
--window       window size (default: 20)
--forward      forward bars for label (default: 1)
--min_pct      minimum price move % to count as signal (default: 0.0)
--output       ONNX output path (auto-generated if omitted)
```
Indicator period args (per script, e.g. `--adx_period`, `--stoch_k`, `--vol_window`) are also accepted.

## ONNX Contract
| Property | Value |
|---|---|
| Input  | `float32[1, WINDOW_SIZE * N_FEATURES]` |
| Output | `float32[1, 2]` — `[P(sell), P(buy)]` |
| Metadata | `feature_names`, `window_size`, `n_features` |

## Verification (after every export)
- Run `onnxruntime` inference on one sample
- Assert output shape `[1, 2]` and values sum ≈ 1.0

## Out of Scope
- Hyperparameter tuning automation
- Walk-forward optimization

## See Also
- `docs/execution_script.md` — live inference + MT5 order execution
