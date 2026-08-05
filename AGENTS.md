# AGENTS.md — AI Agent Instructions

## Project
- Python scripts that train ONNX models for market direction prediction.
- Meta Trader 5 Expert Advisor scripts that execute trades based on ONNX model predictions.
- Models are consumed externally by an inference runtime (not built here).

## Stack
- Python 3.11, venv at `.venv/`
- Libs: scikit-learn, xgboost, onnx, skl2onnx, onnxmltools, onnxruntime, pandas, numpy, ta, yfinance, pyarrow

## Layout
```
AGENTS.md
README.md
requirements.txt
csv/
data/
log/
onnx/
src/
  BuyOnly_ONNX_EA.mq5
  extract_rates_to_csv.py
  indicators.py
  PrintSymbolPipInfo.mq5
  query_onnx_metadata.py
  recompile_mql.py
  SellOnly_ONNX_EA.mq5
  SimpleONNX_EA_M15.mq5
  smoke_test_onnx_training_balance.py
  test_mt5_connection.py
  train_buy_only.py
  train_onnx_from_csv.py
  train_sell_only.py
  verify_mt5_data_flow.py
tmp/
```

## ONNX Contract (never break this)
- Input: `float32[1, window * 3]` — flattened OHLC window, row-major
- Features:
  - `feat_body = close - open`
  - `feat_range = high - low`
  - `feat_rsi = RSI(close, rsi_period) / 100.0`
- Output: `float32[1, 2]` — softmax probabilities `[P(no_buy), P(buy)]`
- Target: `1` when `close[t + forward] > close[t]`, otherwise `0`
- Required metadata: `training.created_utc`, `training.input_csv_path`, `training.input_csv_name`, `training.output_filename`, `training.symbol`, `training.timeframe`, `training.window`, `training.forward`, `training.rsi_period`, `training.features`, `training.feature_count`, `training.input_size`, `training.n_iter`, `training.n_splits`, `training.n_jobs`, `training.records_loaded`, `training.records_after_dropna`, `training.samples_used`, `training.target`, `training.target_class_distribution`, `training.model_type`, `training.random_state`, `training.param_dist`, `training.best_params`, `training.best_cv_score`, `training.scoring`, `training.train_prediction_distribution`, `training.cli_args`

## Naming Conventions
- ONNX files: `<symbol>_<timeframe>_<n>_feat[_<tag>].onnx`
- Training scripts: `train_<description>.py`

## Implemented Feature Sets
| Script | Features (n) | Groups |
|---|---|---|
| `train_buy_only.py` | 3 | Price (2) + RSI (1) |

### Price (2): `feat_body`, `feat_range`
### RSI (1): `feat_rsi` — `RSI(close, rsi_period) / 100.0`

## Classification Target
- **2 classes**: `0 = no_buy`, `1 = buy`
- Label: `close[t + forward] > close[t]` → `1 (buy)`, otherwise `0 (no_buy)`
- Output: `float32[1, 2]` — `[P(no_buy), P(buy)]`

## CLI Contract
Every `train_*.py` script must accept:
```
--input-csv      input CSV file path
--output-filename ONNX output filename/path
--window         feature window size in bars (default: 20)
--forward        forward bars used for buy label (default: 10)
--rsi-period     RSI period (default: 14)
--n-iter         RandomizedSearchCV iterations (default: 5)
--n-splits       TimeSeriesSplit folds (default: 2)
--n-jobs         parallel jobs for search (default: -1)
--symbol         trading symbol for metadata (default: auto)
--timeframe      timeframe for metadata (default: auto)
--date-col       column name for date (default: date)
--time-col       column name for time (default: time)
--open-col       column name for open price (default: open)
--high-col       column name for high price (default: high)
--low-col        column name for low price (default: low)
--close-col      column name for close price (default: close)
--volume-col     column name for volume (default: tick_volume)
```

## Execution Script Contract
`train_buy_only.py` does not define any execution-side CLI contract.

## Code Rules
- Minimal code — no abstractions that don't directly serve the pipeline
- No shuffle on train/test split — preserve time order
- Always print class distribution before training
- Always verify ONNX `probabilities` output shape `[*, 2]` after export — read by name, not index
- sklearn exports two outputs (`label`, `probabilities`) — always use `probabilities` for inference
- Abort training before export if labels contain fewer than 2 classes
- Shared logic goes in `src/python/utils/`, not duplicated across scripts
- Every exported ONNX **must** store `feature_names`, `window_size`, `n_features` in metadata — queryable via `query_onnx_model.py`
- All `argparse.ArgumentParser` instances must use `formatter_class=argparse.ArgumentDefaultsHelpFormatter`
- **After every implementation, feature addition, bug fix, or test: update `docs/execution_script.md` and/or `docs/training_pipeline.md` to reflect the current behaviour before committing**
- **Never break working features** — see the Regression Contract in `docs/execution_script.md`. Any change that would alter a listed behaviour must be explicitly requested and the contract updated accordingly.
- **Any finished change must be committed and pushed to the current branch immediately**
