# AGENTS.md — AI Agent Instructions

## Project
Python scripts that train ONNX models for market direction prediction.
Models are consumed externally by an inference runtime (not built here).

## Stack
- Python 3.11, venv at `.venv/`
- Libs: scikit-learn, xgboost, onnx, skl2onnx, onnxmltools, onnxruntime, pandas, numpy, ta, yfinance, pyarrow

## Layout
```
src/python/
  utils/
    features.py                          # feature engineering helpers
    onnx_export.py                       # ONNX export + metadata helpers
  train_<tag>.py                         # one script per model variant
  execute_onnx_<tag>_on_mt5.py          # live inference + MT5 order execution
  extract_rates_to_csv.py               # pull OHLCV from MT5
  extract_yfinance_rates_to_csv.py      # pull OHLCV from yfinance
  query_onnx_model.py                   # inspect ONNX metadata
csv/                  # raw OHLCV CSVs
data/<SYMBOL>/        # parquet files per symbol/timeframe
onnx/                 # exported ONNX models
docs/                 # specs and design docs
```

## ONNX Contract (never break this)
- Input : `float32[1, WINDOW_SIZE * N_FEATURES]` — flattened window, row-major
- Output: `float32[1, 3]` — softmax probabilities `[P(hold), P(buy), P(sell)]`
- Metadata key `feature_names`: comma-separated feature column names (required)
- Metadata keys `window_size`, `n_features`: integers as strings

## Naming Conventions
- ONNX files: `<symbol>_<timeframe>_<n>_feat[_<tag>].onnx`
- Training scripts: `train_<description>.py`

## Implemented Feature Sets
| Script | Features (n) | Groups |
|---|---|---|
| `train_adx_stoch_vol.py` | 16 | Price (2) + ADX (5) + Stochastic (4) + Volume (5) |

### Price (2): `feat_body`, `feat_range` — normalised by ATR
### ADX (5): `adx_strength`, `adx_di_signal`, `adx_di_sep`, `adx_momentum`, `adx_regime`
### Stochastic (4): `stoch_momentum`, `stoch_position`, `stoch_velocity`, `stoch_divergence`
### Volume (5): `vol_ratio`, `vol_momentum`, `vol_price_div`, `vol_percentile`, `vol_zscore`

## Classification Target
- **3 classes**: `0 = hold`, `1 = buy`, `2 = sell`
- Label: ATR-based — over the next `forward` bars:
  - `1 (buy)`  — upside   >= `min_profit_atr × ATR` AND upside > downside
  - `2 (sell)` — downside >= `min_profit_atr × ATR` AND downside > upside
  - `0 (hold)` — otherwise
- Output: `float32[1, 3]` — `[P(hold), P(buy), P(sell)]`

## CLI Contract
Every `train_*.py` script must accept:
```
--input          CSV or Parquet path
--symbol         symbol name (used in output filename)
--timeframe      M1 M5 M15 M30 H1 H4 D1
--model          mlp | rf  (default: rf)
--window         window size (default: 20)
--forward        forward bars for label (default: 10)
--min_profit_atr minimum upside in ATR units to label as buy (default: 1.5)
--output         ONNX output path (auto-generated if omitted)
```
Indicator period args: `--atr_period`, `--adx_period`, `--adx_min`, `--stoch_k`, `--stoch_d`, `--vol_window`
RF-only args: `--n_iter` (RandomizedSearchCV iterations), `--jobs` (parallel jobs)

## Execution Script Contract
Every `execute_onnx_<tag>_on_mt5.py` script must accept:
```
--model       path to ONNX file
--symbol      MT5 symbol (e.g. NAS100)
--timeframe   M1 M5 M15 M30 H1 H4 D1
--window      window size — must match training (default: 20)
--confidence  minimum probability to open a trade (default: 0.60)
--lot         order lot size (default: 1.0)
--interval    seconds between inference cycles (default: 60)
--atr_period  ATR period for SL/TP (default: 14)
--sl_mult     SL = ATR × sl_mult (default: 1.5)
--tp_mult     imaginary TP = ATR × tp_mult (default: 2.0)
```
Indicator period args must match those used at training time.
Exit logic: hard SL on broker + imaginary TP tracked in Python → trailing exit on first opposite M1 candle.

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
