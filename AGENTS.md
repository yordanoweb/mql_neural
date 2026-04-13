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
- Output: `float32[1, 2]` — softmax probabilities `[P(sell), P(buy)]`
- Metadata key `feature_names`: comma-separated feature column names (required)
- Metadata keys `window_size`, `n_features`: integers as strings

## Naming Conventions
- ONNX files: `<symbol>_<timeframe>_<n>_feat[_<tag>].onnx`
- Training scripts: `train_<description>.py`

## Implemented Feature Sets
| Script | Features (n) | Groups |
|---|---|---|
| `train_adx_stoch_vol.py` | 12 | ADX (3) + Stochastic (4) + Volume (5) |

### ADX (3): `adx_norm`, `dip_norm`, `din_norm`
### Stochastic (4): `stoch_k`, `stoch_d`, `stoch_diff`, `stoch_signal`
### Volume (5): `vol_norm`, `vol_change`, `vol_ma_ratio`, `obv_norm`, `vol_spike`

## Classification Target
- **2 classes**: `0 = sell`, `1 = buy`
- Label: price rises by `> min_pct` in the next `forward` bars
- Output: `float32[1, 2]` — `[P(sell), P(buy)]`

## CLI Contract
Every `train_*.py` script must accept:
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
Indicator period args (per script, e.g. `--adx_period`, `--stoch_k`, `--vol_window`) are also required.

## Execution Script Contract
Every `execute_onnx_<tag>_on_mt5.py` script must accept:
```
--model       path to ONNX file
--symbol      MT5 symbol (e.g. NAS100)
--timeframe   M1 M5 M15 M30 H1 H4 D1
--window      window size — must match training (default: 20)
--confidence  minimum probability to place an order (default: 0.60)
--lot         order lot size (default: 1.0)
--interval    seconds between inference cycles (default: 60)
```
Indicator period args must match those used at training time.

## Code Rules
- Minimal code — no abstractions that don't directly serve the pipeline
- No shuffle on train/test split — preserve time order
- Always print class distribution before training
- Always verify ONNX output shape `[1, 2]` after export
- Shared logic goes in `src/python/utils/`, not duplicated across scripts
- Every exported ONNX **must** store `feature_names`, `window_size`, `n_features` in metadata — queryable via `query_onnx_model.py`
- All `argparse.ArgumentParser` instances must use `formatter_class=argparse.ArgumentDefaultsHelpFormatter`
