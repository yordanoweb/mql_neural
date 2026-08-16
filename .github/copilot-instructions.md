# Copilot Instructions for mql_neural

## Project Overview
This repository trains ONNX models for market direction prediction (buy signals) using scikit-learn RandomForest classifiers. Models are exported with a specific binary classification format and consumed by MetaTrader 5 Expert Advisors (MQL5). External inference is performed by a downstream runtime—*not* built here.

**Core pipeline**: CSV historical data → feature engineering → train/export ONNX → consume in MQL5 EA

## Stack & Environment
- **Python**: 3.11, virtual environment at `.venv/`
- **Key libraries**: scikit-learn, xgboost, onnx, skl2onnx, onnxruntime, pandas, numpy, ta, yfinance
- **MQL5**: MetaTrader 5 (external execution; not compiled here without resource embedding)

## Repository Layout
```
src/
├── train_buy_only.py              # Primary training pipeline; exports ONNX
├── indicators.py                  # RSI + other indicator utilities
├── extract_rates_to_csv.py        # Fetch OHLC data from MT5
├── query_onnx_metadata.py         # Read/inspect model metadata
├── BuyOnly_ONNX_EA.mq5            # Buy-only trading EA
├── SellOnly_ONNX_EA.mq5           # Sell-only trading EA
└── Utils.mqh                      # Shared MQL5 utilities
csv/                               # Training data (OHLC)
onnx/                              # Exported models
```

## Quick Start: Running Training
```bash
python src/train_buy_only.py \
  --input-csv csv/EURUSD_M15.csv \
  --output-filename onnx/EURUSD_M15_buy_only.onnx \
  --window 20 \
  --forward 10 \
  --rsi-period 14
```

All CLI arguments default sensibly; run with `--help` to see them. Symbol and timeframe are auto-inferred from CSV filename if not specified.

## ONNX Contract (Sacred)
**Never break this.** External code depends on it.

- **Input shape**: `float32[1, window * 3]` — flattened OHLC window, row-major
- **Features (raw price, no pip normalization)**:
  - `feat_body = close - open`
  - `feat_range = high - low`
  - `feat_rsi = RSI(close, rsi_period) / 100.0`
- **Output shape**: `float32[1, 2]` — softmax probabilities `[P(no_buy), P(buy)]` (read by name, not index)
- **Target label**: `1` when `close[t + forward] > close[t]`, otherwise `0`
- **Required metadata** (queryable via `query_onnx_metadata.py`):
  - `training.{created_utc, input_csv_path, symbol, timeframe, window, forward, rsi_period, features, feature_count, input_size, n_iter, n_splits, n_jobs, records_loaded, records_after_dropna, samples_used, target, target_class_distribution, model_type, best_cv_score, best_params, train_prediction_distribution, cli_args}`

## Code Patterns & Rules

### Training Scripts
1. All `train_*.py` scripts must use `argparse.ArgumentParser` with `formatter_class=argparse.ArgumentDefaultsHelpFormatter`
2. Accept the standard CLI contract (see AGENTS.md for full list)
3. **Always print class distribution before training**
4. **Abort before export if labels have <2 classes**
5. After export, verify ONNX `probabilities` output is `[*, 2]` — read by name, not index
6. Preserve time order in train/test split — *no shuffle*
7. Store feature names, window size, and n_features in metadata
8. Every finished change must be committed and pushed immediately

### MQL5 EA Scripts
- Input contract: `float32[1, window * 3]` with raw-price body/range + RSI/100
- Output contract: `float32[1, 2]` softmax probabilities; enter BUY when `P(buy) >= InpMinConf`
- Mirror failover: on failed entry, runtime mirror state flips and persists until next failure
- Key inputs: `InpModelFile`, `InpMinConf`, `InpWindow`, `InpATR` params, time window, cooldown, movement filters
- Test mode (`InpTestMode=true`): bypasses confidence/spread/movement filters for order verification

### Naming Conventions
- ONNX files: `<symbol>_<timeframe>_<n>_feat[_<tag>].onnx`
- Training scripts: `train_<description>.py`

### Code Style
- Minimal, focused code — no abstractions that don't directly serve the pipeline
- Shared logic goes in `src/indicators.py` or (planned) `src/python/utils/`, never duplicated
- No comments unless clarifying; obvious code doesn't need explanation

## Feature Sets (Current)
| Script | Features (n) | Composition |
|---|---|---|
| `train_buy_only.py` | 3 | Price (2) + RSI (1) |

- **Price (2)**: `feat_body`, `feat_range`
- **RSI (1)**: `feat_rsi` — `RSI(close, rsi_period) / 100.0`

## Graft: Code Intelligence
This repo is indexed in `graft/` for fast semantic search and call graphs:
- **New to the repo?** Run `graft map` for orientation (dirs, hubs, hotspots)
- **Understand a feature?** `graft ask "<question>" --source` → ranked nodes with code spans
- **Find all callers?** `graft callers <symbol> --depth all` → blast radius
- **Exhaustive search?** `graft grep "<literal>"` → grouped by enclosing symbol
- After big changes: `graft build` (free, deterministic, no API key)

Browse `graft/INDEX.md` to explore the graph directly.

## Important: Regression Contract
This repo has a working Regression Contract documented in `docs/execution_script.md` (when it exists). Before making any change that would alter listed behavior:
1. Get explicit user approval for the change
2. Update the contract before committing

Never silently break working features.

## Documentation
- **README.md** — high-level project description, usage examples
- **AGENTS.md** — detailed AI-focused rules, metadata, CLI contracts (source of truth for contracts)
- **graft/INDEX.md** — semantic graph of systems and relationships
