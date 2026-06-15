# Copilot Instructions for `mql_neural`

## Build, test, and lint commands

```bash
# Install dependencies
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Full test run (current repo test suite)
cd tests && bash run_tests.sh

# Run a single test function
python -c "from tests.test_profit_lock import test_breakeven_logic; test_breakeven_logic()"

# Lint/security checks (configured pre-commit hooks)
pre-commit run --all-files
# or just the configured hook:
pre-commit run gitleaks --all-files
```

## High-level architecture

1. **Data sources and preparation**
   - Data extraction scripts under `src/python/` pull OHLCV and write CSV/Parquet used by training and backtesting.
   - Training data commonly lives in `csv/` (raw) and `data/<SYMBOL>/` (parquet).

2. **Shared feature + ONNX contract layer**
   - `src/python/utils/features.py` is the shared feature-engineering source for training, live execution, and backtesting.
   - `src/python/utils/onnx_export.py` owns model export and ONNX contract checks/metadata injection.
   - `src/python/query_onnx_model.py` is the metadata/introspection utility for exported models.

3. **Training path**
   - `src/python/train_*.py` scripts build labels, compute feature windows, train (`rf` / `mlp` / `xgb`), and export ONNX to `onnx/`.
   - The implemented feature set here is `train_adx_stoch_vol.py` (16-feature windowed input).

4. **Execution path**
   - `src/python/execute_onnx_adx_stoch_vol_on_mt5.py` runs a polling loop: fetch candles -> build same features -> ONNX inference -> risk filters -> MT5 order actions.
   - State is tracked in-memory (`TradeState`), and events are persisted to `trades.csv`.
   - `src/python/backtest_onnx.py` mirrors execution logic on historical data for offline validation.

## Key conventions in this codebase

1. **Copilot response style in this repo**
   - Code only by default.
   - Use bullets over paragraphs.
   - Do not add explanations unless explicitly asked.

2. **ONNX I/O and metadata are strict contracts**
   - Input shape is flattened window: `float32[1, window_size * n_features]`.
   - Inference reads the `probabilities` output by **name** (not index).
   - Required metadata keys on exported models: `feature_names`, `window_size`, `n_features`.
   - Training scripts also embed CLI/training metadata (symbol, timeframe, indicator periods, etc.).

3. **Feature ordering is coupled across train/execute/backtest**
   - `FEATURE_COLS` order is part of the model contract; keep it identical anywhere model input is built.
   - Any indicator-period or window changes must stay aligned between training and execution scripts.

4. **Time-series training behavior is preserved**
   - No shuffle on train/test split.
   - Class distribution is printed before fitting.
   - Training aborts export when labels have fewer than two classes.

5. **CLI behavior is standardized**
   - Script CLIs consistently use `argparse.ArgumentDefaultsHelpFormatter`.
   - Training and execution scripts are expected to maintain the argument contracts documented in `docs/training_pipeline.md` and `docs/execution_script.md`.

6. **Shared logic belongs in `src/python/utils/`**
   - Reused feature/ONNX helpers are centralized in `utils/`; avoid duplicating that logic in script files.

7. **Extraction script naming is historical**
   - `extract_rates_to_csv.py` currently pulls from **yfinance**.
   - `extract_yfinance_rates_to_csv.py` currently pulls from **MetaTrader5**.
   - Keep this in mind before renaming/changing provider logic.

8. **Docs are part of the change contract**
   - When behavior changes in training or execution scripts, update `docs/training_pipeline.md` and/or `docs/execution_script.md` in the same change.
