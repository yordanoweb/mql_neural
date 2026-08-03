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
