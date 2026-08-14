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

### Mirror failover behavior (`BuyOnly_ONNX_EA.mq5` and `SellOnly_ONNX_EA.mq5`)
- `InpMirrorEntryOperation` is used as the startup mirror mode.
- Each EA keeps a runtime mirror state and uses it to decide BUY/SELL execution side.
- On every failed entry order (`m_trade.Buy/Sell` returns `false`), the runtime mirror state is flipped and remains active for subsequent entries until another failed entry flips it again.
- The chart comment displays both configured input mirror mode and current runtime mirror mode.

For backtesting, embed the ONNX model as a resource and rename the `#resource` directive at the top of the file.
