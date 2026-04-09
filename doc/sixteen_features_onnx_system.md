# Sixteen Features ONNX Trading System

This document describes the training and execution scripts for the 16-feature ONNX trading model with enhanced ADX capabilities.

---

## 1. Training Script: `train_onnx_sixteen_features.py`

### Overview
Trains a RandomForest classifier with 16 technical features and exports to ONNX format for deployment.

### Feature Architecture (16 Features)

| # | Feature | Category | Description |
|---|---------|----------|-------------|
| 1 | feat_body | Price | Normalized candle body (close - open) / ATR |
| 2 | feat_range | Price | Normalized candle range (high - low) / ATR |
| 3 | feat_stoch_momentum | Stochastic | Stochastic momentum (K - D) / 100 |
| 4 | feat_stoch_position | Stochastic | Stochastic position (K - 50) / 50 |
| 5 | feat_stoch_velocity | Stochastic | Rate of change of K |
| 6 | feat_stoch_divergence | Stochastic | Overbought/oversold pressure zones |
| 7 | feat_vol_ratio | Volume | Current volume vs MA ratio |
| 8 | feat_vol_momentum | Volume | Volume trend strength (EMA5 - EMA20) |
| 9 | feat_vol_price_div | Volume | Volume-price divergence |
| 10 | feat_vol_percentile | Volume | Volume rank in recent window |
| 11 | feat_vol_zscore | Volume | Statistical anomaly detection |
| 12 | feat_adx_strength | ADX | Normalized ADX centered around adx_min |
| 13 | feat_di_signal | ADX | Directional signal: 1 (DI+ > DI-), -1 (DI- > DI+), 0 |
| 14 | feat_di_separation | ADX | Directional conviction (DI+ - DI-) / (DI+ + DI-) |
| 15 | feat_adx_momentum | ADX | Rate of change of trend strength |
| 16 | feat_adx_regime | ADX | Categorical: 0 (no trend), 0.5 (developing), 1 (strong) |

### Command Line Arguments

```bash
python train_onnx_sixteen_features.py \
  --input_csv data.csv \
  --output_dir ./models \
  --atr_period 14 \
  --window 20 \
  --future 10 \
  --n_iter 10 \
  --min_profit_atr 1.5 \
  --stoch_window 14 \
  --vol_window 20 \
  --adx_period 14 \
  --adx_min 20.0 \
  --jobs 3
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--input_csv` | required | Path to OHLCV CSV data |
| `--output_dir` | `.` | Directory for ONNX output |
| `--atr_period` | 14 | ATR calculation period |
| `--window` | 20 | Feature calculation window |
| `--future` | 10 | Lookahead bars for target |
| `--n_iter` | 10 | RandomizedSearchCV iterations |
| `--min_profit_atr` | 1.5 | Minimum profit in ATR multiples |
| `--stoch_window` | 14 | Stochastic oscillator period |
| `--vol_window` | 20 | Volume analysis window |
| `--adx_period` | 14 | ADX calculation period |
| `--adx_min` | 20.0 | ADX threshold for trend detection |
| `--jobs` | 3 | Parallel jobs for training |

### Output
ONNX model file with naming pattern:
```
{csv_stem}_adx_w{window}_f{future}_atr{atr}_minp{minp}_adx{adx}_adxm{adx_min}.onnx
```

---

## 2. Execution Script: `execute_onnx_sixteen_features_on_mt5.py`

### Overview
Live trading executor for MetaTrader 5 using 16-feature ONNX models.

### Command Line Arguments

```bash
python execute_onnx_sixteen_features_on_mt5.py \
  --model model.onnx \
  --symbol EURUSD \
  --timeframe M1 \
  --confidence 0.55 \
  --window 20 \
  --start_hour 9 \
  --end_hour 23 \
  --interval 60 \
  --consistency_bars 3 \
  --lot 1.0 \
  --atr_period 8 \
  --sl_mult 1.0 \
  --tp_mult 2.0 \
  --stoch_period 5 \
  --vol_window 10 \
  --adx_period 14 \
  --adx_min 20.0 \
  --cooldown 60
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--model` | required | Path to ONNX model file |
| `--symbol` | EURUSD | Trading symbol |
| `--timeframe` | M1 | Chart timeframe |
| `--confidence` | 0.55 | Probability threshold for signals |
| `--window` | 20 | Feature window size |
| `--start_hour` | 9 | Trading start hour (24h) |
| `--end_hour` | 23 | Trading end hour (24h) |
| `--interval` | 60 | Seconds between checks |
| `--force_consistency` | True | Require consecutive signals |
| `--consistency_bars` | 3 | Consecutive bars for confirmation |
| `--lot` | 1.0 | Position volume |
| `--magic` | auto | Order identifier |
| `--atr_period` | 8 | ATR lookback period |
| `--sl_mult` | 1.0 | Stop-loss ATR multiplier |
| `--tp_mult` | 2.0 | Take-profit ATR multiplier |
| `--stoch_period` | 5 | Stochastic period |
| `--vol_window` | 10 | Volume window |
| `--adx_period` | 14 | ADX period |
| `--adx_min` | 20.0 | ADX minimum threshold |
| `--h1_trend` | False | Filter by H1 trend |
| `--log_file` | trading_log.csv | Trade log path |
| `--cooldown` | 60 | Seconds between trades |

### Signal Logic
- **BUY**: prob ≥ confidence
- **SELL**: 1 - prob ≥ confidence
- **Consistency**: Requires N consecutive signals before entry
- **H1 Filter**: Optional trend alignment with hourly candle

### Output Format
```
Hour: 07:54:15 | Prob: +0.519/+0.55 | Expected: 0.55
Buffer: ['HOLD', 'HOLD', 'HOLD', 'HOLD'] | Signal: HOLD | Positions: 0
```

### Logging

CSV structure for `execute_onnx_sixteen_features_on_mt5_balanced.py`:

```
| Column            | Description                                         |
| ----------------- | --------------------------------------------------- |
| `timestamp`       | Event time (YYYY-MM-DD HH:MM:SS)                    |
| `symbol`          | Trading symbol                                      |
| `timeframe`       | Chart timeframe                                     |
| `candle_time`     | Candle timestamp                                    |
| `hold_prob`       | Probability for HOLD class                          |
| `buy_prob`        | Probability for BUY class                           |
| `sell_prob`       | Probability for SELL class                          |
| `predicted_class` | Model prediction (0=HOLD, 1=BUY, 2=SELL)            |
| `raw_signal`      | Raw signal label (HOLD/BUY/SELL)                    |
| `buffer`          | Signal history array                                |
| `signal`          | Final signal after consistency (HOLD/BUY/SELL/NONE) |
| `action`          | Action taken (HOLD/BUY/SELL/CLOSE)                  |
| `price`           | Entry/exit price                                    |
| `sl`              | Stop loss                                           |
| `tp`              | Take profit                                         |
| `atr`             | ATR value                                           |
| `balance`         | Account balance                                     |
| `equity`          | Account equity                                      |
```

