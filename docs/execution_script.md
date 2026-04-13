# Execution Script Spec

## Goal
Poll MT5 for live candles, run feature engineering, run ONNX inference,
and place/close market orders based on model output probabilities.

## Implemented Scripts
| Script | Model |
|---|---|
| `execute_onnx_adx_stoch_vol_on_mt5.py` | `train_adx_stoch_vol.py` output |

## Inference Loop
```
every --interval seconds:
  1. fetch last (window + 100) candles from MT5
  2. compute features (same as training)
  3. dropna, take last `window` rows, flatten → float32[1, N]
  4. run ONNX session → [P(sell), P(buy)]
  5. if P(buy)  >= confidence → close any open SELL, open BUY
     if P(sell) >= confidence → close any open BUY,  open SELL
     else → hold
```

## Order Logic
- One position at a time per symbol
- Reversal: close opposite position before opening new one
- Uses `mt5.TRADE_ACTION_DEAL` (market order), `ORDER_FILLING_IOC`

## CLI Contract
```
--model       path to ONNX file
--symbol      MT5 symbol (e.g. NAS100)
--timeframe   M1 M5 M15 M30 H1 H4 D1
--window      window size — must match training (default: 20)
--confidence  minimum probability to place an order (default: 0.60)
--lot         order lot size (default: 1.0)
--interval    seconds between inference cycles (default: 60)
```
Indicator period args (`--adx_period`, `--stoch_k`, `--stoch_d`, `--vol_window`) must match training values.

## Critical Rules
- Feature columns and indicator periods **must** match the training script exactly
- `--window` must match the value used at training time
- MT5 must be running and logged in before starting the script
