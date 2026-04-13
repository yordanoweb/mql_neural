# Execution Script Spec

## Goal
Poll MT5 for live candles, run feature engineering, run ONNX inference,
and place market orders. Exit is managed by SL and a trailing logic — not by signal reversal.

## Implemented Scripts
| Script | Model |
|---|---|
| `execute_onnx_adx_stoch_vol_on_mt5.py` | `train_adx_stoch_vol.py` output |

## Inference Loop
```
every --interval seconds:
  if position open:
    → run manage_open_trade()
  else:
    → run inference → [P(sell), P(buy)]
    → if P(buy)  >= confidence: open BUY
    → if P(sell) >= confidence: open SELL
    → else: hold
```
Only one position at a time. No new trade is opened while one is active.

## Entry
- Market order with hard SL: `entry_price ± ATR(atr_period) × sl_mult`
- No TP sent to broker

## Exit Logic
1. **Hard SL** — set on MT5 at open, broker handles it
2. **Imaginary TP** — tracked internally in Python: `entry_price ± ATR × tp_mult`
3. Once imaginary TP is reached, **trailing mode** activates:
   - BUY trade → close on first M1 candle where `close < open` (bearish)
   - SELL trade → close on first M1 candle where `close > open` (bullish)
4. M1 candle check always uses the last *closed* M1 candle (index -2)

## CLI Contract
```
--model       path to ONNX file
--symbol      MT5 symbol (e.g. NAS100)
--timeframe   M1 M5 M15 M30 H1 H4 D1
--window      window size — must match training (default: 20)
--confidence  minimum probability to open a trade (default: 0.60)
--lot         order lot size (default: 1.0)
--interval    seconds between cycles (default: 60)
--atr_period  ATR period for SL/TP calculation (default: 14)
--sl_mult     SL distance = ATR × sl_mult (default: 1.5)
--tp_mult     imaginary TP distance = ATR × tp_mult (default: 2.0)
```
Indicator period args (`--adx_period`, `--stoch_k`, `--stoch_d`, `--vol_window`) must match training values.

## Output (colorized)
Every cycle prints one of:

- FLAT (cyan): `[HH:MM:SS] FLAT — running inference...` + probabilities
- Signal (green/red): `P(buy)=0.72  P(sell)=0.28  → BUY signal`
- No signal (yellow): `→ no signal`
- Open trade (green=BUY/red=SELL): `[HH:MM:SS] BUY | HOLDING | entry=... price=... PnL=+... | SL=... iTP=...`
- Trailing active (magenta): same line with `TRAILING` instead of `HOLDING`
- Close (magenta): `→ CLOSED (trailing_exit): retcode=10009`

## Critical Rules
- Feature columns and indicator periods must match the training script exactly
- `--window` must match the value used at training time
- MT5 must be running and logged in before starting the script
- Script state (`TradeState`) is in-memory only — restarting resets it
