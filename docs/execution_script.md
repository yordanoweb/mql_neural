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
    → run inference → [P(hold), P(buy), P(sell)]
    → if P(buy)  >= confidence: open BUY
    → if P(sell) >= confidence: open SELL
    → else: hold
```
Only one position at a time. No new trade is opened while one is active.

## Entry
- Market order with hard SL: `entry_price ± ATR(atr_period, trading_timeframe) × sl_mult`
- ATR is always computed on the **trading timeframe** (not hardcoded to M5)
- No TP sent to broker
- MT5 order comment: `F16_B@<p_buy>` or `F16_S@<p_sell>` — the inference probability that triggered the entry (3 decimal places)

## Exit Logic
1. **Hard SL** — set on MT5 at open, broker handles it
2. **Imaginary TP** — tracked internally in Python: `entry_price ± ATR × tp_mult`
3. **Breakeven** — when directional profit (not absolute distance) reaches 0.5× ATR, SL is moved to entry price (zero-risk position)
4. **Profit lock** — on every new candle at the **profit lock timeframe** (one step below trading timeframe: M5→M1, M15→M5, M30→M15, H1→M30, H4→H1, D1→H4), SL is ratcheted to the previous candle's low (BUY) or high (SELL), subject to three conditions all being true: (a) current price is in profit (above entry for BUY, below entry for SELL), (b) the new SL level itself is past entry (locks at least breakeven), and (c) the new level is better than the current SL (never widens it)
5. Once imaginary TP is reached, **trailing mode** activates:
   - The trailing candle timeframe is one step below the trading timeframe (M5→M1, M15→M5, M30→M15, H1→M30, H4→H1, D1→H4). Falls back to M1 if no mapping exists.
   - BUY trade → close on first opposite candle where `close < open` (bearish) on the trailing timeframe
   - SELL trade → close on first opposite candle where `close > open` (bullish) on the trailing timeframe
6. Trailing candle check always uses the last *closed* candle (index -2) on the trailing timeframe

## CLI Contract
```
--model           path to ONNX file
--symbol          MT5 symbol (e.g. NAS100)
--timeframe       M1 M5 M15 M30 H1 H4 D1
--window          window size — must match training (default: 20)
--confidence      minimum probability to open a trade (default: 0.60)
--lot             order lot size — used when --max_risk is not set (default: 1.0)
--interval        seconds between cycles (default: 60)
--atr_period      ATR period for SL/TP calculation (default: 14)
--sl_mult         SL distance = ATR × sl_mult (default: 1.5)
--tp_mult         imaginary TP distance = ATR × tp_mult (default: 2.0)
--deviation       max slippage in points for order requests (default: 20)
--magic           magic number for MT5 orders (default: 0)
--ema_period      EMA period for trend filter (default: 18)
--max_risk        fraction of free margin to risk per trade (e.g. 0.01 = 1%). Overrides --lot when > 0 (default: 0.0)
--decrease_factor consecutive-loss lot reduction factor — 0 = disabled (default: 0.0)
```
Indicator period args (`--adx_period`, `--stoch_k`, `--stoch_d`, `--vol_window`) must match training values.

## Inference Stats
Every FLAT cycle updates in-memory running min/max of raw model probabilities since script start.
After each inference, a stats line is printed:

```
  stats(42): buy=[0.481–0.731]  sell=[0.269–0.519]
```

- `count` — number of FLAT inference cycles so far
- `buy=[min–max]` — observed range of `P(buy)` across all cycles
- `sell=[min–max]` — observed range of `P(sell)` across all cycles

Use this to calibrate `--confidence`: if `max_buy` never exceeds 0.63 after many candles,
a confidence of 0.70 will never fire. Stats reset on script restart (in-memory only).

## Output (colorized)
Every cycle prints one of:

- FLAT (cyan): `[HH:MM:SS] SYMBOL FLAT — running inference...` + probabilities + stats line
- Signal (green/red): `P(buy)=0.72  P(sell)=0.28  → BUY signal`
- No signal (yellow): `→ no signal`
- Open trade (green=BUY/red=SELL): `[HH:MM:SS] SYMBOL BUY | HOLDING | entry=... price=... PnL=$+1.23 | SL=... iTP=...`
- Trailing active (magenta): same line with `TRAILING` instead of `HOLDING`
- Close (magenta): `→ CLOSED (trailing_exit): retcode=10009`

## Trade Logging
Every entry and exit is appended to `trades.csv` in the working directory.

| Column | Description |
|---|---|
| `timestamp` | ISO datetime of the event |
| `event` | `OPEN` or `CLOSE` |
| `symbol` | MT5 symbol |
| `direction` | `BUY` or `SELL` |
| `price` | Execution price |
| `sl` | SL price at time of event |
| `tp_target` | Imaginary TP level |
| `atr` | ATR value at entry (OPEN only) |
| `confidence` | Model probability that triggered entry (OPEN only) |
| `pnl_pts` | Price points PnL at close (CLOSE only) |
| `reason` | Exit reason: `trailing_exit` or `sl_hit` |

Exit reason `sl_hit` is detected when the broker closes the position (position disappears from MT5 while `_state.ticket != 0`).

## EMA Trend Filter
Applied only at execution time — training is unchanged.

- EMA is computed on the trading timeframe candles already fetched for inference
- Filter uses the **forming candle** (`iloc[-1]`) for both `close` and EMA — reflects the live price at decision time
- Filter only activates when a signal fires (`p >= confidence`)
- BUY signal: allowed only if `close > EMA(ema_period)` (strictly above the line)
- SELL signal: allowed only if `close < EMA(ema_period)` (strictly below the line)
- When blocked: prints `→ EMA filter: close=X <= EMA=Y — BUY blocked` (yellow)
- Inference always runs and stats always update regardless of filter outcome

## Telegram Notifications
Implemented in `src/python/utils/telegram.py`. Reads `BOT_TOKEN` and `CHAT_ID` from `.env` at the project root (quotes stripped automatically). Errors are silently ignored (fire-and-forget).

Three events trigger a message:

| Event | Trigger | Message content |
|---|---|---|
| Script start | `run()` entry | Symbol, timeframe, confidence, model filename |
| Trade open | `open_position()` — retcode 10009 | Direction, symbol, price, SL, iTP, confidence |
| Trade close | `close_position()` (trailing exit) or SL-hit detection | Direction, symbol, price, PnL pts, reason |

Example messages:
```
🚀 Bot started
📊 NAS100  M5
🎯 Confidence: 0.6
📁 ustec_m5_16_feat_adx_stoch_vol.onnx

🟢 BUY OPENED — NAS100
💵 Price:      19500.00000
🛡 SL:         19450.00000
🎯 iTP:        19580.00000
📈 Confidence: 0.724

🔴 SELL CLOSED — NAS100
💵 Price:  19480.00000
🔻 PnL:    -12.50 USD
📉 Balance: 10934.57 USD
🛑 Reason: SL hit
```

PnL is computed as `pnl_pts / tick_size * tick_value * lot` using MT5 symbol info. The PnL emoji is `✅` for profit and `🔻` for loss.

## Dynamic Lot Sizing
When `--max_risk > 0`, lot size is computed dynamically instead of using `--lot`:

```
lot = free_margin × max_risk / margin_per_lot
```

If `--decrease_factor > 0`, consecutive losing trades on the same symbol+magic reduce the lot:
```
losses = count of consecutive losing deals (same symbol+magic, walking back until a profit)
if losses > 1:
    lot = lot - lot × losses / decrease_factor
```
Lot is always clamped to `[volume_min, volume_max]` and snapped to `volume_step`.
Set `--decrease_factor 0` (default) to disable the reduction.


- Feature columns and indicator periods must match the training script exactly
- `--window` must match the value used at training time
- MT5 must be running and logged in before starting the script
- Script state (`TradeState`) is in-memory only — restarting resets it

## Maintenance Rule
After every implementation, feature addition, bug fix, or test: update this doc and `docs/training_pipeline.md` to reflect the current behaviour before committing.

## Regression Contract (never break working features)
Any change to the execution script must preserve the following behaviours. If a change would alter any of these, it must be explicitly requested and this contract must be updated accordingly.

1. **EMA trend filter** — BUY only when `close > EMA`; SELL only when `close < EMA`. No entry at or on the wrong side of the line.
2. **Hard SL** — always set on the broker at order open; never removed or widened.
3. **Breakeven** — SL moves to entry price when directional profit reaches 0.5× ATR; never moves SL backwards; never triggers on a losing position.
4. **Profit lock** — SL ratchets to previous candle low/high on the profit-lock timeframe; only tightens, never widens.
5. **Imaginary TP + trailing exit** — trailing mode activates only after imaginary TP is hit; exit on first opposite candle on the trailing timeframe.
6. **Single position** — never opens a new trade while one is already open.
7. **ONNX metadata contract** — `feature_names`, `window_size`, `n_features` always present in exported models.
8. **Trade logging** — every open and close event is appended to `trades.csv`.
9. **Telegram notifications** — bot start, trade open, and trade close events always fire a notification.
