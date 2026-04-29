# Correlated Executor Implementation Summary

## File Created
`src/python/execute_correlated_onnx_on_mt5.py`

## Features Implemented

### 1. Multi-Symbol Processing ✓
- Processes 3 symbols simultaneously: NAS100, SP500, DOW30
- Per-symbol ONNX inference execution
- Per-symbol state tracking (consecutive signals, last signal)

### 2. CLI Arguments ✓
- `--model_nas`, `--model_sp`, `--model_dow`: Individual ONNX model paths
- `--symbols`: Comma-separated symbol list (default: "NAS100,SP500,DOW30")
- `--dry_run`: Prevents trades, logs "WOULD EXECUTE" instead
- All standard args: `--timeframe`, `--window`, `--confidence`, `--lot`, `--interval`, `--atr_period`, `--adx_period`, `--adx_min`, `--stoch_k`, `--stoch_d`, `--vol_window`, `--sl_mult`, `--tp_mult`, `--magic`, `--deviation`

### 3. 7-Factor Scoring System ✓
```
score = (signal_strength * 0.3) + (probability * 0.3) + 
        (filter_passed_bonus * 0.2) + (volume_norm * 0.1) + 
        (consec_signals_norm * 0.1)
```
- `signal_strength`: Difference between max and min probabilities
- `probability`: Max of buy/sell probabilities
- `filter_passed_bonus`: 1.0 if signal exists, 0.0 otherwise
- `volume_norm`: Normalized volume (capped at 1M)
- `consec_signals_norm`: Consecutive signal count normalized to 5

### 4. Correlation Alignment Detection ✓
- `check_alignment()`: Detects if all symbols have same signal (BUY/SELL) or mixed
- Returns alignment status: "ALL_BUY", "ALL_SELL", "MIXED(nB_nS)", "NO_SIGNAL"
- Only executes when signals are aligned

### 5. Symbol Selection ✓
- `select_best_symbol()`: Chooses highest-scoring symbol among aligned signals
- Tracks consecutive signal count per symbol
- Returns (symbol, score, alignment_status)

### 6. Dry-Run Mode ✓
- `--dry_run` flag prevents actual trades
- Logs "WOULD_EXECUTE" event with full details
- Sends Telegram notification with [DRY] prefix
- Useful for testing without risking capital

### 7. Enhanced Logging ✓
- `_log()`: Trading events with correlation metrics
- `_log_inference()`: Per-symbol inference results with scores
- Log files: `log/correlated_trading_log.csv`, `log/correlated_inference_log.csv`
- Includes: timestamp, symbol, probabilities, scores, alignment, consecutive counts

### 8. Telegram Notifications ✓
- Notifies on execution with symbol, direction, price, SL
- Notifies on failures with error reason
- Includes [DRY] prefix in dry-run mode

## Execution Flow
1. Parse CLI arguments and load 3 ONNX models
2. Initialize per-symbol state tracking
3. Main loop (every `--interval` seconds):
   - Run inference on all 3 symbols
   - Calculate 7-factor score for each
   - Check signal alignment
   - If aligned: select highest-scoring symbol
   - If not dry_run: open position; else: log "WOULD_EXECUTE"
   - Log inference results and scores

## Backward Compatibility
- Single symbol fallback: Can be used with `--symbols "NAS100"` and single model
- All standard execution script args supported
- Same feature engineering pipeline as base template

## Testing Recommendations
1. Start with `--dry_run` to verify alignment detection and scoring
2. Check `log/correlated_inference_log.csv` for per-symbol scores
3. Verify Telegram notifications include correlation info
4. Test with different symbol combinations
5. Monitor alignment patterns over time

## Known Limitations
- Requires all 3 models to be provided (no optional symbols)
- Executes only when signals are perfectly aligned (all BUY or all SELL)
- No position management (SL/TP) implemented yet (can be added from base template)
