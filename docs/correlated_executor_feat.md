# Comprehensive Plan for Correlated Executor:

## Core Features
1. **File Name**: execute_correlated_onnx_on_mt5.py
2. **Base Template**: Modified from execute_onnx_adx_stoch_vol_on_mt5.py
3. **Multi-Symbol**: Process 3 symbols simultaneously (NAS100,SP500,DOW30)
4. **CLI Args**: --symbols "NAS100,SP500,DOW30" + --dry_run flag

## Processing Pipeline
5. **Inference Collection**: Run ONNX inference on all 3 symbols each cycle
6. **State Management**: Per-symbol state tracking (position, consecutive signals)
7. **Correlation Analysis**: Detect signal alignment (all BUY/SELL vs mixed)

## Decision Engine
8. **7-Factor Scoring**:
   ```
   score = (signal_strength * 0.3) + (probability * 0.3) + 
           (filter_passed_bonus * 0.2) + (volume_norm * 0.1) + 
           (consec_signals_norm * 0.1)
   ```
9. **Symbol Selection**: Execute highest-scoring symbol with aligned signals

## Safety & Testing
10. **Dry Run Mode**: --dry_run prevents trades, logs "WOULD EXECUTE"
11. **Backward Compatible**: Single symbol fallback if --symbols not specified

## Output & Monitoring
12. **Enhanced Logging**: Correlation metrics, scores per symbol, alignment status
13. **Telegram Notifications**: Include correlation info, scores, selected symbol

## Execution Flow
```
1. Parse symbols from CLI
2. For each cycle:
   ├─ Run inference on all symbols
   ├─ Calculate per-symbol scores
   ├─ Check correlation alignment
   ├─ Select highest score symbol
   └─ If not dry_run → Execute trade
       Else → Log "WOULD EXECUTE"
```

**NEXT STEP**: Ready to implement complete executor code.
