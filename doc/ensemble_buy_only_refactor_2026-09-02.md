# Ensemble Buy-Only Refactor Changelog

**Date:** 2026-09-02  
**Plan:** `.kilo/plans/1788354928147-ensemble-buy-only-refactor.md`

## Summary

Refactored `train_ensemble_buy_only.py` to fix train/inference mismatches, improve target expressiveness, expand hyperparameters, normalize metadata, and add auto-scoring for imbalanced classes.

---

## 1. Fixed Train/Inference Feature Mismatches

### `src/indicators.py`
- Added `calculate_atr_wilder(highs, lows, closes, period)` implementing Wilder's recursive smoothing: `ATR[t] = (ATR[t-1] * (period-1) + TR[t]) / period`.
- This exactly matches MT5 `iATR()` smoothing, fixing the previous SMA-based approximation used for model D.

### `src/train_ensemble_buy_only.py`
- **Model D (structure):** Replaced SMA-based `range_norm` with Wilder-smoothed ATR via `_calculate_atr_wilder_pd()`. The feature is now `feat_range_norm = range / Wilder_ATR(20)`, matching the EA's `range / iATR(20)`.
- **Model E (volatility):** Verified `feat_range_expansion = range / range.shift(1)` matches EA logic (`range / prevRange`). The `shift(1)` gives the previous bar, consistent with `barIdx+1` in the EA's time-series arrays.

---

## 2. Made Target Definition More Expressive

### Per-model `inc_percent`
Added `inc_percent` overrides to `ENSEMBLE_CONFIG`:
- A (impulse, window=5): `0.3` — fast moves need tighter threshold
- B (swing, window=15): `0.5`
- C (trend, window=30): `1.0` — longer horizon allows larger move
- D (structure, window=20): `0.5`
- E (volatility, window=20): `0.5`

### New CLI flag
- `--inc-percent` (default `0.5`): Default threshold overridden by per-model config when training full ensemble.

### Target formula
Now stored exactly as used in metadata:
```
max(high[t+1..t+{forward}]) > close[t] * (1 + {inc_percent}/100)
```

---

## 3. Improved Hyperparameters and Validation

### Expanded `param_dist`
- `n_estimators`: `[200, 300, 400, 500]`
- `max_depth`: `[8, 12, 16, 20, None]`
- `min_samples_leaf`: `[1, 2, 5, 10]`
- `min_samples_split`: `[2, 5, 10]`
- `max_features`: `['sqrt', 'log2', None]`
- `class_weight`: `[None, 'balanced', 'balanced_subsample']`

### Default `n_splits`
Increased from `3` to `5` for more robust OOS estimates.

### Train/test split validation
Added explicit checks aborting training if either split has fewer than 2 classes, preventing cryptic sklearn crashes on extreme imbalance.

---

## 4. Normalized Metadata to Match AGENTS.md Contract

Added flat keys to every exported ONNX:
- `training.feature_names` (flat list)
- `training.window_size` (int)
- `training.n_features` (int)
- `training.records_after_dropna` (int, after `dropna`)
- `training.samples_used` (int, `len(X)`)
- `training.input_size` (int)
- `training.train_prediction_distribution` (dict)
- `training.cli_args` (JSON string of parsed args)
- `training.param_dist` (JSON string)

Existing `ensemble.*` keys are preserved as additive metadata.

---

## 5. Improved Class Imbalance Handling and Auto-Scoring

### Auto-scoring
- If minority class < 5% of samples, scoring automatically switches to `average_precision` (PR-AUC).
- Otherwise, uses `balanced_accuracy`.
- Printed to console: `Scoring metric: {scoring} (minority class: {minority_pct:.2f}%)`

### New CLI flag
- `--scoring` with choices `['balanced_accuracy', 'average_precision', 'roc_auc']` (default: auto-select).

---

## Files Changed

| File | Changes |
|------|---------|
| `src/indicators.py` | Added `calculate_atr_wilder()` |
| `src/train_ensemble_buy_only.py` | Fixed model D ATR, per-model inc_percent, expanded param_dist, metadata normalization, auto-scoring, train/test validation |
| `graft/src/indicators.md` | Updated line numbers, added `calculate_atr_wilder` |
| `graft/src/train_ensemble_buy_only.md` | New graft card for ensemble script |
| `README.md` | Added ensemble script and EA documentation |
| `AGENTS.md` | Added ensemble contract, updated feature sets, CLI contract, layout |

---

## Verification

- Syntax check passed for both modified Python files.
- Smoke tests ran successfully on balanced and imbalanced synthetic data.
- Metadata verified queryable via `query_onnx_metadata.py`.
- ONNX `probabilities` output shape `[*, 2]` verified after export.

---

## Risks & Mitigations

| Risk | Mitigation |
|------|------------|
| ATR computation change for model D | Existing ONNX models trained with old SMA-based `range_norm` become stale. Retrain all models after fix. |
| Metadata key rename | `query_onnx_metadata.py` consumers may need updates. Old `ensemble.*` keys kept as aliases. |
| Target threshold change | Per-model `inc_percent` changes label distribution. OOS metrics will shift; this is expected and desired. |
