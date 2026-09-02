# Refactor Plan: `train_ensemble_buy_only.py` — Better Future Price Increase Prediction

**Constraint**: Preserve the 3-feature ONNX contract and keep the existing `EnsembleBuyEA.mq5` unchanged. Do not touch other files. Only modify `train_ensemble_buy_only.py` and its helper modules if required.
**Goal**: Improve predictive accuracy for future price increases by fixing train/inference mismatches, making the target more expressive, extracting shared code, and tightening metadata compliance.

---

## 1. Fix Train/Inference Feature Mismatches

**Problem**: Python training computes features differently than the MQL5 EA computes them at inference time. This is the highest-impact bug to fix.

| Model | Python (current) | EA (inference) | Mismatch? | Fix |
|---|---|---|---|---|
| A, B, C | `feat_body = close - open`, `feat_range = high - low`, `feat_rsi = RSI/100` | `body = c-o`, `range = h-l`, `rsi = RSI/100` | None | No change |
| D (structure) | `feat_body_ratio = body/range`, `feat_range_norm = range / SMA(range,20)`, `feat_rsi` | `bodyRatio = body/range`, `rangeNorm = range / iATR(20)`, `rsi` | **Yes** | Replace Python SMA with Wilder-smoothed ATR matching MT5 `iATR` |
| E (volatility) | `feat_range = h-l`, `feat_range_expansion = range / range.shift(1)`, `feat_rsi` | `range = h-l`, `rangeExpansion = range / prevRange`, `rsi` | Off-by-one risk | Verify and align shift direction with EA's `barIdx+1` logic |

**Action**: Implement a `calculate_atr_wilder(highs, lows, closes, period)` helper in `indicators.py` (or a new utils module) that exactly matches MT5 `iATR` smoothing. Use it in the `structure` feature set.

---

## 2. Make Target Definition More Expressive

**Problem**: `inc_percent = 0.5` is hardcoded. Different models operate on different time horizons; a single threshold is suboptimal.

**Action**:
- Add `--inc-percent` CLI flag (default `0.5`) to `train_ensemble_buy_only.py`.
- In `ENSEMBLE_CONFIG`, add per-model `inc_percent` overrides:
  - A (impulse, window=5): `0.3` — fast moves need tighter threshold
  - B (swing, window=15): `0.5`
  - C (trend, window=30): `1.0` — longer horizon allows larger move
  - D (structure, window=20): `0.5`
  - E (volatility, window=20): `0.5`
- Keep the target logic in `create_windows` parameterized by `inc_percent`.
- Store `training.target` in metadata as the exact formula string used.

---

## 3. Improve Hyperparameters and Validation

**Problem**: Current search space and CV settings are modest.

**Action**:
- Expand `param_dist`:
  - `n_estimators`: `[200, 300, 400, 500]`
  - `max_depth`: `[8, 12, 16, 20, None]`
  - `min_samples_leaf`: `[1, 2, 5, 10]`
  - `max_features`: `['sqrt', 'log2', None]`
  - `class_weight`: `[None, 'balanced', 'balanced_subsample']`
- Increase default `n_splits` from `3` to `5` for more robust OOS estimates.
- Add `min_samples_split`: `[2, 5, 10]`.
- Keep `refit=True` on `RandomizedSearchCV`.

---

## 4. Normalize Metadata to Match AGENTS.md Contract

**Problem**: Ensemble script uses `ensemble.*` keys but misses required flat keys.

**Action**: In `set_onnx_metadata`, always write:
- `training.feature_names` (flat list)
- `training.window_size` (int)
- `training.n_features` (int)
- `training.records_after_dropna` (int, after `dropna`)
- `training.samples_used` (int, `len(X)`)
- `training.cli_args` (JSON string of parsed args)
- `training.param_dist` (JSON string)

Keep existing `ensemble.*` keys as additive metadata. Do not remove them.

---

## 5. Improve Class Imbalance and Early Stopping

**Problem**: The `max(high) > close * 1.005` target can produce highly imbalanced classes on ranging markets.

**Action**:
- Print class distribution before training (already done).
- If minority class < 5% of samples, automatically switch scoring to `'average_precision'` (PR-AUC) instead of `balanced_accuracy`, because PR-AUC is more informative under imbalance.
- Add `--scoring` CLI flag with choices `['balanced_accuracy', 'average_precision', 'roc_auc']` (default: auto-select based on class balance).

---

## Execution Order

1. Fix model D ATR computation and add Wilder-smoothed ATR helper.
2. Add per-model `inc_percent` and `--inc-percent` CLI flag.
3. Add metadata normalization (flat keys).
4. Expand hyperparameter search space and improve default `n_splits`.
5. Update documentation.

---

## Risks

- **ATR computation change for model D**: Existing ONNX models trained with the old (SMA-based) `range_norm` will become stale. Retrain all models after the fix.
- **Metadata key rename**: `query_onnx_metadata.py` consumers may need updates if they depend on old `ensemble.*` keys. Mitigation: keep old keys as aliases.
- **Target threshold change**: Per-model `inc_percent` changes the label distribution. OOS metrics will shift; this is expected and desired.

