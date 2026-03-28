# PATCHED Documentation (RSI -> Volume Effort/Result)

## Feature Engineering Update

RSI has been replaced with a professional volume-based feature.

### New Feature: Effort vs Result

Effort/Result = Relative Volume × (Body / Range)

Where:
- Relative Volume = tick_volume / rolling_mean(volume)
- Body = abs(close - open)
- Range = high - low

### Interpretation
- High effort + low result → absorption
- High effort + high result → breakout strength
- Low effort + high result → weak move

This provides a more orthogonal and institutional-grade signal than RSI.
