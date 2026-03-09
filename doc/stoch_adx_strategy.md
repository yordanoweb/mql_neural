# SGRADT 5.0 - Minor Timeframe Strategy Analysis

This document details the trading strategy for the Stochastic and ADX indicators as implemented in the minor timeframe of the SGRADT 5.0 trading robot.

## Stochastic Oscillator Strategy (Minor Timeframe)

The strategy uses the Stochastic Oscillator to identify potential overbought and oversold conditions, generating buy and sell signals based on crossovers and momentum.

### Indicator Configuration

The following input parameters from `Markets.mqh` are used for the minor timeframe Stochastic:

*   `StochasticKPeriodInput_M1`: 7
*   `StochasticDPeriodInput_M1`: 3
*   `StochasticSlowingInput_M1`: 3
*   `StochasticMaxBuyLevelInput_M1`: 80.0 (Overbought level)
*   `StochasticMinSellLevelInput_M1`: 20.0 (Oversold level)
*   `StochBypass_M1_Input`: `true` (if true, signals are ignored)

### Buy Signal (`StochGoLong_M1`)

A buy signal is generated under one of the following conditions:

1.  **Oversold Crossover:** The main Stochastic line crosses above the signal line while in the oversold area. This is checked for the last two candles.
    *   Condition: `(Stochastic main line (2 candles ago) < Stochastic signal line (2 candles ago) && Stochastic main line (1 candle ago) > Stochastic signal line (1 candle ago) && Stochastic main line (1 candle ago) <= Minor Timeframe's Minimum Sell Level)`
    *   *Alternative lookback*: `(Stochastic main line (3 candles ago) < Stochastic signal line (3 candles ago) && Stochastic main line (2 candles ago) > Stochastic signal line (2 candles ago) && Stochastic main line (2 candles ago) <= Minor Timeframe's Minimum Sell Level)`

2.  **Strong Upward Momentum:** The main line shows a rapid increase in value, indicating strong buying pressure.
    *   Condition: `Stochastic main line (current candle) > Stochastic main line (1 candle ago) + 7 && Stochastic main line (1 candle ago) > Stochastic main line (2 candles ago) + 7` (The value `7` is a hardcoded momentum factor).

### Sell Signal (`StochGoShort_M1`)

A sell signal is generated under one of the following conditions:

1.  **Overbought Crossover:** The main Stochastic line crosses below the signal line while in the overbought area.
    *   Condition: `(Stochastic main line (2 candles ago) > Stochastic signal line (2 candles ago) && Stochastic main line (1 candle ago) < Stochastic signal line (1 candle ago) && Stochastic main line (1 candle ago) >= Minor Timeframe's Maximum Buy Level)`
    *   *Alternative lookback*: `(Stochastic main line (3 candles ago) > Stochastic signal line (3 candles ago) && Stochastic main line (2 candles ago) < Stochastic signal line (2 candles ago) && Stochastic main line (2 candles ago) >= Minor Timeframe's Maximum Buy Level)`

2.  **Strong Downward Momentum:** The main line shows a rapid decrease in value, indicating strong selling pressure.
    *   Condition: `Stochastic main line (current candle) < Stochastic main line (1 candle ago) - 7 && Stochastic main line (1 candle ago) < Stochastic main line (2 candles ago) - 7` (The value `7` is a hardcoded momentum factor).

---

## Average Directional Index (ADX) Strategy (Minor Timeframe)

The ADX strategy is used to confirm the strength of a trend before entering a trade.

### Indicator Configuration

The following input parameters from `Markets.mqh` are used for the minor timeframe ADX:

*   `ADXPeriodInput_M1`: 8
*   `ADXLimitInput_M1`: 32
*   `ADXBypass_M1_Input`: `false` (if true, signals are ignored)
*   `ADX_DI_OverInput_M1`: `false` (If true, requires DI+ > DI- for buy, and vice-versa for sell)

### Pre-condition for Signals

All buy or sell signals are only valid if the market is trending. The pre-condition is met if:
*   The ADX value for the current or previous candle is above `ADXLimitInput_M1`.
*   OR the ADX shows strong upward movement (`ADX line (1 candle ago) - ADX line (2 candles ago) > 5` or `ADX line (current candle) - ADX line (1 candle ago) > 5`).

### Buy Signal (`ADXGoLong_M1`)

If the pre-condition is met, a buy signal is generated if one of the following occurs:

1.  **DI+ Trending Up, DI- Trending Down:** The +DI line is in a clear uptrend while the -DI line is in a downtrend.
    *   Condition: `(+DI line (current candle) > +DI line (2 candles ago) && +DI line (1 candle ago) > +DI line (2 candles ago) && +DI line (current candle) > +DI line (1 candle ago)) && (-DI line (current candle) < -DI line (1 candle ago) && -DI line (1 candle ago) < -DI line (2 candles ago))`

2.  **-DI Reversal:** The -DI line shows a sustained downtrend, and the +DI line begins to rise.
    *   Condition: `(-DI line (2 candles ago) < -DI line (3 candles ago) && -DI line (1 candle ago) < -DI line (2 candles ago) && -DI line (current candle) < -DI line (1 candle ago) && +DI line (current candle) > +DI line (2 candles ago))`

### Sell Signal (`ADXGoShort_M1`)

If the pre-condition is met, a sell signal is generated if one of the following occurs:

1.  **DI- Trending Up, DI+ Trending Down:** The -DI line is in a clear uptrend while the +DI line is in a downtrend.
    *   Condition: `(-DI line (current candle) > -DI line (2 candles ago) && -DI line (1 candle ago) > -DI line (2 candles ago) && -DI line (current candle) > -DI line (1 candle ago)) && (+DI line (current candle) < +DI line (1 candle ago) && +DI line (1 candle ago) < +DI line (2 candles ago))`

2.  **+DI Reversal:** The +DI line shows a sustained downtrend, and the -DI line begins to rise.
    *   Condition: `(+DI line (2 candles ago) < +DI line (3 candles ago) && +DI line (1 candle ago) < +DI line (2 candles ago) && +DI line (current candle) < +DI line (1 candle ago) && -DI line (current candle) > -DI line (2 candles ago))
