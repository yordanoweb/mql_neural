# ***MT5 Trading Script Documentation**

## ***Overview**

***This Python script implements an automated trading system for MetaTrader 5 that combines EMA-based entry protection, Stochastic oscillator confirmation, ADX trend strength filtering, and an intelligent forecast-based exit mechanism. The script exits positions based on ATR-derived Stop Loss/Take Profit levels, with an optional forecast fulfillment check that closes profitable positions after a specified number of candles.**


## ***Architecture**

***plain**

***Copy**

```
`┌─────────────────────────────────────────────────────────────┐`
`│                     ***MAIN LOOP                               │`**
`│  ***Runs every N seconds (configurable via --interval)         │`**
`└─────────────────────────────────────────────────────────────┘`
`                            │`
`                            ▼`
`┌─────────────────────────────────────────────────────────────┐`
`│  ***1. SESSION CHECK                                           │`**
`│     • ***Verify current hour within trading window             │`**
`│     • ***Skip processing if outside hours                      │`**
`└─────────────────────────────────────────────────────────────┘`
`                            │`
`                            ▼`
`┌─────────────────────────────────────────────────────────────┐`
`│  ***2. DATA ACQUISITION                                        │`**
`│     • ***Fetch current bid/ask prices                          │`**
`│     • ***Retrieve historical candles (min 50 bars)             │`**
`│     • ***Compute EMA, ATR, Stochastic, ADX indicators          │`**
`└─────────────────────────────────────────────────────────────┘`
`                            │`
`                            ▼`
`┌─────────────────────────────────────────────────────────────┐`
`│  ***3. POSITION MANAGEMENT                                     │`**
`│     ***IF position exists:                                       │`**
`│       → ***Check forecast exit (bars ≥ horizon + profit \> 0)   │`**
`│     ***ELSE:                                                   │`**
`│       → ***Evaluate entry conditions                             │`**
`└─────────────────────────────────────────────────────────────┘`
`                            │`
`                            ▼`
`┌─────────────────────────────────────────────────────────────┐`
`│  ***4. ENTRY EVALUATION (if no position)                       │`**
`│     • ***EMA crossover condition (price vs EMA)                  │`**
`│     • ***Stochastic signal (oversold/overbought + momentum)      │`**
`│     • ***ADX trend confirmation (strength + DI alignment)        │`**
`│     • ***All three must align → execute market order             │`**
`└─────────────────────────────────────────────────────────────┘`
`                            │`
`                            ▼`
`┌─────────────────────────────────────────────────────────────┐`
`│  ***5. LOGGING & MONITORING                                    │`**
`│     • ***Display position status, P&L, bars held               │`**
`│     • ***Show indicator values (Stoch, ADX, DI+, DI-)          │`**
`│     • ***Color-coded console output                            │`**
`└─────────────────────────────────────────────────────────────┘`
```

## ***Entry Logic**

### ***EMA Protection (Primary Filter)**

***The EMA serves as the directional gate for all entries, ensuring trades align with the dominant trend:**

***Table**

| **Condition** | **Requirement** | **Protection Purpose** |
| - | - | - |
| **BUY** | Price must be **above** EMA + entry threshold | Prevents buying in downtrends |
| **SELL** | Price must be **below** EMA - entry threshold | Prevents selling in uptrends |

***Technical Implementation:**

***Python**

***Copy**

```
***`\# BUY: Previous candle crossed up through EMA, current price above EMA + buffer`***
***`buy\_ema\_cond = (`**
`    ***prev\_candle\['open'\] \< prev\_ema and           *\# Was below EMA`***
`    ***prev\_candle\['close'\] \> current\_ema and       *\# Crossed above`***
`    ***current\_price \> current\_ema + entry\_threshold *\# Confirmed above with buffer`***
***`)`**

***`\# SELL: Previous candle crossed down through EMA, current price below EMA - buffer  `***
***`sell\_ema\_cond = (`**
`    ***prev\_candle\['open'\] \> prev\_ema and           *\# Was above EMA`***
`    ***prev\_candle\['close'\] \< current\_ema and       *\# Crossed below`***
`    ***current\_price \< current\_ema - entry\_threshold *\# Confirmed below with buffer`***
***`)`**
```

***Entry Threshold: Configurable distance from EMA (default 10 points) to avoid entries too close to the moving average.**

### ***Stochastic Confirmation**

***Provides momentum timing for entries, identifying oversold bounces (for BUY) and overbought rejections (for SELL).**

***BUY Signal Conditions (any one satisfied):**

1. ***Oversold Crossover: %K crosses above %D while %K ≤ 20 (oversold)**

2. ***Alternative Crossover: Lookback to 3 candles ago for same pattern**

3. ***Strong Momentum: %K rises \>7 points for 2 consecutive candles**

***SELL Signal Conditions (any one satisfied):**

1. ***Overbought Crossover: %K crosses below %D while %K ≥ 80 (overbought)**

2. ***Alternative Crossover: Lookback to 3 candles ago for same pattern**

3. ***Strong Momentum: %K falls \>7 points for 2 consecutive candles**

***Bypass Mode: `--stoch\_bypass` flag disables Stochastic requirement (always returns True).**

### ***ADX Trend Strength Filter**

***Ensures entries occur during sufficient trend strength, avoiding choppy/ranging markets.**

***Trending Condition (any one satisfied):**

- ***ADX \> limit (default 32) on current or previous candle**

- ***ADX rising \>5 points between candles (acceleration)**

***BUY-Specific Conditions:**

- ***DI+ trending upward while DI- trending downward (bullish alignment)**

- ***OR sustained DI- downtrend with DI+ beginning to rise**

***SELL-Specific Conditions:**

- ***DI- trending upward while DI+ trending downward (bearish alignment)**

- ***OR sustained DI+ downtrend with DI- beginning to rise**

***DI Overlap Mode: `--adx\_di\_over` flag requires DI+ \> DI- for BUY and DI- \> DI+ for SELL (stronger confirmation).**

***Bypass Mode: `--adx\_bypass` flag disables ADX requirement (always returns True).**

### ***Entry Execution Flow**

***plain**

***Copy**

```
`┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐`
`│   ***EMA CROSS?    │ ──► │  STOCH SIGNAL?  │ ──► │   ADX TREND?    │`**
`│  ***(Trend Filter) │     │ (Timing Filter) │     │(Strength Filter)│`**
`└─────────────────┘     └─────────────────┘     └─────────────────┘`
`       │                         │                         │`
`       ***NO                        NO                        NO`**
`       │                         │                         │`
`       ▼                         ▼                         ▼`
`   ***\[SKIP ENTRY\]              \[SKIP ENTRY\]              \[SKIP ENTRY\]`**
`       `
`       ***YES ─────────────────────► YES ───────────────────► YES`**
`                                                        │`
`                                                        ▼`
`                                              ┌─────────────────┐`
`                                              │  ***EXECUTE ORDER  │`**
`                                              │  • ***ATR-based SL │`**
`                                              │  • ***ATR-based TP │`**
`                                              │  • ***Market price │`**
`                                              └─────────────────┘`
```

## ***Exit Logic**

### ***Primary Exit: Forecast-Based Intelligent Exit**

***The script implements a time-aware profit capture system that respects the model's prediction horizon.**

***Parameters:**

- ***`--forecast\_horizon N` (default 4): Number of candles to wait before evaluating exit**

***Behavior:**

***Table**

| **Phase** | **Candles Held** | **Condition** | **Action** |
| - | - | - | - |
| **Accumulation** | 0 to N-1 | — | Hold position, ignore profit/loss |
| **Evaluation** | ≥ N | Profit \> 0 | **Close immediately** — forecast fulfilled |
| **Evaluation** | ≥ N | Profit ≤ 0 | **Hold and monitor** — wait for profit |
| **Extended** | \> N | Profit turns \> 0 | Close on first profitable bar |
| **Safety** | Any | SL/TP hit | Close at SL/TP level |

***Logic Implementation:**

***Python**

***Copy**

```
***`def check\_forecast\_exit(position, df, current\_candle\_time):`**
`    ***bars\_held = get\_bars\_since\_entry(position, df)`**
`    `
`    ***if bars\_held \< FORECAST\_HORIZON:`**
`        ***return False  *\# Too early, keep holding`***
`    `
`    ***if position.profit \> 0:`**
`        ***\# Forecast fulfilled — price moved as predicted`***
`        ***close\_position(position)`**
`        ***return True`**
`    ***else:`**
`        ***\# Not yet profitable — log status but keep open`***
`        ***\# SL/TP will protect if trend reverses`***
`        ***return False`**
```

***Key Insight: The script assumes the AI model was trained with a specific forecast window (e.g., 4 candles). After this window, if the prediction was correct (profit \> 0), it captures gains. If incorrect (loss), it waits rather than crystallizing losses, allowing either:**

- ***Price eventually moves favorably (close when profitable)**

- ***Stop Loss triggers (controlled risk)**

- ***Take Profit triggers (if price overshoots horizon)**

### ***Secondary Exit: ATR-Based Stop Loss / Take Profit**

***Set at entry time based on current market volatility.**

***Calculation:**

***Python**

***Copy**

```
***`atr\_value = compute\_atr(df, ATR\_PERIOD)  *\# Default 14 periods`***

***`\# BUY order`***

***`sl\_price = ask - (atr\_value \* SL\_MULT)   *\# Default 2.0x ATR below entry`***
***`tp\_price = ask + (atr\_value \* TP\_MULT)   *\# Default 3.0x ATR above entry`***

***`\# SELL order  `***

***`sl\_price = bid + (atr\_value \* SL\_MULT)   *\# 2.0x ATR above entry`***
***`tp\_price = bid - (atr\_value \* TP\_MULT)   *\# 3.0x ATR below entry`***
```

***Rationale: ATR-based stops adapt to market volatility — wider stops in volatile conditions, tighter in calm markets. The 1:1.5 risk/reward ratio (SL 2x vs TP 3x) provides positive expectancy.**

### ***Exit Priority Hierarchy**

***plain**

***Copy**

```
`┌────────────────────────────────────────┐`
`│  ***1. FORECAST EXIT (if enabled)         │`**
`│     • ***Trigger: Bars ≥ horizon AND      │`**
`│              ***profit \> 0                │`**
`│     • ***Priority: Highest (captures      │`**
`│       ***model-predicted moves)           │`**
`├────────────────────────────────────────┤`
`│  ***2. TAKE PROFIT (hard stop)            │`**
`│     • ***Trigger: Price hits TP level      │`**
`│     • ***Set at: Entry ± (ATR × TP\_MULT)   │`**
`├────────────────────────────────────────┤`
`│  ***3. STOP LOSS (hard stop)              │`**
`│     • ***Trigger: Price hits SL level      │`**
`│     • ***Set at: Entry ± (ATR × SL\_MULT)   │`**
`│     • ***Priority: Safety floor            │`**
`└────────────────────────────────────────┘`
```

## ***Command-Line Parameters**

### ***Core Trading Parameters**

***Table**

| **Parameter** | **Default** | **Description** |
| - | - | - |
| `--symbol` | EURUSD | Trading instrument (e.g., EURUSD, GBPUSD, XAUUSD) |
| `--timeframe` | M1 | Candle timeframe (M1, M5, H1, D1, etc.) |
| `--volume` | 0.1 | Order size in lots |
| `--magic` | 91234569 | Magic number for position identification |
| `--interval` | 5.0 | Seconds between processing loops |

### ***EMA & Entry Parameters**

***Table**

| **Parameter** | **Default** | **Description** |
| - | - | - |
| `--ema\_period` | 9 | EMA calculation period (entry trend filter) |
| `--entry\_points` | 10.0 | Minimum points from EMA to trigger entry |

### ***ATR & Risk Parameters**

***Table**

| **Parameter** | **Default** | **Description** |
| - | - | - |
| `--atr\_period` | 14 | ATR lookback period for volatility calculation |
| `--sl\_multiplier` | 2.0 | Stop Loss = ATR × multiplier |
| `--tp\_multiplier` | 3.0 | Take Profit = ATR × multiplier |

### ***Forecast Exit Parameters**

***Table**

| **Parameter** | **Default** | **Description** |
| - | - | - |
| `--forecast\_horizon` | 4 | Candles to wait before profit evaluation (0=disabled) |

### ***Stochastic Parameters**

***Table**

| **Parameter** | **Default** | **Description** |
| - | - | - |
| `--stoch\_k` | 7 | %K period (momentum line) |
| `--stoch\_d` | 3 | %D period (signal line) |
| `--stoch\_slowing` | 3 | Smoothing factor |
| `--stoch\_overbought` | 80.0 | Overbought threshold |
| `--stoch\_oversold` | 20.0 | Oversold threshold |
| `--stoch\_bypass` | False | Disable Stochastic requirement |

### ***ADX Parameters**

***Table**

| **Parameter** | **Default** | **Description** |
| - | - | - |
| `--adx\_period` | 8 | ADX calculation period |
| `--adx\_limit` | 32.0 | Minimum ADX for trend consideration |
| `--adx\_bypass` | False | Disable ADX requirement |
| `--adx\_di\_over` | False | Require DI alignment (DI+ \> DI- for BUY) |

### ***Session Parameters**

***Table**

| **Parameter** | **Default** | **Description** |
| - | - | - |
| `--start\_hour` | 0 | Trading window start (24h format) |
| `--end\_hour` | 23 | Trading window end (24h format) |

## ***Usage Examples**

### ***Basic Usage (Default Settings)**

***bash**

***Copy**

```
***`python script.py --symbol EURUSD --timeframe M5`**
```

### ***Aggressive Scalping (1-minute, tight stops, no forecast exit)**

***bash**

***Copy**

```
***`python script.py --symbol EURUSD --timeframe M1 \\`**
`    ***--ema\_period 5 --atr\_period 10 \\`**
`    ***--sl\_multiplier 1.5 --tp\_multiplier 2.0 \\`**
`    ***--forecast\_horizon 0 --stoch\_bypass`**
```

### ***Swing Trading (4-hour, conservative, forecast-based exit)**

***bash**

***Copy**

```
***`python script.py --symbol GBPUSD --timeframe H4 \\`**
`    ***--ema\_period 21 --atr\_period 14 \\`**
`    ***--sl\_multiplier 3.0 --tp\_multiplier 5.0 \\`**
`    ***--forecast\_horizon 6 --adx\_di\_over`**
```

### ***Gold Trading (XAUUSD, high volatility adjustments)**

***bash**

***Copy**

```
***`python script.py --symbol XAUUSD --timeframe M15 \\`**
`    ***--entry\_points 50 --atr\_period 14 \\`**
`    ***--sl\_multiplier 2.5 --tp\_multiplier 4.0 \\`**
`    ***--forecast\_horizon 3`**
```

### ***London Session Only**

***bash**

***Copy**

```
***`python script.py --symbol EURUSD --timeframe M5 \\`**
`    ***--start\_hour 8 --end\_hour 17`**
```


## ***Console Output Format**

***plain**

***Copy**

```
***`\[2024-01-15 14:32:08\] EURUSD price=1.08542 EMA9=1.08510 | No position`**
`  ***Stoch %K=25.40 %D=22.15 | ADX=35.20 +DI=28.50 -DI=18.30`**

***`--- New candle: 2024-01-15 14:32:00 ---`**
***`\[2024-01-15 14:32:13\] EURUSD price=1.08548 EMA9=1.08512 | Position: BUY 0.1 lots, profit=12.50 \[Bars: 3/4\]`**
`  ***Price is 0.00036 above EMA`**
`  ***Stoch %K=45.20 %D=38.90 | ADX=36.10 +DI=30.20 -DI=16.80`**

***`\[FORECAST EXIT\] Position held for 4 bars (horizon: 4), profit: 15.20. Closing...`**
***`Position closed: 123456789`**
```

***Color Coding:**

- ***Green: Profitable positions, executed orders, fulfilled forecasts**

- ***Red: Losses, failed orders, price below EMA for BUY / above for SELL**

- ***Yellow: Warnings, waiting states, missing conditions**

- ***Cyan: New candles, indicator values**

- ***Blue: Timestamps**

- ***Magenta: EMA values**

## ***Risk Management Features**

1. ***Single Position Rule: Only one position per symbol/magic number at a time**

2. ***Session Boundaries: No entries outside specified hours (existing positions remain open)**

3. ***ATR-Based Sizing: Stop distances adapt to volatility (wider in volatile markets)**

4. ***Positive Risk/Reward: Default 2:3 ratio (risk 2x ATR to gain 3x ATR)**

5. ***Forecast Safety Valve: Prevents holding unprofitable positions indefinitely after prediction window expires (SL/TP still active)**

## ***Dependencies**

***bash**

***Copy**

```
***`pip install MetaTrader5 pandas ta colorama`**
```

***Table**

| **Package** | **Purpose** |
| - | - |
| `MetaTrader5` | MT5 terminal integration |
| `pandas` | Data manipulation, time series handling |
| `ta` | Technical indicator calculations (EMA, ATR, Stochastic, ADX) |
| `colorama` | Cross-platform colored console output (optional) |


## ***Limitations & Considerations**

***Table**

| **Aspect** | **Behavior** |
| - | - |
| **Tick Frequency** | Processes every N seconds (not every tick), may miss rapid moves |
| **Fill Assumption** | Assumes IOC (Immediate-or-Cancel) filling — slippage possible |
| **Spread Ignored** | Entry/exit logic uses mid-price; wide spreads affect actual fills |
| **No Partial Close** | All-or-nothing position closure |
| **No Trailing Stop** | Fixed SL/TP only, no dynamic adjustment |
| **Single Symbol** | One symbol per script instance |
| **No Persistence** | Entry tracking lost if script restarts (relies on MT5 position history) |

## ***Flowchart Summary**

***plain**

***Copy**

```
***`START`**
`  │`
`  ▼`
***`\[Initialize MT5\]`**
`  │`
`  ▼`
***`\[Fetch Market Data\] ──► \[Compute Indicators\]`**
`  │`
`  ▼`
***`\[Check Session Hours\] ──NO──► \[Sleep\]`**
`  │ ***YES`**
`  ▼`
***`\[Get Open Position?\]`**
`  │`
`  ├─***YES──► \[Check Forecast Exit\] ──► \[Close if fulfilled\]`**
`  │          │`
`  │          └─***NO─► \[Continue Holding\]`**
`  │`
`  └─***NO──► \[Check Entry Conditions\]`**
`            │`
`            ├─***\[EMA Cross?\] ──NO──► \[Wait\]`**
`            │ ***YES`**
`            ├─***\[Stoch Signal?\] ──NO──► \[Log Missing\]`**
`            │ ***YES`**
`            ├─***\[ADX Confirm?\] ──NO──► \[Log Missing\]`**
`            │ ***YES`**
`            ▼`
`         ***\[Execute Order with SL/TP\]`**
`            │`
`            ▼`
`         ***\[Log & Sleep\]`**
```

***This documentation covers the complete behavior of the trading script. Adjust parameters based on your specific strategy requirements and backtesting results.**

