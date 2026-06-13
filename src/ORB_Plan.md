# MetaTrader 5 Expert Advisor Plan
## New York Opening Range Strategy for NASDAQ, S&P 500, or Dow Jones

## 1) Goal

Build a **single-file** MetaTrader 5 Expert Advisor in one `.mq5` source file that trades US indices during the New York open using the first 15-minute candle as the initial range, then watches the next 2 or 3 M15 candles for border tests, pullbacks, continuations, and volume confirmation. MT5 supports algorithmic testing and optimization in the Strategy Tester, and MQL5 provides candle tick volume through `CopyTicks()` for bar-by-bar analysis.

The EA should be designed for NASDAQ, S&P 500, or Dow Jones symbols, using the exact broker symbol through _Symbol global. The strategy should evaluate whether price bounces from the range border and continues, or reverses back toward the 50% midpoint region of the initial range. Session-based ORB systems are a standard MQL5 pattern, and the plan below adapts that idea of exact price-action and volume logic.

---

## 2) Strategy Concept

### 2.1 Trading window
Trade only during the New York opening session. Use a narrow opening focus around the market open, then stop after the observation window ends. A practical configuration is to begin at the New York open and allow trading only for the first few M15 candles after the opening range candle completes.

### 2.2 Opening range
The first completed M15 candle after the session begins is the opening range candle. Its high is the upper border, its low is the lower border, and its midpoint is the 50% internal level. This range becomes the reference for all later decisions in the session.

### 2.3 Observation phase
After the opening range candle, observe the next 2 or 3 M15 candles. For each observed candle, determine:
- whether price is testing the upper or lower range border,
- whether the candle is a pullback candle or a continuation candle,
- whether tick volume supports the move,
- whether the move is likely to bounce and continue or reverse toward the midpoint.

### 2.4 Trade direction logic
If price tests the upper border and confirms continuation or bounce with supportive volume, consider a long trade. If price tests the lower border and confirms continuation or bounce with supportive volume, consider a short trade. If the border test is weak and volume indicates failure, the EA should optionally trade toward the 50% midpoint instead of treating it as a continuation.

---

## 3) Required Market Logic

### 3.1 Scenario set
The EA must handle all of these cases:

- Upper border test, then pullback candle, then continuation upward.
- Upper border test, then weak rejection and reverse back to the midpoint.
- Lower border test, then pullback candle, then continuation downward.
- Lower border test, then weak rejection and reverse back to the midpoint.
- Border touch with strong rejection and volume support.
- Border touch with low conviction and fading tick volume.

### 3.2 Candle interpretation
For each M15 candle, evaluate:
- open, high, low, close,
- candle body size,
- wick size,
- direction,
- tick volume via `CopyTicks()`.
- hammer formation
- engulfing body

### 3.3 Border test definition
A candle is testing a border if:
- upper test: candle high reaches or nearly reaches the initial range high,
- lower test: candle low reaches or nearly reaches the initial range low.

### 3.4 Pullback definition
A pullback candle is a candle that pauses or retraces after the initial border test but does not invalidate the border test. It often has a smaller body, a wick rejecting the border, or a close that moves away from the border before a continuation candle appears.

### 3.5 Continuation definition
A continuation candle is a candle that confirms the move away from the border in the same direction as the border reaction. It should show:
- a close beyond the local reaction point,
- body strength in the intended direction,
- supportive tick volume.

### 3.6 Midpoint reversal definition
If the border test is weak and the reaction loses momentum, the EA may treat the move as a reversal candidate toward the midpoint of the opening range. Midpoint logic should be explicit and not assumed implicitly.

---

## 4) Single File Design

### 4.1 File requirement
Use one file only:
- `OpeningRangeNYEA.mq5`

Do not split into multiple source files, classes, or project modules. Everything should be in one MQ5 source file for portability and easy testing.

### 4.2 Suggested structure
The file should contain these sections in order:

1. Input parameters.
2. Global variables and state.
3. Utility functions.
4. Session and time functions.
5. Candle reading functions.
6. Range detection functions.
7. Pattern recognition functions.
8. Volume confirmation functions.
9. Entry decision function.
10. Trade execution function.
11. Position management function.
12. `OnInit()`.
13. `OnTick()`.
14. Optional `OnDeinit()` cleanup.

---

## 5) Input Parameters

Expose user-editable inputs for all important settings.

### 5.1 Market and session inputs
- `int InpNewYorkOpenHour`
- `int InpNewYorkOpenMinute`
- `int InpObservationCandles = 3`
- `int InpTradeEndHour`
- `int InpTradeEndMinute`

### 5.2 Range logic inputs
- `int InpRangeCandleShift = 1`
- `double InpBorderTolerancePoints`
- `double InpMidpointBufferPoints`
- `bool InpAllowMidpointReversal`

### 5.3 Volume inputs
- `int InpVolumeLookback`
- `double InpVolumeRatioConfirm`
- `double InpMinVolumeThreshold`

### 5.4 Risk inputs
- `double InpLots`
- `int InpStopLossPoints`
- `int InpTakeProfitPoints`
- `bool InpUseTrailingStop`
- `int InpTrailingStartPoints`
- `int InpTrailingStepPoints`

### 5.5 Trade filters
- `int InpMaxTradesPerSession = 1`
- `bool InpOneTradeOnly = true`
- `bool InpUseTimeFilter = true`

---

## 6) State Variables

The EA needs persistent state for the session.

### 6.1 Session state
- `datetime g_sessionStart`
- `datetime g_sessionEnd`
- `bool g_sessionActive`
- `bool g_rangeCaptured`
- `bool g_tradePlaced`

### 6.2 Range state
- `double g_rangeHigh`
- `double g_rangeLow`
- `double g_rangeMid`

### 6.3 Candle state
- `datetime g_lastBarTime`
- `int g_observationCount`
- `MqlRates g_candles[]`

### 6.4 Decision state
- `int g_signalDirection`
- `string g_signalReason`
- `double g_signalConfidence`

---

## 7) Core Functions

### 7.1 Session detection
Create a function that checks whether the current time is within the New York opening window. The function should:
- convert broker time logic consistently,
- verify the current time is inside the allowed trading interval,
- reset the session state when a new day begins,
- prevent trading outside the opening window.

### 7.2 Opening range capture
Create a function that captures the first completed M15 candle after the session begins. Store:
- high,
- low,
- midpoint,
- candle timestamp.

The midpoint is:
- `(rangeHigh + rangeLow) / 2`

### 7.3 Candle reader
Create a reusable function that loads the latest M15 candles into an array. Each candle should include:
- open,
- high,
- low,
- close,
- tick volume,
- time.

Use `CopyTicks()` for volume confirmation. The `CopyTicks` function fills a `MqlTick` array passed by reference.

### 7.4 Border test detector
Create a function that checks whether a candle touches or nearly touches the range border. Use point-based tolerance so broker digits do not break the logic.

### 7.5 Pullback detector
Create a function that detects a pullback candle after a border test. For example:
- after upper test, a bearish or indecisive candle that pulls back from the high,
- after lower test, a bullish or indecisive candle that pulls back from the low.

### 7.6 Continuation detector
Create a function that detects whether the next candle confirms direction after the pullback. Example:
- after upper test, candle closes stronger and moves higher,
- after lower test, candle closes weaker and moves lower.

### 7.7 Midpoint reversal detector
Create a function that evaluates weak border reactions. If rejection strength is low and tick volume fades, mark the setup as a midpoint reversal candidate instead of a continuation.

### 7.8 Volume confirmation
Create a function that compares current and prior tick volumes. Use tick volume as a confirmation filter, not as the sole trigger. Tick volume is available from the MQL5 series function `CopyTicks()`.

Possible logic:
- continuation needs rising or strong relative volume,
- weak rejection needs decreasing or failed volume expansion,
- pullback should not be accepted if volume is inconsistent.

---

## 8) Decision Engine

### 8.1 Step-by-step logic
The main decision engine should work in this order:

1. Verify the session is active.
2. Verify the opening range candle is closed.
3. Capture opening range high, low, and midpoint.
4. Wait for the next 2 or 3 M15 candles.
5. For each candle, detect upper or lower border test.
6. Classify the candle as pullback, continuation, or weak rejection.
7. Check tick volume confirmation.
8. Decide whether the move is a continuation or a reversal to midpoint.
9. If valid and no trade has been placed, execute one trade only.
10. After entry, manage position and stop further entries for that session.

### 8.2 Signal hierarchy
Use this priority:
1. Strong continuation with volume confirmation.
2. Valid pullback followed by continuation.
3. Weak border rejection with midpoint reversal.
4. Ignore ambiguous candles.

### 8.3 Trade direction mapping
- Upper border continuation: buy.
- Lower border continuation: sell.
- Upper border failure: sell toward midpoint.
- Lower border failure: buy toward midpoint.

### 8.4 One-trade rule
Once a trade is placed, disable further entries for the rest of the session. That keeps the EA focused on the first valid opening-range opportunity.

---

## 9) Trade Execution

### 9.1 Market order or pending order
Use market orders for simplicity and direct confirmation after candle close. The logic is based on completed candles, so market orders after confirmation are cleaner than pre-placed pending orders.

### 9.2 Stop loss placement
Set stop loss using one of these methods:
- fixed point distance,
- opposite side of the opening range,
- midpoint-based protective stop,
- volatility-adjusted stop.

### 9.3 Take profit placement
Set take profit using:
- fixed reward multiple,
- full range projection,
- midpoint target for reversal trades,
- multiple of range size for continuation trades.

### 9.4 Spread filter
Before entry, reject the trade if spread is above the configured threshold.

### 9.5 One-position enforcement
Only one position may exist per session. If a position is already open, skip all new entry logic.

---

## 10) Position Management

### 10.1 Trailing stop
If enabled, move stop loss only after price has moved enough in favor of the trade. A trailing stop helps lock in gains on strong opening moves.

### 10.2 Breakeven option
Optionally move stop loss to breakeven after a predefined number of points or after reaching a partial target.

### 10.3 Exit logic
Exit positions when:
- take profit is hit,
- stop loss is hit,
- session ends,
- an opposite invalidation condition appears if you choose to support manual-style closure.

---

## 11) OnTick Flow

### 11.1 Main loop order
The `OnTick()` function should follow this sequence:

1. Return immediately if session is inactive.
2. Return if current bar has not changed.
3. Update candle buffer.
4. Capture opening range if needed.
5. Evaluate only after the opening range candle is complete.
6. Analyze the next 2 or 3 candles.
7. Confirm with tick volume.
8. Generate a signal.
9. Open a trade if valid.
10. Manage any open position.

### 11.2 Bar-change control
Use a last-bar-time check to ensure the EA only processes once per new M15 bar instead of on every tick.

---

## 12) Pseudocode Outline

```mq5
int OnInit()
{
   // Validate inputs
   // Set session state
   // Prepare arrays and timers
   return(INIT_SUCCEEDED);
}

void OnTick()
{
   if(!IsWithinTradingWindow()) return;
   if(!IsNewBar()) return;

   UpdateCandles();

   if(!g_rangeCaptured)
   {
      if(CanCaptureOpeningRange())
      {
         CaptureOpeningRange();
         g_rangeCaptured = true;
      }
      return;
   }

   if(g_tradePlaced) return;

   int signal = EvaluateSetup();

   if(signal != 0)
   {
      if(SpreadIsAcceptable() && NoOpenPosition())
      {
         PlaceTrade(signal);
         g_tradePlaced = true;
      }
   }

   ManagePosition();
}
```

---

## 13) Pattern Rules to Implement

### 13.1 Upper border sequence
A valid long or reversal setup from the upper border may be:
- candle 1 touches the upper border,
- candle 2 pulls back or pauses,
- candle 3 confirms continuation or rejects back to midpoint.

### 13.2 Lower border sequence
A valid short or reversal setup from the lower border may be:
- candle 1 touches the lower border,
- candle 2 pulls back or pauses,
- candle 3 confirms continuation or rejects back to midpoint.

### 13.3 Volume interpretation
Use volume like this:
- rising volume on continuation = stronger confirmation,
- fading volume on rejection = weaker continuation,
- abnormal spike with no follow-through = possible exhaustion,
- flat or weak volume = avoid aggressive entry.

### 13.4 Midpoint behavior
If price fails to hold the border, target the midpoint rather than assuming a full reversal. That keeps the EA from overcommitting when the market is undecided.

---

## 14) Testing Plan

### 14.1 MT5 Strategy Tester
Test the EA in the MT5 Strategy Tester, which is the correct environment for validating Expert Advisors and optimizing settings.

### 14.2 Symbols to test
- NASDAQ index symbol from your broker.
- S&P 500 index symbol from your broker.
- Dow Jones index symbol from your broker.

### 14.3 Timeframe
Use M15 only, since the strategy is defined around 15-minute opening range structure.

### 14.4 Test categories
- strong trend opening,
- choppy opening,
- false breakout opening,
- reversal to midpoint,
- low-volume day,
- high-volume day.

### 14.5 Validation metrics
Track:
- win rate,
- profit factor,
- average R multiple,
- maximum drawdown,
- average trade duration,
- average slippage,
- session-specific hit rate.

---

## 15) Coding Notes

### 15.1 Keep it single-file
Do not create include files, libraries, or multiple source modules. Everything should be inside one `.mq5` file.

### 15.2 Make logic modular
Even in one file, structure the code with small functions so it stays readable.

### 15.3 Avoid hardcoded symbol assumptions
Use inputs for the symbol name because broker naming differs across NASDAQ, S&P 500, and Dow Jones products.

### 15.4 Use completed candles only
All decisions should be based on closed candles to avoid repaint-like behavior and unstable signals.

### 15.5 Keep the first version simple
The first version should prioritize correctness and clean session logic over over-optimization.

---

## 16) Suggested First Version Scope

The first working version should include:
- NY session time filter,
- opening range capture from the first M15 candle,
- 2 or 3 candle observation window,
- border touch detection,
- pullback and continuation detection,
- tick volume confirmation,
- midpoint reversal logic,
- one trade per session,
- market order entry,
- SL and TP,
- session end lockout.

Later versions can add:
- trailing stop,
- breakeven,
- better candle scoring,
- spread logic improvements,
- volatility filters,
- news filter,
- multi-symbol adaptation.

---

## 17) Final Build Target

The final deliverable should be:
- one `.mq5` file,
- fully compilable in MetaEditor,
- designed for MT5 Strategy Tester,
- ready for NASDAQ, S&P 500, or Dow Jones,
- focused on New York opening hours,
- based on the first 15-minute range,
- able to distinguish continuation, pullback, bounce, and midpoint reversal,
- using tick volume for confirmation.

---

## 18) Recommended Next Step

Implement the EA in one file using this structure, then backtest each index separately in the Strategy Tester with the same logic and different symbol inputs. Start with conservative settings and only expand the logic after the first version behaves consistently.