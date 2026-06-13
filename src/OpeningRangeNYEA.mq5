//+------------------------------------------------------------------+
//|                                           OpeningRangeNYEA.mq5      |
//|   New York Opening Range Strategy for NASDAQ / S&P 500 / Dow      |
//|   Single-file MT5 Expert Advisor                                   |
//+------------------------------------------------------------------+
#property copyright "ORB EA"
#property version   "1.00"
#property strict

#include <Trade\Trade.mqh>

//====================================================================
// 1) INPUT PARAMETERS
//====================================================================

// --- Market and session inputs
input int    InpNewYorkOpenHour    = 9;     // NY open hour (broker server time)
input int    InpNewYorkOpenMinute  = 30;    // NY open minute (broker server time)
input int    InpObservationCandles = 3;     // Number of M15 candles to observe (2 or 3)
input int    InpTradeEndHour       = 12;    // Hour to stop trading (broker server time)
input int    InpTradeEndMinute     = 0;     // Minute to stop trading

// --- Range logic inputs
input int    InpRangeCandleShift     = 1;       // Shift of the opening range candle (1 = most recently closed)
input double InpBorderTolerancePoints = 30;     // Tolerance (points) for "border test"
input double InpMidpointBufferPoints  = 20;     // Buffer (points) around midpoint
input bool   InpAllowMidpointReversal = true;   // Allow midpoint reversal trades

// --- Volume inputs
input int    InpVolumeLookback       = 5;       // Number of candles for average volume baseline
input double InpVolumeRatioConfirm   = 1.10;    // Ratio vs average volume required for confirmation
input double InpMinVolumeThreshold   = 1;       // Minimum absolute tick volume to consider valid

// --- Risk inputs
input double InpLots                 = 0.10;    // Lot size
input int    InpStopLossPoints       = 300;     // Stop loss in points
input int    InpTakeProfitPoints     = 600;     // Take profit in points
input bool   InpUseTrailingStop      = false;   // Enable trailing stop
input int    InpTrailingStartPoints  = 200;     // Points in profit before trailing starts
input int    InpTrailingStepPoints   = 50;      // Trailing step in points

// --- Trade filters
input int    InpMaxTradesPerSession  = 1;       // Max trades per session
input bool   InpOneTradeOnly         = true;    // Only allow one trade total per session
input bool   InpUseTimeFilter        = true;    // Enable session time filter
input double InpMaxSpreadPoints      = 50;      // Maximum allowed spread in points

//====================================================================
// 2) GLOBAL VARIABLES AND STATE
//====================================================================

CTrade trade;

// --- Session state
datetime g_sessionStart   = 0;
datetime g_sessionEnd     = 0;
bool     g_sessionActive  = false;
bool     g_rangeCaptured  = false;
bool     g_tradePlaced    = false;
datetime g_currentDay     = 0;
int      g_tradesThisSession = 0;

// --- Range state
double   g_rangeHigh = 0.0;
double   g_rangeLow  = 0.0;
double   g_rangeMid  = 0.0;
datetime g_rangeCandleTime = 0;

// --- Candle state
datetime g_lastBarTime    = 0;
int      g_observationCount = 0;

// --- Decision state
int      g_signalDirection = 0;    // 1 = buy, -1 = sell, 0 = none
string   g_signalReason    = "";
double   g_signalConfidence = 0.0;

//====================================================================
// 3) UTILITY FUNCTIONS
//====================================================================

//--- Convert points to price distance for current symbol
double PointsToPrice(double points)
  {
   return points * _Point;
  }

//--- Get candle body size
double CandleBody(const MqlRates &c)
  {
   return MathAbs(c.close - c.open);
  }

//--- Get candle full range size
double CandleRange(const MqlRates &c)
  {
   return c.high - c.low;
  }

//--- Get upper wick size
double CandleUpperWick(const MqlRates &c)
  {
   double topOfBody = MathMax(c.open, c.close);
   return c.high - topOfBody;
  }

//--- Get lower wick size
double CandleLowerWick(const MqlRates &c)
  {
   double botOfBody = MathMin(c.open, c.close);
   return botOfBody - c.low;
  }

//--- Candle direction: 1 = bullish, -1 = bearish, 0 = doji/flat
int CandleDirection(const MqlRates &c)
  {
   if(c.close > c.open) return 1;
   if(c.close < c.open) return -1;
   return 0;
  }

//--- Detect a hammer formation (long lower wick, small body, small upper wick)
bool IsHammer(const MqlRates &c)
  {
   double range = CandleRange(c);
   if(range <= 0) return false;

   double body  = CandleBody(c);
   double lower = CandleLowerWick(c);
   double upper = CandleUpperWick(c);

   bool smallBody     = (body <= range * 0.35);
   bool longLowerWick = (lower >= range * 0.5);
   bool smallUpperWick= (upper <= range * 0.2);

   return (smallBody && longLowerWick && smallUpperWick);
  }

//--- Detect an inverted hammer / shooting-star type formation (long upper wick)
bool IsShootingStar(const MqlRates &c)
  {
   double range = CandleRange(c);
   if(range <= 0) return false;

   double body  = CandleBody(c);
   double lower = CandleLowerWick(c);
   double upper = CandleUpperWick(c);

   bool smallBody     = (body <= range * 0.35);
   bool longUpperWick = (upper >= range * 0.5);
   bool smallLowerWick= (lower <= range * 0.2);

   return (smallBody && longUpperWick && smallLowerWick);
  }

//--- Detect a bullish engulfing pattern between two candles (prev, current)
bool IsBullishEngulfing(const MqlRates &prev, const MqlRates &curr)
  {
   bool prevBear = (prev.close < prev.open);
   bool currBull = (curr.close > curr.open);
   if(!prevBear || !currBull) return false;

   return (curr.close >= prev.open && curr.open <= prev.close);
  }

//--- Detect a bearish engulfing pattern between two candles (prev, current)
bool IsBearishEngulfing(const MqlRates &prev, const MqlRates &curr)
  {
   bool prevBull = (prev.close > prev.open);
   bool currBear = (curr.close < curr.open);
   if(!prevBull || !currBear) return false;

   return (curr.close <= prev.open && curr.open >= prev.close);
  }

//====================================================================
// 4) SESSION AND TIME FUNCTIONS
//====================================================================

//--- Returns the start-of-day datetime (midnight) for the given time
datetime StartOfDay(datetime t)
  {
   MqlDateTime dt;
   TimeToStruct(t, dt);
   dt.hour = 0;
   dt.min  = 0;
   dt.sec  = 0;
   return StructToTime(dt);
  }

//--- Build a datetime for "today" at a given hour/minute
datetime TimeAtHourMinute(datetime baseDay, int hour, int minute)
  {
   MqlDateTime dt;
   TimeToStruct(baseDay, dt);
   dt.hour = hour;
   dt.min  = minute;
   dt.sec  = 0;
   return StructToTime(dt);
  }

//--- Reset all per-session state for a new trading day
void ResetSessionState(datetime now)
  {
   datetime today = StartOfDay(now);
   g_currentDay   = today;

   g_sessionStart = TimeAtHourMinute(today, InpNewYorkOpenHour, InpNewYorkOpenMinute);
   g_sessionEnd   = TimeAtHourMinute(today, InpTradeEndHour, InpTradeEndMinute);

   // Handle the (unlikely) case where end time is before start time on same day
   if(g_sessionEnd <= g_sessionStart)
      g_sessionEnd += 24 * 3600;

   g_sessionActive = false;
   g_rangeCaptured = false;
   g_tradePlaced   = false;
   g_tradesThisSession = 0;

   g_rangeHigh = 0.0;
   g_rangeLow  = 0.0;
   g_rangeMid  = 0.0;
   g_rangeCandleTime = 0;

   g_observationCount = 0;

   g_signalDirection  = 0;
   g_signalReason     = "";
   g_signalConfidence = 0.0;

   Print("ORB EA: New session initialized for ", TimeToString(today, TIME_DATE),
         " | Session start=", TimeToString(g_sessionStart, TIME_MINUTES),
         " end=", TimeToString(g_sessionEnd, TIME_MINUTES));
  }

//--- Checks whether current time is within the New York opening trading window.
//    Also resets state when a new day begins.
bool IsWithinTradingWindow(datetime now)
  {
   datetime today = StartOfDay(now);

   // New day -> reset session state
   if(today != g_currentDay)
      ResetSessionState(now);

   if(!InpUseTimeFilter)
     {
      g_sessionActive = true;
      return true;
     }

   if(now >= g_sessionStart && now <= g_sessionEnd)
     {
      g_sessionActive = true;
      return true;
     }

   g_sessionActive = false;
   return false;
  }

//====================================================================
// 5) CANDLE READING FUNCTIONS
//====================================================================

//--- Loads a single completed M15 candle by shift (1 = last closed)
bool GetCandle(int shift, MqlRates &outCandle)
  {
   MqlRates rates[];
   ArraySetAsSeries(rates, true);

   int copied = CopyRates(_Symbol, PERIOD_M15, shift, 1, rates);
   if(copied <= 0)
      return false;

   outCandle = rates[0];
   return true;
  }

//--- Loads N completed M15 candles starting at shift (shift=1 is last closed),
//    returned newest-first (index 0 = most recent of the requested range).
bool GetCandles(int shift, int count, MqlRates &outArr[])
  {
   ArraySetAsSeries(outArr, true);
   int copied = CopyRates(_Symbol, PERIOD_M15, shift, count, outArr);
   return (copied == count);
  }

//--- Gets the tick volume of a completed M15 candle by shift
long GetCandleTickVolume(int shift)
  {
   long volumes[];
   ArraySetAsSeries(volumes, true);

   int copied = CopyTickVolume(_Symbol, PERIOD_M15, shift, 1, volumes);
   if(copied <= 0)
      return 0;

   return volumes[0];
  }

//--- Computes the average tick volume over InpVolumeLookback candles,
//    starting at the given shift (inclusive), looking further back in history.
double GetAverageTickVolume(int startShift, int lookback)
  {
   long volumes[];
   ArraySetAsSeries(volumes, true);

   int copied = CopyTickVolume(_Symbol, PERIOD_M15, startShift, lookback, volumes);
   if(copied <= 0)
      return 0.0;

   double sum = 0.0;
   for(int i = 0; i < copied; i++)
      sum += (double)volumes[i];

   return sum / copied;
  }

//====================================================================
// 6) RANGE DETECTION FUNCTIONS
//====================================================================

//--- Returns true if the opening range candle (at InpRangeCandleShift) is
//    available and corresponds to the candle that closed at/after session start.
bool CanCaptureOpeningRange()
  {
   MqlRates rangeCandle;
   if(!GetCandle(InpRangeCandleShift, rangeCandle))
      return false;

   // The opening range candle must have OPENED at or after the session start time,
   // i.e. it is the first M15 candle of the session.
   if(rangeCandle.time < g_sessionStart)
      return false;

   // Must already be closed: the candle's close time should be <= now,
   // which is guaranteed because CopyRates only returns completed bars for shift>=1.
   return true;
  }

//--- Captures opening range high/low/midpoint from the configured candle shift
void CaptureOpeningRange()
  {
   MqlRates rangeCandle;
   if(!GetCandle(InpRangeCandleShift, rangeCandle))
     {
      Print("ORB EA: Failed to capture opening range candle.");
      return;
     }

   g_rangeHigh = rangeCandle.high;
   g_rangeLow  = rangeCandle.low;
   g_rangeMid  = (g_rangeHigh + g_rangeLow) / 2.0;
   g_rangeCandleTime = rangeCandle.time;

   g_observationCount = 0;

   Print("ORB EA: Opening range captured. Time=", TimeToString(g_rangeCandleTime),
         " High=", DoubleToString(g_rangeHigh, _Digits),
         " Low=", DoubleToString(g_rangeLow, _Digits),
         " Mid=", DoubleToString(g_rangeMid, _Digits));
  }

//====================================================================
// 7) PATTERN RECOGNITION FUNCTIONS
//====================================================================

// Border test result codes
#define BORDER_NONE   0
#define BORDER_UPPER  1
#define BORDER_LOWER  2

//--- 7.4 Border test detector: checks whether a candle touches/nearly touches
//    the opening range borders, using point-based tolerance.
int DetectBorderTest(const MqlRates &c)
  {
   double tol = PointsToPrice(InpBorderTolerancePoints);

   bool upperTest = (c.high >= g_rangeHigh - tol);
   bool lowerTest = (c.low  <= g_rangeLow  + tol);

   if(upperTest && lowerTest)
     {
      // Candle touched both borders (wide candle) - prefer the side closer
      // to where the candle closed.
      double distToUpper = MathAbs(g_rangeHigh - c.close);
      double distToLower = MathAbs(c.close - g_rangeLow);
      return (distToUpper <= distToLower) ? BORDER_UPPER : BORDER_LOWER;
     }

   if(upperTest) return BORDER_UPPER;
   if(lowerTest) return BORDER_LOWER;

   return BORDER_NONE;
  }

//--- 7.5 Pullback detector: detects a pullback candle after a border test.
//    borderSide: BORDER_UPPER or BORDER_LOWER (the side that was tested)
bool IsPullbackCandle(const MqlRates &testCandle, const MqlRates &pullbackCandle, int borderSide)
  {
   int dir = CandleDirection(pullbackCandle);

   if(borderSide == BORDER_UPPER)
     {
      // After an upper test, a pullback candle is bearish or indecisive,
      // and/or shows a wick rejecting the high, and closes away from the border.
      bool bearishOrFlat = (dir <= 0);
      bool closesAwayFromBorder = (pullbackCandle.close < testCandle.high);
      bool hasRejectionWick = (CandleUpperWick(pullbackCandle) >= CandleBody(pullbackCandle) * 0.5);

      return (bearishOrFlat && closesAwayFromBorder) || hasRejectionWick;
     }

   if(borderSide == BORDER_LOWER)
     {
      // After a lower test, a pullback candle is bullish or indecisive,
      // and/or shows a wick rejecting the low, and closes away from the border.
      bool bullishOrFlat = (dir >= 0);
      bool closesAwayFromBorder = (pullbackCandle.close > testCandle.low);
      bool hasRejectionWick = (CandleLowerWick(pullbackCandle) >= CandleBody(pullbackCandle) * 0.5);

      return (bullishOrFlat && closesAwayFromBorder) || hasRejectionWick;
     }

   return false;
  }

//--- 7.6 Continuation detector: checks whether a candle confirms direction
//    after a border test/pullback.
//    referenceCandle: the candle whose extreme/close acts as the local reaction point.
bool IsContinuationCandle(const MqlRates &referenceCandle, const MqlRates &candle, int borderSide)
  {
   int dir = CandleDirection(candle);
   double body = CandleBody(candle);
   double range = CandleRange(candle);
   bool strongBody = (range > 0 && body >= range * 0.5);

   if(borderSide == BORDER_UPPER)
     {
      // Continuation upward: close beyond the local reaction high, bullish, strong body
      bool closesBeyond = (candle.close > referenceCandle.high);
      return (dir > 0 && strongBody && closesBeyond);
     }

   if(borderSide == BORDER_LOWER)
     {
      // Continuation downward: close beyond the local reaction low, bearish, strong body
      bool closesBeyond = (candle.close < referenceCandle.low);
      return (dir < 0 && strongBody && closesBeyond);
     }

   return false;
  }

//--- 7.7 Midpoint reversal detector: evaluates weak border reactions.
//    Returns true if the reaction is weak (small body, weak rejection) and
//    therefore a midpoint reversal candidate instead of continuation.
bool IsMidpointReversalCandidate(const MqlRates &testCandle, const MqlRates &followCandle, int borderSide, bool volumeFading)
  {
   double bodyTest   = CandleBody(testCandle);
   double rangeTest  = CandleRange(testCandle);
   double bodyFollow = CandleBody(followCandle);
   double rangeFollow= CandleRange(followCandle);

   bool weakTestBody   = (rangeTest  > 0 && bodyTest   <= rangeTest  * 0.35);
   bool weakFollowBody = (rangeFollow> 0 && bodyFollow <= rangeFollow* 0.35);

   // A weak rejection: small bodies near the border and fading volume
   bool weakReaction = (weakTestBody || weakFollowBody);

   if(borderSide == BORDER_UPPER)
     {
      // For upper border failure, follow candle should NOT be making strong new highs
      bool noNewHighPush = (followCandle.high <= testCandle.high + PointsToPrice(InpBorderTolerancePoints));
      return (weakReaction && volumeFading && noNewHighPush);
     }

   if(borderSide == BORDER_LOWER)
     {
      bool noNewLowPush = (followCandle.low >= testCandle.low - PointsToPrice(InpBorderTolerancePoints));
      return (weakReaction && volumeFading && noNewLowPush);
     }

   return false;
  }

//====================================================================
// 8) VOLUME CONFIRMATION FUNCTIONS
//====================================================================

//--- 7.8 / Section 8 volume confirmation: compares a candle's tick volume
//    against a baseline average. shift = bar shift of the candle being checked.
//    Returns true if the candle's volume meets or exceeds the configured ratio
//    of the average volume, and is above the minimum threshold.
bool VolumeConfirms(int shift, int lookback)
  {
   long candleVol = GetCandleTickVolume(shift);
   if(candleVol < InpMinVolumeThreshold)
      return false;

   double avgVol = GetAverageTickVolume(shift + 1, lookback);
   if(avgVol <= 0)
      return (candleVol >= InpMinVolumeThreshold); // fallback if no history

   double ratio = (double)candleVol / avgVol;
   return (ratio >= InpVolumeRatioConfirm);
  }

//--- Returns true if volume is fading (current candle volume below average,
//    i.e. ratio below 1.0)
bool VolumeIsFading(int shift, int lookback)
  {
   long candleVol = GetCandleTickVolume(shift);
   double avgVol  = GetAverageTickVolume(shift + 1, lookback);

   if(avgVol <= 0)
      return false;

   double ratio = (double)candleVol / avgVol;
   return (ratio < 1.0);
  }

//====================================================================
// 9) ENTRY DECISION FUNCTION
//====================================================================

// Signal direction codes
#define SIGNAL_NONE  0
#define SIGNAL_BUY   1
#define SIGNAL_SELL -1

//--- Main decision engine. Evaluates the observation window candles
//    (the InpObservationCandles candles after the opening range candle)
//    and returns a signal direction (SIGNAL_BUY / SIGNAL_SELL / SIGNAL_NONE).
//    Also sets g_signalReason and g_signalConfidence as side effects.
int EvaluateSetup()
  {
   g_signalDirection  = SIGNAL_NONE;
   g_signalReason     = "";
   g_signalConfidence = 0.0;

   if(!g_rangeCaptured)
      return SIGNAL_NONE;

   // We need at least 2 candles after the opening range candle to evaluate
   // (test candle + follow candle). Determine how many candles have closed
   // since the range candle.
   // Range candle is at InpRangeCandleShift. Candles after it are at
   // shifts InpRangeCandleShift-1, InpRangeCandleShift-2, ... down to 1 (last closed).
   int availableAfterRange = InpRangeCandleShift - 1; // shift of the most recent candle after the range candle... 
   // availableAfterRange counts how many bars have closed after the range candle:
   // if InpRangeCandleShift == 1, the range candle IS the last closed bar -> 0 candles after it yet.
   int candlesAfterRange = InpRangeCandleShift - 1;

   if(candlesAfterRange < 2)
      return SIGNAL_NONE; // need at least a test candle and a follow candle

   int observe = MathMin(InpObservationCandles, candlesAfterRange);
   if(observe < 2)
      observe = 2;
   if(observe > candlesAfterRange)
      observe = candlesAfterRange;

   // Load 'observe' candles after the range candle, newest-first.
   // Shift 1 = most recently closed candle overall.
   MqlRates obs[];
   if(!GetCandles(1, observe, obs))
      return SIGNAL_NONE;

   // obs[observe-1] is the OLDEST observed candle (closest to the range candle) -> the "test" candle
   // obs[0] is the most recent candle
   int testIdx = observe - 1;
   MqlRates testCandle = obs[testIdx];

   int borderSide = DetectBorderTest(testCandle);
   if(borderSide == BORDER_NONE)
     {
      g_signalReason = "No border test detected";
      return SIGNAL_NONE;
     }

   // Shift in the rates series corresponding to the test candle (for volume lookups)
   int testShift = 1 + testIdx; // shift=1 is obs[0]; obs[testIdx] -> shift = 1+testIdx

   bool testVolumeConfirms = VolumeConfirms(testShift, InpVolumeLookback);
   bool testVolumeFading   = VolumeIsFading(testShift, InpVolumeLookback);

   // --- Walk forward through remaining observed candles to classify pullback/continuation
   bool   pullbackFound      = false;
   bool   continuationFound  = false;
   bool   continuationVolOK  = false;
   MqlRates lastReference = testCandle;
   MqlRates pullbackCandle;
   bool   havePullback = false;

   for(int idx = testIdx - 1; idx >= 0; idx--)
     {
      MqlRates candle = obs[idx];
      int shift = 1 + idx;

      if(!havePullback)
        {
         if(IsPullbackCandle(testCandle, candle, borderSide))
           {
            pullbackFound = true;
            havePullback  = true;
            pullbackCandle = candle;
            lastReference  = candle;
            continue;
           }
         else
           {
            // No pullback; check directly for continuation from the test candle
            if(IsContinuationCandle(testCandle, candle, borderSide))
              {
               continuationFound = true;
               continuationVolOK = VolumeConfirms(shift, InpVolumeLookback);
               lastReference = candle;
               break;
              }
           }
        }
      else
        {
         // We already have a pullback; check this candle for continuation
         if(IsContinuationCandle(pullbackCandle, candle, borderSide))
           {
            continuationFound = true;
            continuationVolOK = VolumeConfirms(shift, InpVolumeLookback);
            lastReference = candle;
            break;
           }
        }
     }

   // Check for engulfing patterns as additional confirmation context
   bool engulfingSupport = false;
   if(observe >= 2)
     {
      MqlRates prevC = obs[testIdx];
      MqlRates currC = obs[testIdx - 1];
      if(borderSide == BORDER_UPPER && IsBullishEngulfing(prevC, currC))
         engulfingSupport = true;
      if(borderSide == BORDER_LOWER && IsBearishEngulfing(prevC, currC))
         engulfingSupport = true;
     }

   // Hammer / shooting star context at the test candle
   bool hammerAtTest        = IsHammer(testCandle);
   bool shootingStarAtTest  = IsShootingStar(testCandle);

   //--- Signal hierarchy ---------------------------------------------------
   // 1) Strong continuation with volume confirmation
   if(continuationFound && continuationVolOK)
     {
      if(borderSide == BORDER_UPPER)
        {
         g_signalDirection  = SIGNAL_BUY;
         g_signalReason     = "Upper border continuation with volume confirmation";
         g_signalConfidence = 0.9;
        }
      else
        {
         g_signalDirection  = SIGNAL_SELL;
         g_signalReason     = "Lower border continuation with volume confirmation";
         g_signalConfidence = 0.9;
        }
      if(engulfingSupport) g_signalConfidence = MathMin(1.0, g_signalConfidence + 0.05);
      return g_signalDirection;
     }

   // 2) Valid pullback followed by continuation (volume not strictly required)
   if(pullbackFound && continuationFound)
     {
      if(borderSide == BORDER_UPPER)
        {
         g_signalDirection  = SIGNAL_BUY;
         g_signalReason     = "Upper border pullback then continuation";
         g_signalConfidence = 0.7;
        }
      else
        {
         g_signalDirection  = SIGNAL_SELL;
         g_signalReason     = "Lower border pullback then continuation";
         g_signalConfidence = 0.7;
        }
      return g_signalDirection;
     }

   // 3) Weak border rejection with midpoint reversal
   if(InpAllowMidpointReversal)
     {
      MqlRates followCandle = (observe >= 2) ? obs[testIdx - 1] : testCandle;
      int followShift = (observe >= 2) ? (1 + (testIdx - 1)) : testShift;

      bool volFading = testVolumeFading || VolumeIsFading(followShift, InpVolumeLookback);
      bool weakRejection = IsMidpointReversalCandidate(testCandle, followCandle, borderSide, volFading);

      // Strong rejection: hammer at upper->bearish reversal context or shooting star at lower
      bool strongRejectionPattern =
         (borderSide == BORDER_UPPER && shootingStarAtTest) ||
         (borderSide == BORDER_LOWER && hammerAtTest);

      if(weakRejection || (strongRejectionPattern && testVolumeFading))
        {
         if(borderSide == BORDER_UPPER)
           {
            // Upper border failure: sell toward midpoint
            g_signalDirection  = SIGNAL_SELL;
            g_signalReason     = "Upper border weak rejection - midpoint reversal (sell)";
            g_signalConfidence = 0.5;
           }
         else
           {
            // Lower border failure: buy toward midpoint
            g_signalDirection  = SIGNAL_BUY;
            g_signalReason     = "Lower border weak rejection - midpoint reversal (buy)";
            g_signalConfidence = 0.5;
           }
         return g_signalDirection;
        }
     }

   // 4) Ignore ambiguous candles
   g_signalReason = "Ambiguous setup - no trade";
   return SIGNAL_NONE;
  }

//====================================================================
// SUPPORTING CHECKS (spread / position)
//====================================================================

//--- Spread filter: returns true if current spread is acceptable
bool SpreadIsAcceptable()
  {
   double spreadPoints = (double)SymbolInfoInteger(_Symbol, SYMBOL_SPREAD);
   return (spreadPoints <= InpMaxSpreadPoints);
  }

//--- Returns true if there is no open position for this symbol/magic
bool NoOpenPosition()
  {
   for(int i = PositionsTotal() - 1; i >= 0; i--)
     {
      ulong ticket = PositionGetTicket(i);
      if(ticket == 0) continue;
      if(PositionGetString(POSITION_SYMBOL) == _Symbol)
         return false;
     }
   return true;
  }

//====================================================================
// 10) TRADE EXECUTION FUNCTION
//====================================================================

//--- Places a market trade in the given direction, with SL/TP based on
//    the opening range and configured point distances.
bool PlaceTrade(int signal)
  {
   if(signal == SIGNAL_NONE)
      return false;

   double ask = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
   double bid = SymbolInfoDouble(_Symbol, SYMBOL_BID);

   double slDistance = PointsToPrice(InpStopLossPoints);
   double tpDistance = PointsToPrice(InpTakeProfitPoints);

   bool isMidpointTrade = (StringFind(g_signalReason, "midpoint") >= 0);

   double price = 0.0, sl = 0.0, tp = 0.0;
   bool ok = false;

   if(signal == SIGNAL_BUY)
     {
      price = ask;

      // Stop loss: opposite side of the opening range (the lower border),
      // or fixed distance, whichever is tighter/valid.
      double slFromRange = g_rangeLow - PointsToPrice(InpBorderTolerancePoints);
      double slFixed     = price - slDistance;
      sl = (slFromRange < price) ? MathMax(slFromRange, slFixed) : slFixed;

      if(isMidpointTrade)
        {
         // Target the midpoint for reversal trades
         tp = g_rangeMid;
         if(tp <= price) tp = price + tpDistance; // safety fallback
        }
      else
        {
         // Continuation trade: fixed reward multiple / range projection
         double rangeSize = g_rangeHigh - g_rangeLow;
         double projTarget = price + rangeSize;
         tp = MathMax(price + tpDistance, projTarget);
        }

      ok = trade.Buy(InpLots, _Symbol, price, sl, tp, "ORB-NY Buy");
     }
   else if(signal == SIGNAL_SELL)
     {
      price = bid;

      // Stop loss: opposite side of the opening range (the upper border)
      double slFromRange = g_rangeHigh + PointsToPrice(InpBorderTolerancePoints);
      double slFixed     = price + slDistance;
      sl = (slFromRange > price) ? MathMin(slFromRange, slFixed) : slFixed;

      if(isMidpointTrade)
        {
         tp = g_rangeMid;
         if(tp >= price) tp = price - tpDistance; // safety fallback
        }
      else
        {
         double rangeSize = g_rangeHigh - g_rangeLow;
         double projTarget = price - rangeSize;
         tp = MathMin(price - tpDistance, projTarget);
        }

      ok = trade.Sell(InpLots, _Symbol, price, sl, tp, "ORB-NY Sell");
     }

   if(ok)
     {
      Print("ORB EA: Trade placed. Direction=", (signal == SIGNAL_BUY ? "BUY" : "SELL"),
            " Reason=", g_signalReason,
            " Confidence=", DoubleToString(g_signalConfidence, 2),
            " Price=", DoubleToString(price, _Digits),
            " SL=", DoubleToString(sl, _Digits),
            " TP=", DoubleToString(tp, _Digits));
     }
   else
     {
      Print("ORB EA: Trade execution FAILED. Error=", GetLastError(),
            " RetCode=", trade.ResultRetcode(),
            " Comment=", trade.ResultRetcodeDescription());
     }

   return ok;
  }

//====================================================================
// 11) POSITION MANAGEMENT FUNCTION
//====================================================================

//--- Manages any open position for this symbol: trailing stop, session-end exit.
void ManagePosition()
  {
   for(int i = PositionsTotal() - 1; i >= 0; i--)
     {
      ulong ticket = PositionGetTicket(i);
      if(ticket == 0) continue;
      if(PositionGetString(POSITION_SYMBOL) != _Symbol) continue;
      if(!PositionSelectByTicket(ticket)) continue;

      long   posType  = PositionGetInteger(POSITION_TYPE);
      double openPrice= PositionGetDouble(POSITION_PRICE_OPEN);
      double currSL   = PositionGetDouble(POSITION_SL);
      double currTP   = PositionGetDouble(POSITION_TP);

      // --- Session end exit ---
      datetime now = TimeCurrent();
      if(InpUseTimeFilter && now > g_sessionEnd)
        {
         trade.PositionClose(ticket);
         Print("ORB EA: Session ended - closing position #", ticket);
         continue;
        }

      // --- Trailing stop ---
      if(!InpUseTrailingStop)
         continue;

      double trailStart = PointsToPrice(InpTrailingStartPoints);
      double trailStep  = PointsToPrice(InpTrailingStepPoints);

      if(posType == POSITION_TYPE_BUY)
        {
         double bid = SymbolInfoDouble(_Symbol, SYMBOL_BID);
         double profit = bid - openPrice;

         if(profit >= trailStart)
           {
            double newSL = bid - trailStep;
            if(newSL > currSL && newSL > openPrice)
              {
               trade.PositionModify(ticket, newSL, currTP);
              }
           }
        }
      else if(posType == POSITION_TYPE_SELL)
        {
         double ask = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
         double profit = openPrice - ask;

         if(profit >= trailStart)
           {
            double newSL = ask + trailStep;
            if((currSL == 0 || newSL < currSL) && newSL < openPrice)
              {
               trade.PositionModify(ticket, newSL, currTP);
              }
           }
        }
     }
  }

//====================================================================
// BAR-CHANGE CONTROL
//====================================================================

//--- Returns true exactly once per new M15 bar
bool IsNewBar()
  {
   datetime currentBarTime = (datetime)SeriesInfoInteger(_Symbol, PERIOD_M15, SERIES_LASTBAR_DATE);

   if(currentBarTime == 0)
     {
      // Fallback using iTime-equivalent via CopyRates
      MqlRates r[];
      ArraySetAsSeries(r, true);
      if(CopyRates(_Symbol, PERIOD_M15, 0, 1, r) <= 0)
         return false;
      currentBarTime = r[0].time;
     }

   if(currentBarTime != g_lastBarTime)
     {
      g_lastBarTime = currentBarTime;
      return true;
     }

   return false;
  }

//====================================================================
// 12) OnInit()
//====================================================================

int OnInit()
  {
   if(InpObservationCandles < 2 || InpObservationCandles > 3)
     {
      Print("ORB EA: InpObservationCandles must be 2 or 3. Clamping to valid range.");
     }

   if(InpRangeCandleShift < 1)
     {
      Print("ORB EA: InpRangeCandleShift must be >= 1.");
      return INIT_PARAMETERS_INCORRECT;
     }

   trade.SetExpertMagicNumber(20240601);
   trade.SetDeviationInPoints(20);

   g_lastBarTime = 0;
   ResetSessionState(TimeCurrent());

   Print("ORB EA: Initialized on symbol ", _Symbol, " Period=", EnumToString((ENUM_TIMEFRAMES)Period()));

   return(INIT_SUCCEEDED);
  }

//====================================================================
// 13) OnTick()
//====================================================================

void OnTick()
  {
   datetime now = TimeCurrent();

   // 1. Time/session filter (also handles daily reset)
   if(!IsWithinTradingWindow(now))
     {
      // Even outside the window, manage any leftover position (session-end close)
      ManagePosition();
      return;
     }

   // 2. Only process once per new M15 bar
   if(!IsNewBar())
     {
      // Still manage open positions intra-bar (trailing stop)
      ManagePosition();
      return;
     }

   // 3. Capture the opening range if not yet captured
   if(!g_rangeCaptured)
     {
      if(CanCaptureOpeningRange())
        {
         CaptureOpeningRange();
         g_rangeCaptured = true;
        }
      ManagePosition();
      return;
     }

   // 4. If a trade has already been placed (one-trade rule), just manage position
   if(g_tradePlaced && InpOneTradeOnly)
     {
      ManagePosition();
      return;
     }

   if(g_tradesThisSession >= InpMaxTradesPerSession)
     {
      ManagePosition();
      return;
     }

   // 5. Evaluate the observation window for a signal
   int signal = EvaluateSetup();

   // 6. If valid, check spread and position, then execute
   if(signal != SIGNAL_NONE)
     {
      if(SpreadIsAcceptable() && NoOpenPosition())
        {
         if(PlaceTrade(signal))
           {
            g_tradePlaced = true;
            g_tradesThisSession++;
           }
        }
      else
        {
         Print("ORB EA: Signal generated (", g_signalReason, ") but trade skipped. ",
               "SpreadOK=", SpreadIsAcceptable(), " NoOpenPos=", NoOpenPosition());
        }
     }

   // 7. Manage any open position
   ManagePosition();
  }

//====================================================================
// 14) OnDeinit()
//====================================================================

void OnDeinit(const int reason)
  {
   Print("ORB EA: Deinitialized. Reason=", reason);
  }
//+------------------------------------------------------------------+