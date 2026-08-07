//+------------------------------------------------------------------+
//|                                                      ProjectName |
//|                                      Copyright 2020, CompanyName |
//|                                       http://www.companyname.net |
//+------------------------------------------------------------------+
enum EVOLATILITY
  {
   VOLATILITY_LOW,
   VOLATILITY_NORMAL,
   VOLATILITY_HIGH,
   VOLATILITY_VERY_HIGH
  };

//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
EVOLATILITY GetCurrentVolatility(int atrPeriod = 14)
  {
   static int atrHandle = INVALID_HANDLE;
   if(atrHandle == INVALID_HANDLE)
      atrHandle = iATR(_Symbol, PERIOD_CURRENT, atrPeriod); // Bug 2 fix: use atrPeriod

   double atr[];
   ArrayResize(atr, atrPeriod);
   if(CopyBuffer(atrHandle, 0, 0, atrPeriod, atr) != atrPeriod)
      return VOLATILITY_NORMAL;

   double avgATR = 0.0;
   // Average excluding the current ATR bar (index 0)
   for(int i = 1; i < atrPeriod; i++)
      avgATR += atr[i];
   avgATR /= (atrPeriod - 1); // Bug 1 fix: was 99.0, should be 13 (for default period)

   if(avgATR <= 0.0)
      return VOLATILITY_NORMAL;

   double score = atr[0] / avgATR;

   if(score < 0.80)  return VOLATILITY_LOW;
   if(score < 1.20)  return VOLATILITY_NORMAL;
   if(score < 1.50)  return VOLATILITY_HIGH;
   return VOLATILITY_VERY_HIGH;
  }

//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
double CalculateVolumeByPercent(double            percent,
                                ENUM_ORDER_TYPE   orderType = ORDER_TYPE_BUY)
  {
//--- 1. Validate input
   if(percent <= 0.0 || percent > 100.0)
     {
      PrintFormat("%s: 'percent' must be in (0, 100]. Received: %.4f",
                  __FUNCTION__, percent);
      return 0.0;
     }

   string symbol = Symbol();

//--- 2. Free margin guard
   double freeMargin = AccountInfoDouble(ACCOUNT_MARGIN_FREE);
   if(freeMargin <= 0.0)
     {
      PrintFormat("%s: No free margin available (%.2f).", __FUNCTION__, freeMargin);
      return 0.0;
     }

//--- 3. Price for the requested direction
   bool isBuy = (orderType == ORDER_TYPE_BUY            ||
                 orderType == ORDER_TYPE_BUY_LIMIT       ||
                 orderType == ORDER_TYPE_BUY_STOP        ||
                 orderType == ORDER_TYPE_BUY_STOP_LIMIT);

   double price = isBuy ? SymbolInfoDouble(symbol, SYMBOL_ASK)
                  : SymbolInfoDouble(symbol, SYMBOL_BID);

   if(price <= 0.0)
     {
      PrintFormat("%s: Invalid price (%.5f) for %s.", __FUNCTION__, price, symbol);
      return 0.0;
     }

//--- 4. Margin cost of exactly 1 lot (broker / leverage / hedging-aware)
   double marginPer1Lot = 0.0;
   if(!OrderCalcMargin(orderType, symbol, 1.0, price, marginPer1Lot) ||
      marginPer1Lot <= 0.0)
     {
      PrintFormat("%s: OrderCalcMargin() failed for %s. Error: %d",
                  __FUNCTION__, symbol, GetLastError());
      return 0.0;
     }

//--- 5. Symbol volume constraints
   double lotStep = SymbolInfoDouble(symbol, SYMBOL_VOLUME_STEP);
   double lotMin  = SymbolInfoDouble(symbol, SYMBOL_VOLUME_MIN);
   double lotMax  = SymbolInfoDouble(symbol, SYMBOL_VOLUME_MAX);

   if(lotStep <= 0.0)
     {
      PrintFormat("%s: Invalid VOLUME_STEP (%.8f) for %s.", __FUNCTION__, lotStep, symbol);
      return 0.0;
     }

//--- 6. Raw lot from the requested margin budget
   double marginBudget = freeMargin * (percent / 100.0);
   double rawLot       = marginBudget / marginPer1Lot;

//--- 7. Floor to the nearest lot step (never exceed the intended budget)
   int    digits = (int)MathRound(MathLog10(1.0 / lotStep));  // e.g. step 0.01 → 2 dp
   double lot    = NormalizeDouble(MathFloor(rawLot / lotStep) * lotStep, digits);

//--- 8. Below-minimum handling  ← key fix
   if(lot < lotMin)
     {
      double marginForMin = marginPer1Lot * lotMin;

      if(freeMargin >= marginForMin)
        {
         // Enough margin exists for the minimum lot – clamp up and warn.
         PrintFormat("%s: Budget %.2f (%.1f%% of %.2f) covers only %.5f lots "
                     "(min=%.5f, margin/lot=%.2f). Clamping to VOLUME_MIN.",
                     __FUNCTION__, marginBudget, percent, freeMargin,
                     lot, lotMin, marginPer1Lot);
         lot = lotMin;
        }
      else
        {
         // Cannot afford even the minimum lot – skip the trade.
         PrintFormat("%s: Insufficient margin. Need %.2f for VOLUME_MIN %.5f "
                     "but free margin is %.2f. Returning 0.",
                     __FUNCTION__, marginForMin, lotMin, freeMargin);
         return 0.0;
        }
     }

//--- 9. Cap at maximum allowed volume
   if(lot > lotMax)
     {
      PrintFormat("%s: Clamping lot %.5f to VOLUME_MAX %.5f.", __FUNCTION__, lot, lotMax);
      lot = NormalizeDouble(lotMax, digits);
     }

   PrintFormat("%s: percent=%.2f%% | free=%.2f | budget=%.2f | "
               "margin/lot=%.2f | lots=%.5f",
               __FUNCTION__, percent, freeMargin, marginBudget, marginPer1Lot, lot);

   return lot;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
string GetTimeframeString(ENUM_TIMEFRAMES tf)
  {
   switch(tf)
     {
      case PERIOD_M1:
         return "M1";
      case PERIOD_M5:
         return "M5";
      case PERIOD_M15:
         return "M15";
      case PERIOD_M30:
         return "M30";
      case PERIOD_H1:
         return "H1";
      case PERIOD_H4:
         return "H4";
      case PERIOD_D1:
         return "D1";
      default:
         return "Unknown TF";
     }
  }
//+------------------------------------------------------------------+
