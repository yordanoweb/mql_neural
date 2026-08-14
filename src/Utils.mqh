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

double g_score = 0.0;

//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
EVOLATILITY GetCurrentVolatility(int atrPeriod = 14)
  {
   static int atrHandle = INVALID_HANDLE;
   if(atrHandle == INVALID_HANDLE)
      atrHandle = iATR(_Symbol, PERIOD_CURRENT, atrPeriod);

   double atr[];
   ArraySetAsSeries(atr, true); // FIX: atr[0]=current, atr[1]=prev, ...
   ArrayResize(atr, atrPeriod);
   if(CopyBuffer(atrHandle, 0, 0, atrPeriod, atr) != atrPeriod)
      return VOLATILITY_NORMAL;

   double avgATR = 0.0;
   for(int i = 1; i < atrPeriod; i++)
      avgATR += atr[i];
   avgATR /= (atrPeriod - 1);

   if(avgATR <= 0.0)
      return VOLATILITY_NORMAL;

   double score = atr[0] / avgATR; // now truly the current bar

   if(score < 0.80)
      return VOLATILITY_LOW;
   if(score < 1.20)
      return VOLATILITY_NORMAL;
   if(score < 1.50)
      return VOLATILITY_HIGH;

   g_score = score;

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
   if(tf == PERIOD_CURRENT)
      tf = (ENUM_TIMEFRAMES)_Period;

   // Converts PERIOD_M1 -> "PERIOD_M1" -> returns "M1"
   string tfStr = EnumToString(tf);
   StringReplace(tfStr, "PERIOD_", "");
   
   return tfStr;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool GetIndicators()
  {
   if(g_rsi_handle == INVALID_HANDLE)
      return false;

   ArraySetAsSeries(g_rsi_buffer, true);
   if(CopyBuffer(g_rsi_handle, 0, 0, InpWindow, g_rsi_buffer) < InpWindow)
      return false;

   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool BuildInputBuffer()
  {
   if(ArraySize(g_close) < InpWindow || ArraySize(g_open) < InpWindow ||
      ArraySize(g_high) < InpWindow || ArraySize(g_low) < InpWindow ||
      ArraySize(g_rsi_buffer) < InpWindow)
      return false;

   ArrayResize(g_input_buffer, InpWindow * FEATURES);

   for(int i=0; i < InpWindow; i++)
     {
      int mql_idx = InpWindow - 1 - i;
      g_input_buffer[i * 3 + 0] = (float)(g_close[mql_idx] - g_open[mql_idx]);
      g_input_buffer[i * 3 + 1] = (float)(g_high[mql_idx] - g_low[mql_idx]);
      g_input_buffer[i * 3 + 2] = (float)(g_rsi_buffer[mql_idx] / 100.0);
     }

   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool GetData()
  {
   ArraySetAsSeries(g_close, true);
   ArraySetAsSeries(g_open, true);
   ArraySetAsSeries(g_high, true);
   ArraySetAsSeries(g_low, true);

   if(CopyClose(_Symbol, _Period, 0, InpWindow + 15, g_close) < InpWindow + 15)
      return false;
   if(CopyOpen(_Symbol, _Period, 0, InpWindow, g_open) < InpWindow)
      return false;
   if(CopyHigh(_Symbol, _Period, 0, InpWindow, g_high) < InpWindow)
      return false;
   if(CopyLow(_Symbol, _Period, 0, InpWindow, g_low) < InpWindow)
      return false;

   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
string GetDealReasonText(const long reason)
  {
   switch((ENUM_DEAL_REASON)reason)
     {
      case DEAL_REASON_SL:
         return "STOP_LOSS";
      case DEAL_REASON_TP:
         return "TAKE_PROFIT";
      case DEAL_REASON_SO:
         return "STOP_OUT";
      case DEAL_REASON_CLIENT:
         return "MANUAL_CLIENT";
      case DEAL_REASON_MOBILE:
         return "MANUAL_MOBILE";
      case DEAL_REASON_WEB:
         return "MANUAL_WEB";
      case DEAL_REASON_EXPERT:
         return "EXPERT";
      default:
         return "OTHER";
     }
  }
//+------------------------------------------------------------------+
//| Calcula el Stop Loss basado en el ATR                            |
//+------------------------------------------------------------------+
double GetStopLoss(int atrPeriod, ENUM_ORDER_TYPE orderType = ORDER_TYPE_BUY)
  {
// Usamos static para mantener el handle del indicador en memoria
   static int atr_handle = INVALID_HANDLE;
   static int current_atr_period = 0;

// Solo creamos el handle si no existe o si el periodo cambia
   if(atr_handle == INVALID_HANDLE || current_atr_period != atrPeriod)
     {
      if(atr_handle != INVALID_HANDLE)
         IndicatorRelease(atr_handle);

      atr_handle = iATR(_Symbol, _Period, atrPeriod);
      current_atr_period = atrPeriod;

      if(atr_handle == INVALID_HANDLE)
        {
         Print("Error al inicializar ATR para Stop Loss: ", GetLastError());
         return 0.0;
        }
     }

   double atr_array[];
   ArraySetAsSeries(atr_array, true);

// Copiamos el valor de la vela anterior (índice 1) para evitar repintado
   if(CopyBuffer(atr_handle, 0, 1, 1, atr_array) <= 0)
     {
      Print("Error al copiar datos de ATR para Stop Loss: ", GetLastError());
      return 0.0;
     }

   double atr_value = atr_array[0];
   double sl_price = 0.0;

// Cálculo del nivel de precio exacto
   if(orderType == ORDER_TYPE_BUY)
     {
      double ask = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
      sl_price = ask - atr_value; // En compras, el SL va por debajo del Ask
     }
   else
      if(orderType == ORDER_TYPE_SELL)
        {
         double bid = SymbolInfoDouble(_Symbol, SYMBOL_BID);
         sl_price = bid + atr_value; // En ventas, el SL va por encima del Bid
        }

   return NormalizeDouble(sl_price, _Digits);
  }

//+------------------------------------------------------------------+
//| Calcula el Take Profit basado en el ATR                          |
//+------------------------------------------------------------------+
double GetTakeProfit(int atrPeriod, ENUM_ORDER_TYPE orderType = ORDER_TYPE_BUY)
  {
// Usamos static para mantener el handle del indicador en memoria
   static int atr_handle_tp = INVALID_HANDLE;
   static int current_atr_period_tp = 0;

// Solo creamos el handle si no existe o si el periodo cambia
   if(atr_handle_tp == INVALID_HANDLE || current_atr_period_tp != atrPeriod)
     {
      if(atr_handle_tp != INVALID_HANDLE)
         IndicatorRelease(atr_handle_tp);

      atr_handle_tp = iATR(_Symbol, _Period, atrPeriod);
      current_atr_period_tp = atrPeriod;

      if(atr_handle_tp == INVALID_HANDLE)
        {
         Print("Error al inicializar ATR para Take Profit: ", GetLastError());
         return 0.0;
        }
     }

   double atr_array[];
   ArraySetAsSeries(atr_array, true);

// Copiamos el valor de la vela anterior (índice 1) para mayor estabilidad
   if(CopyBuffer(atr_handle_tp, 0, 1, 1, atr_array) <= 0)
     {
      Print("Error al copiar datos de ATR para Take Profit: ", GetLastError());
      return 0.0;
     }

   double atr_value = atr_array[0];
   double tp_price = 0.0;

// Cálculo del nivel de precio exacto
   if(orderType == ORDER_TYPE_BUY)
     {
      double ask = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
      tp_price = ask + atr_value; // En compras, el TP va por encima del Ask
     }
   else
      if(orderType == ORDER_TYPE_SELL)
        {
         double bid = SymbolInfoDouble(_Symbol, SYMBOL_BID);
         tp_price = bid - atr_value; // En ventas, el TP va por debajo del Bid
        }

   return NormalizeDouble(tp_price, _Digits);
  }
//+------------------------------------------------------------------+
//| Saves current Expert Advisor input parameters to a .set file     |
//| Parameters:                                                      |
//|   file_name    - Base name of the destination file (e.g., "EA.set") |
//|   common_folder- True to save to Common/Files, False for local Files|
//+------------------------------------------------------------------+
bool SaveCurrentExperAdvisorInputs(string file_name = "EA_Settings.set", bool common_folder = false)
  {
// 1. Generate YYYYDDMMHHmmss timestamp prefix
   MqlDateTime dt;
   TimeToStruct(TimeCurrent(), dt);

// Formats: YYYY (year), DD (day), MM (month), HH (hour), mm (minute), ss (second)
   string timestamp = StringFormat("%04d%02d%02d%02d%02d%02d_",
                                   dt.year,
                                   dt.day,
                                   dt.mon,
                                   dt.hour,
                                   dt.min,
                                   dt.sec);

   string final_file_name = timestamp + file_name;

// 2. Set file flags (Text mode, ANSI encoding, Write access)
   int flags = FILE_WRITE | FILE_TXT | FILE_ANSI;
   if(common_folder)
      flags |= FILE_COMMON;

// 3. Open the file
   int file_handle = FileOpen(final_file_name, flags);
   if(file_handle == INVALID_HANDLE)
     {
      PrintFormat("Error: Failed to open file '%s' for writing. Code: %d", final_file_name, GetLastError());
      return false;
     }

// 4. Write Header
   FileWriteString(file_handle, "; Expert Advisor Saved Inputs\r\n");
   FileWriteString(file_handle, StringFormat("; Saved on: %s\r\n\r\n", TimeToString(TimeCurrent(), TIME_DATE | TIME_MINUTES)));

// 5. Write Input Variables
   FileWriteString(file_handle, StringFormat("InpModelFile=%s\r\n", InpModelFile));
   FileWriteString(file_handle, StringFormat("InpMinConf=%f\r\n", InpMinConf));
   FileWriteString(file_handle, StringFormat("InpStartHour=%d\r\n", InpStartHour));
   FileWriteString(file_handle, StringFormat("InpEndHour=%d\r\n", InpEndHour));
   FileWriteString(file_handle, StringFormat("InpWindow=%d\r\n", InpWindow));
   FileWriteString(file_handle, StringFormat("InpMirrorEntryOperation=%s\r\n", InpMirrorEntryOperation ? "true" : "false"));
   FileWriteString(file_handle, StringFormat("InpLot=%.2f\r\n", InpLot));
   FileWriteString(file_handle, StringFormat("InpMagic=%d\r\n", InpMagic));
   FileWriteString(file_handle, StringFormat("InpSLATR=%d\r\n", InpSLATR));
   FileWriteString(file_handle, StringFormat("InpTPATR=%d\r\n", InpTPATR));
   FileWriteString(file_handle, StringFormat("InpMinDollars=%d\r\n", InpMinDollars));
   FileWriteString(file_handle, StringFormat("InpUseSL=%s\r\n", InpUseSL ? "true" : "false"));
   FileWriteString(file_handle, StringFormat("InpCooldownBars=%d\r\n", InpCooldownBars));
   FileWriteString(file_handle, StringFormat("InpRequirePrevCandleDir=%s\r\n", InpRequirePrevCandleDir ? "true" : "false"));
   FileWriteString(file_handle, StringFormat("InpRequireCurrCandleDir=%s\r\n", InpRequireCurrCandleDir ? "true" : "false"));
   FileWriteString(file_handle, StringFormat("InpRequireVolatility=%s\r\n", InpRequireVolatility ? "true" : "false"));
   FileWriteString(file_handle, StringFormat("InpVolatilityPeriod=%d\r\n", InpVolatilityPeriod));
   FileWriteString(file_handle, StringFormat("InpTimerSeconds=%d\r\n", InpTimerSeconds));
   FileWriteString(file_handle, StringFormat("InpDebug=%s\r\n", InpDebug ? "true" : "false"));

// 6. Flush and close handle
   FileClose(file_handle);

   PrintFormat("Success: Current EA inputs saved to '%s'", final_file_name);
   return true;
  }
//+------------------------------------------------------------------+
