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
      double bid = SymbolInfoDouble(_Symbol, SYMBOL_BID);
      sl_price = bid - atr_value; // En compras se cierra al Bid, anclar SL al lado de cierre
     }
   else
      if(orderType == ORDER_TYPE_SELL)
        {
         double ask = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
         sl_price = ask + atr_value; // En ventas se cierra al Ask, anclar SL al lado de cierre
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
      double bid = SymbolInfoDouble(_Symbol, SYMBOL_BID);
      tp_price = bid + atr_value; // En compras se cierra al Bid, anclar TP al lado de cierre
     }
   else
      if(orderType == ORDER_TYPE_SELL)
        {
         double ask = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
         tp_price = ask - atr_value; // En ventas se cierra al Ask, anclar TP al lado de cierre
        }

   return NormalizeDouble(tp_price, _Digits);
  }

//+------------------------------------------------------------------+
//| Normalize SL/TP to broker minimum stop/freeze distances          |
//+------------------------------------------------------------------+
void NormalizeStopsForBroker(ENUM_ORDER_TYPE orderType,
                             double price,
                             double &sl,
                             double &tp,
                             bool use_sl = true)
  {
   bool is_buy = (orderType == ORDER_TYPE_BUY            ||
                  orderType == ORDER_TYPE_BUY_LIMIT       ||
                  orderType == ORDER_TYPE_BUY_STOP        ||
                  orderType == ORDER_TYPE_BUY_STOP_LIMIT);

   if(price <= 0.0)
      price = SymbolInfoDouble(_Symbol, is_buy ? SYMBOL_ASK : SYMBOL_BID);

   double stop_reference_price = SymbolInfoDouble(_Symbol, is_buy ? SYMBOL_BID : SYMBOL_ASK);

   if(price <= 0.0 || stop_reference_price <= 0.0)
     {
      PrintFormat("%s: invalid execution/close price for %s. Keeping original SL/TP.", __FUNCTION__, _Symbol);
      return;
     }

   int digits = (int)SymbolInfoInteger(_Symbol, SYMBOL_DIGITS);
   if(digits <= 0)
      digits = _Digits;

   double point = SymbolInfoDouble(_Symbol, SYMBOL_POINT);
   if(point <= 0.0)
      point = _Point;

   int stops_level_points = (int)SymbolInfoInteger(_Symbol, SYMBOL_TRADE_STOPS_LEVEL);
   int freeze_level_points = (int)SymbolInfoInteger(_Symbol, SYMBOL_TRADE_FREEZE_LEVEL);
   int required_points = MathMax(stops_level_points, freeze_level_points);
   double min_distance = MathMax((double)required_points * point, point);

   double sl_before = sl;
   double tp_before = tp;

   if(use_sl)
     {
      if(is_buy)
        {
         double max_sl = stop_reference_price - min_distance;
         if(sl <= 0.0 || sl > max_sl)
            sl = max_sl;
         else
            sl = MathMin(sl, max_sl);
        }
      else
        {
         double min_sl = stop_reference_price + min_distance;
         if(sl <= 0.0 || sl < min_sl)
            sl = min_sl;
         else
            sl = MathMax(sl, min_sl);
        }

      sl = NormalizeDouble(sl, digits);

      if(is_buy)
        {
         if((stop_reference_price - sl) < min_distance || sl >= stop_reference_price)
            sl = NormalizeDouble(stop_reference_price - min_distance, digits);
         if(sl >= stop_reference_price)
            sl = NormalizeDouble(stop_reference_price - point, digits);
        }
      else
        {
         if((sl - stop_reference_price) < min_distance || sl <= stop_reference_price)
            sl = NormalizeDouble(stop_reference_price + min_distance, digits);
         if(sl <= stop_reference_price)
            sl = NormalizeDouble(stop_reference_price + point, digits);
        }
     }
   else
      sl = 0.0;

   if(is_buy)
     {
      double min_tp = stop_reference_price + min_distance;
      if(tp <= 0.0 || tp < min_tp)
         tp = min_tp;
      else
         tp = MathMax(tp, min_tp);
     }
   else
     {
      double max_tp = stop_reference_price - min_distance;
      if(tp <= 0.0 || tp > max_tp)
         tp = max_tp;
      else
         tp = MathMin(tp, max_tp);
     }

   tp = NormalizeDouble(tp, digits);

   if(is_buy)
     {
      if((tp - stop_reference_price) < min_distance || tp <= stop_reference_price)
         tp = NormalizeDouble(stop_reference_price + min_distance, digits);
      if(tp <= stop_reference_price)
         tp = NormalizeDouble(stop_reference_price + point, digits);
     }
   else
     {
      if((stop_reference_price - tp) < min_distance || tp >= stop_reference_price)
         tp = NormalizeDouble(stop_reference_price - min_distance, digits);
      if(tp >= stop_reference_price)
         tp = NormalizeDouble(stop_reference_price - point, digits);
     }

   if(MathAbs(sl_before - sl) > (point * 0.1) || MathAbs(tp_before - tp) > (point * 0.1))
     {
      PrintFormat("Stops normalized | Symbol=%s | Side=%s | ExecPrice=%.*f | ClosePrice=%.*f | MinDist=%.*f (%d pts; stops=%d freeze=%d) | SL %.*f -> %.*f | TP %.*f -> %.*f",
                  _Symbol,
                  (is_buy ? "BUY" : "SELL"),
                  digits, price,
                  digits, stop_reference_price,
                  digits, min_distance,
                  required_points, stops_level_points, freeze_level_points,
                  digits, sl_before, digits, sl,
                  digits, tp_before, digits, tp);
     }
  }

//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool IsBuyOrderType(ENUM_ORDER_TYPE orderType)
  {
   return (orderType == ORDER_TYPE_BUY            ||
           orderType == ORDER_TYPE_BUY_LIMIT       ||
           orderType == ORDER_TYPE_BUY_STOP        ||
           orderType == ORDER_TYPE_BUY_STOP_LIMIT);
  }

//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
ENUM_ORDER_TYPE_FILLING GetPreferredFillingMode()
  {
   long filling_modes = SymbolInfoInteger(_Symbol, SYMBOL_FILLING_MODE);
   if((filling_modes & SYMBOL_FILLING_FOK) == SYMBOL_FILLING_FOK)
      return ORDER_FILLING_FOK;
   if((filling_modes & SYMBOL_FILLING_IOC) == SYMBOL_FILLING_IOC)
      return ORDER_FILLING_IOC;
   return ORDER_FILLING_RETURN;
  }

//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
double GetExecutionPriceBySide(ENUM_ORDER_TYPE orderType)
  {
   return SymbolInfoDouble(_Symbol, IsBuyOrderType(orderType) ? SYMBOL_ASK : SYMBOL_BID);
  }

//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
double GetClosePriceBySide(ENUM_ORDER_TYPE orderType)
  {
   return SymbolInfoDouble(_Symbol, IsBuyOrderType(orderType) ? SYMBOL_BID : SYMBOL_ASK);
  }

//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void WidenStopsByStep(ENUM_ORDER_TYPE orderType,
                      double price,
                      double &sl,
                      double &tp,
                      bool use_sl,
                      int step_multiplier)
  {
   int digits = (int)SymbolInfoInteger(_Symbol, SYMBOL_DIGITS);
   if(digits <= 0)
      digits = _Digits;

   double point = SymbolInfoDouble(_Symbol, SYMBOL_POINT);
   if(point <= 0.0)
      point = _Point;

   int stops_level_points = (int)SymbolInfoInteger(_Symbol, SYMBOL_TRADE_STOPS_LEVEL);
   int freeze_level_points = (int)SymbolInfoInteger(_Symbol, SYMBOL_TRADE_FREEZE_LEVEL);
   int required_points = MathMax(MathMax(stops_level_points, freeze_level_points), 1);
   double distance = (required_points + step_multiplier) * point;
   bool is_buy = IsBuyOrderType(orderType);
   double close_price = GetClosePriceBySide(orderType);

   if(close_price <= 0.0)
      close_price = price;

   if(use_sl)
     {
      if(is_buy)
         sl = MathMin(sl, close_price - distance);
      else
         sl = MathMax(sl, close_price + distance);
     }

   if(is_buy)
      tp = MathMax(tp, close_price + distance);
   else
      tp = MathMin(tp, close_price - distance);

   if(use_sl)
      sl = NormalizeDouble(sl, digits);
   tp = NormalizeDouble(tp, digits);
  }

//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool NormalizeStopsWithOrderCheck(ENUM_ORDER_TYPE orderType,
                                  double volume,
                                  double &price,
                                  double &sl,
                                  double &tp,
                                  bool use_sl,
                                  int max_attempts = 10)
  {
   MqlTradeRequest check_request;
   MqlTradeCheckResult check_result;
   ZeroMemory(check_request);
   ZeroMemory(check_result);

   check_request.action = TRADE_ACTION_DEAL;
   check_request.symbol = _Symbol;
   check_request.volume = volume;
   check_request.type = orderType;
   check_request.type_time = ORDER_TIME_GTC;
   check_request.type_filling = GetPreferredFillingMode();

   for(int attempt = 0; attempt < max_attempts; attempt++)
     {
      price = GetExecutionPriceBySide(orderType);
      if(price <= 0.0)
         continue;

      NormalizeStopsForBroker(orderType, price, sl, tp, use_sl);

      check_request.price = price;
      check_request.sl = (use_sl ? sl : 0.0);
      check_request.tp = tp;

      if(!OrderCheck(check_request, check_result))
        {
         PrintFormat("OrderCheck failed for %s. attempt=%d error=%d",
                     _Symbol, attempt + 1, GetLastError());
         continue;
        }

      if(check_result.retcode != TRADE_RETCODE_INVALID_STOPS)
         return true;

      WidenStopsByStep(orderType, price, sl, tp, use_sl, (attempt + 1) * 2);
      PrintFormat("OrderCheck invalid stops. attempt=%d side=%s price=%.*f sl=%.*f tp=%.*f",
                  attempt + 1,
                  (IsBuyOrderType(orderType) ? "BUY" : "SELL"),
                  _Digits, price,
                  _Digits, sl,
                  _Digits, tp);
     }

   return false;
  }

//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool SendEntryWithManagedStops(CTrade &trade,
                               double volume,
                               ENUM_ORDER_TYPE orderType,
                               double &price,
                               double &sl,
                               double &tp,
                               string comment,
                               bool use_sl = true)
  {
   if(volume <= 0.0)
      return false;

   bool is_buy = IsBuyOrderType(orderType);
   long execution_mode = SymbolInfoInteger(_Symbol, SYMBOL_TRADE_EXEMODE);
   bool must_send_without_stops = (execution_mode == SYMBOL_TRADE_EXECUTION_EXCHANGE);

   if(!NormalizeStopsWithOrderCheck(orderType, volume, price, sl, tp, use_sl))
      NormalizeStopsForBroker(orderType, price, sl, tp, use_sl);

   double send_sl = (must_send_without_stops ? 0.0 : (use_sl ? sl : 0.0));
   double send_tp = (must_send_without_stops ? 0.0 : tp);
   PrintFormat("Managed order send | Symbol=%s | Side=%s | Mode=%s | Price=%.*f | SL=%.*f | TP=%.*f",
               _Symbol,
               (is_buy ? "BUY" : "SELL"),
               (must_send_without_stops ? "OPEN_THEN_MODIFY" : "DIRECT_WITH_STOPS"),
               _Digits, price,
               _Digits, send_sl,
               _Digits, send_tp);
   bool trade_ok = (is_buy
                    ? trade.Buy(volume, _Symbol, price, send_sl, send_tp, comment)
                    : trade.Sell(volume, _Symbol, price, send_sl, send_tp, comment));

   if(!trade_ok && trade.ResultRetcode() == TRADE_RETCODE_INVALID_STOPS && !must_send_without_stops)
     {
      trade_ok = (is_buy
                  ? trade.Buy(volume, _Symbol, price, 0.0, 0.0, comment)
                  : trade.Sell(volume, _Symbol, price, 0.0, 0.0, comment));
      must_send_without_stops = trade_ok;
     }

   if(!trade_ok)
      return false;

   if(!must_send_without_stops)
      return true;

   bool position_selected = false;
   for(int select_attempt = 0; select_attempt < 20; select_attempt++)
     {
      if(PositionSelect(_Symbol))
        {
         position_selected = true;
         break;
        }
      Sleep(100);
     }

   if(!position_selected)
     {
      PrintFormat("Order opened for %s but position could not be selected for stop modification.", _Symbol);
      if(use_sl)
        {
         trade.PositionClose(_Symbol);
         return false;
        }
      return true;
     }

   double current_price = 0.0;
   double current_sl = sl;
   double current_tp = tp;
   double position_volume = PositionGetDouble(POSITION_VOLUME);
   if(position_volume <= 0.0)
      position_volume = volume;

   for(int attempt = 0; attempt < 20; attempt++)
     {
      if(!NormalizeStopsWithOrderCheck(orderType, position_volume, current_price, current_sl, current_tp, use_sl))
         NormalizeStopsForBroker(orderType, current_price, current_sl, current_tp, use_sl);

      if(trade.PositionModify(_Symbol, (use_sl ? current_sl : 0.0), current_tp))
        {
         sl = current_sl;
         tp = current_tp;
         return true;
        }

      if(trade.ResultRetcode() != TRADE_RETCODE_INVALID_STOPS)
         break;

      WidenStopsByStep(orderType, GetExecutionPriceBySide(orderType), current_sl, current_tp, use_sl, (attempt + 1) * 3);
      Sleep(100);
     }

   PrintFormat("Order opened for %s but failed to apply SL/TP after retries. Retcode=%d %s",
               _Symbol,
               trade.ResultRetcode(),
               trade.ResultRetcodeDescription());
   if(use_sl)
     {
      bool close_ok = trade.PositionClose(_Symbol);
      PrintFormat("Unprotected position fallback close for %s | Closed=%s | Retcode=%d %s",
                  _Symbol,
                  (close_ok ? "true" : "false"),
                  trade.ResultRetcode(),
                  trade.ResultRetcodeDescription());
      return false;
     }
   return true;
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
