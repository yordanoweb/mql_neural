//+------------------------------------------------------------------+
//|                                            TrendFollowing_H1.mq5 |
//|                                    Improved version with 3-class |
//+------------------------------------------------------------------+
#property strict

#include <Trade\Trade.mqh>

// Resource with the ONNX model (must be compiled with the file)
#resource "\\Files\\ndx100_rates_h1_trend_up_down.onnx" as uchar ExtModel[]

//--- INPUTS
input group "AI Config"
input float      InpMinConf    = 0.52;         // Minimum confidence (0..1)
input int        InpStartHour  = 12;           // Trading start hour (server time)
input int        InpEndHour    = 18;           // Trading end hour (exclusive)
input group "Risk Management"
input double     InpRiskPercent = 1.0;          // Risk per trade (% of balance)
input int        InpMagic      = 123456;        // Magic number
input int        InpATR        = 14;            // ATR period
input double     InpATRMultSL   = 1.5;          // Stop Loss = ATR * multiplier
input double     InpRiskReward  = 2.0;           // Take Profit = SL * RiskReward
input bool       InpUseTrailing = true;          // Enable trailing stop
input double     InpTrailActivate = 1.0;         // Trailing activates when profit >= ATR * this
input double     InpTrailDistance = 2.0;         // Trailing stop distance (in ATR multiples)
input group "Trend Filter"
input double     InpMinADX      = 25.0;           // Minimum ADX to consider trending
input int        InpADXPeriod   = 14;             // ADX period

//--- GLOBAL VARIABLES
long     onnx_handle = INVALID_HANDLE;
CTrade   m_trade;
const int WINDOW_SIZE = 20;       // Must match training window
const int FEATURES    = 6;         // body, range, rsi, adx, plus_di, minus_di

//--- CACHES FOR COMMENT
static float  g_confidence = 0;
static long   g_prediction = -1;   // 0=no trade, 1=buy, 2=sell
static double g_atr        = 0;
static double g_adx        = 0;
static double g_plus_di    = 0;
static double g_minus_di   = 0;

//+------------------------------------------------------------------+
//| Display status on chart                                         |
//+------------------------------------------------------------------+
void ShowStatus()
{
   MqlDateTime dt;
   TimeCurrent(dt);
   bool valid_time = (dt.hour >= InpStartHour && dt.hour < InpEndHour);

   string info = "\n\n\n";
   info += MQLInfoString(MQL_PROGRAM_NAME) + " | " + _Symbol + " | " + EnumToString(_Period);
   info += "\nMagic: " + IntegerToString(InpMagic) + " | Risk: " + DoubleToString(InpRiskPercent,1) + "%";
   info += "\nATR(" + IntegerToString(InpATR) + "): " + DoubleToString(g_atr, _Digits) + 
           " | SL mult: " + DoubleToString(InpATRMultSL,1) + " | RR: " + DoubleToString(InpRiskReward,1);
   info += "\nSpread: " + IntegerToString(SymbolInfoInteger(_Symbol, SYMBOL_SPREAD)) + " pts | MinConf: " + DoubleToString(InpMinConf * 100, 1) + "%";
   info += "\nADX: " + DoubleToString(g_adx,1) + " | +DI: " + DoubleToString(g_plus_di,1) + " | -DI: " + DoubleToString(g_minus_di,1);
   
   string signal_str = "WAITING";
   if(g_prediction >= 0)
   {
      if(g_prediction == 1) signal_str = "BUY";
      else if(g_prediction == 2) signal_str = "SELL";
   }
   info += "\nConfidence: " + DoubleToString(g_confidence * 100, 2) + "% | Signal: " + signal_str;
   info += "\nSchedule: " + StringFormat("%02d:00-%02d:00", InpStartHour, InpEndHour) + " > " + (valid_time ? "ACTIVE" : "RESTRICTED");

   if(PositionSelect(_Symbol))
   {
      long   pos_type   = PositionGetInteger(POSITION_TYPE);
      double pos_open   = PositionGetDouble(POSITION_PRICE_OPEN);
      double pos_profit = PositionGetDouble(POSITION_PROFIT);
      double pos_sl     = PositionGetDouble(POSITION_SL);
      double pos_tp     = PositionGetDouble(POSITION_TP);
      double pos_lots   = PositionGetDouble(POSITION_VOLUME);
      info += "\n--- Open Trade ---";
      info += "\nType: " + (pos_type == POSITION_TYPE_BUY ? "BUY" : "SELL") + " | Lots: " + DoubleToString(pos_lots, 2);
      info += "\nEntry: " + DoubleToString(pos_open, _Digits) + " | P/L: " + (pos_profit >= 0 ? "+" : "") + DoubleToString(pos_profit, 2);
      info += "\nSL: " + DoubleToString(pos_sl, _Digits) + " | TP: " + DoubleToString(pos_tp, _Digits);
   }
   else
      info += "\n--- No Open Trade ---";

   Comment(info);
}

//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit()
{
   // Load ONNX model from resource
   onnx_handle = OnnxCreateFromBuffer(ExtModel, ONNX_DEFAULT);
   if(onnx_handle == INVALID_HANDLE)
   {
      Print("Failed to create ONNX handle.");
      return(INIT_FAILED);
   }

   // Set input shape: (1, WINDOW_SIZE * FEATURES)
   long input_shape[] = {1, WINDOW_SIZE * FEATURES};
   if(!OnnxSetInputShape(onnx_handle, 0, input_shape))
   {
      Print("Failed to set input shape.");
      return(INIT_FAILED);
   }

   // Output shapes: label (1,) and probabilities (1, 3)
   long out_shape_label[] = {1};
   OnnxSetOutputShape(onnx_handle, 0, out_shape_label);
   long out_shape_probs[] = {1, 3};
   OnnxSetOutputShape(onnx_handle, 1, out_shape_probs);

   m_trade.SetExpertMagicNumber(InpMagic);
   EventSetTimer(5);  // update status every 5 seconds
   return(INIT_SUCCEEDED);
}

//+------------------------------------------------------------------+
//| Expert deinitialization function                                 |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
{
   if(onnx_handle != INVALID_HANDLE)
      OnnxRelease(onnx_handle);
   EventKillTimer();
}

//+------------------------------------------------------------------+
//| Timer function                                                   |
//+------------------------------------------------------------------+
void OnTimer()
{
   ShowStatus();
}

//+------------------------------------------------------------------+
//| Expert tick function                                             |
//+------------------------------------------------------------------+
void OnTick()
{
   // 1. Time filter
   MqlDateTime dt;
   TimeCurrent(dt);
   bool valid_time = (dt.hour >= InpStartHour && dt.hour < InpEndHour);

   // 2. Bar control: run only once per bar
   static datetime last_bar = 0;
   datetime current_bar = iTime(_Symbol, _Period, 0);
   if(current_bar == last_bar)
      return;
   last_bar = current_bar;

   // 3. Get price data
   double close[], open[], high[], low[];
   ArraySetAsSeries(close, true); ArraySetAsSeries(open, true);
   ArraySetAsSeries(high, true);  ArraySetAsSeries(low, true);

   int needed = WINDOW_SIZE + 15; // extra for indicators
   if(CopyClose(_Symbol, _Period, 0, needed, close) < needed ||
      CopyOpen(_Symbol, _Period, 0, WINDOW_SIZE, open) < WINDOW_SIZE ||
      CopyHigh(_Symbol, _Period, 0, WINDOW_SIZE, high) < WINDOW_SIZE ||
      CopyLow(_Symbol, _Period, 0, WINDOW_SIZE, low) < WINDOW_SIZE)
   {
      Print("Failed to copy price data");
      return;
   }

   // 4. Indicators
   // RSI
   int rsi_handle = iRSI(_Symbol, _Period, 14, PRICE_CLOSE);
   if(rsi_handle == INVALID_HANDLE) return;
   double rsi_buffer[];
   ArraySetAsSeries(rsi_buffer, true);
   CopyBuffer(rsi_handle, 0, 0, WINDOW_SIZE, rsi_buffer);
   IndicatorRelease(rsi_handle);

   // ATR
   int atr_handle = iATR(_Symbol, _Period, InpATR);
   if(atr_handle == INVALID_HANDLE) return;
   double atr_buffer[];
   ArraySetAsSeries(atr_buffer, true);
   CopyBuffer(atr_handle, 0, 0, 1, atr_buffer);
   double current_atr = atr_buffer[0];
   g_atr = current_atr;
   IndicatorRelease(atr_handle);

   // ADX
   int adx_handle = iADX(_Symbol, _Period, InpADXPeriod);
   if(adx_handle == INVALID_HANDLE) return;
   double adx_buffer[], plus_di_buffer[], minus_di_buffer[];
   ArraySetAsSeries(adx_buffer, true);
   ArraySetAsSeries(plus_di_buffer, true);
   ArraySetAsSeries(minus_di_buffer, true);
   CopyBuffer(adx_handle, 0, 0, WINDOW_SIZE, adx_buffer);      // ADX
   CopyBuffer(adx_handle, 1, 0, WINDOW_SIZE, plus_di_buffer);  // +DI
   CopyBuffer(adx_handle, 2, 0, WINDOW_SIZE, minus_di_buffer); // -DI
   // Store latest values for status
   g_adx = adx_buffer[0];
   g_plus_di = plus_di_buffer[0];
   g_minus_di = minus_di_buffer[0];
   IndicatorRelease(adx_handle);

   // 5. Build input buffer (normalize by pip size)
   float input_buffer[];
   ArrayResize(input_buffer, WINDOW_SIZE * FEATURES);
   double pip_unit = _Point * (_Digits == 5 || _Digits == 3 ? 10 : 1);

   for(int i=0; i<WINDOW_SIZE; i++)
   {
      int idx = WINDOW_SIZE - 1 - i; // from oldest to newest?
      // In Python we used sequential order, but MQL expects same order.
      // We'll keep index 0 as most recent bar? Let's ensure consistency:
      // In Python, we used iloc[i-window:i] which takes oldest first.
      // So to match, we need to fill input_buffer with oldest bar first.
      // However, typical approach is to have input_buffer[0] = oldest bar, input_buffer[WINDOW_SIZE-1] = newest.
      // In our loop i=0 -> oldest (index WINDOW_SIZE-1), i=WINDOW_SIZE-1 -> newest (index 0). That's fine.
      int bar_index = WINDOW_SIZE - 1 - i; // i=0 -> bar_index=19 (oldest)
      // Alternative: use mql_idx = bar_index; that would get older bars first.
      // We'll follow what we did in original code: mql_idx = WINDOW_SIZE-1-i, which is oldest first.
      // In original code they used mql_idx = WINDOW_SIZE-1-i and then took close[mql_idx] etc.
      // That means input_buffer[0] corresponds to oldest bar. So we keep that.
      int mql_idx = WINDOW_SIZE - 1 - i; // oldest to newest
      input_buffer[i * FEATURES + 0] = (float)((close[mql_idx] - open[mql_idx]) / pip_unit);
      input_buffer[i * FEATURES + 1] = (float)((high[mql_idx] - low[mql_idx]) / pip_unit);
      input_buffer[i * FEATURES + 2] = (float)(rsi_buffer[mql_idx] / 100.0);
      input_buffer[i * FEATURES + 3] = (float)(adx_buffer[mql_idx]);
      input_buffer[i * FEATURES + 4] = (float)(plus_di_buffer[mql_idx]);
      input_buffer[i * FEATURES + 5] = (float)(minus_di_buffer[mql_idx]);
   }

   // 6. Run inference
   long output_label[];
   float output_probs[];
   ArrayResize(output_label, 1);
   ArrayResize(output_probs, 3); // 3 classes
   if(!OnnxRun(onnx_handle, ONNX_NO_CONVERSION, input_buffer, output_label, output_probs))
   {
      Print("ONNX run failed");
      return;
   }

   long prediction = output_label[0]; // 0, 1, or 2
   float confidence;
   if(prediction == 0)
      confidence = output_probs[0];
   else if(prediction == 1)
      confidence = output_probs[1];
   else
      confidence = output_probs[2];
   g_confidence = confidence;
   g_prediction = prediction;

   // 7. Trend filter in real time
   bool is_trending = (g_adx >= InpMinADX);
   bool trend_up = (g_plus_di > g_minus_di);
   bool trend_down = (g_minus_di > g_plus_di);

   // 8. Check if we already have a position
   if(PositionSelect(_Symbol))
   {
      // Manage trailing stop if enabled
      if(InpUseTrailing)
      {
         double pos_sl = PositionGetDouble(POSITION_SL);
         double pos_open = PositionGetDouble(POSITION_PRICE_OPEN);
         double pos_type = PositionGetInteger(POSITION_TYPE);
         double current_price = (pos_type == POSITION_TYPE_BUY) ? SymbolInfoDouble(_Symbol, SYMBOL_BID) : SymbolInfoDouble(_Symbol, SYMBOL_ASK);
         double new_sl = 0;
         double activation_dist = current_atr * InpTrailActivate;
         double trail_dist = current_atr * InpTrailDistance;

         if(pos_type == POSITION_TYPE_BUY)
         {
            if(current_price > pos_open + activation_dist)
            {
               new_sl = current_price - trail_dist;
               if(new_sl > pos_sl)  // only move SL forward
               {
                  m_trade.PositionModify(_Symbol, new_sl, PositionGetDouble(POSITION_TP));
               }
            }
         }
         else if(pos_type == POSITION_TYPE_SELL)
         {
            if(current_price < pos_open - activation_dist)
            {
               new_sl = current_price + trail_dist;
               if(new_sl < pos_sl || pos_sl == 0)
               {
                  m_trade.PositionModify(_Symbol, new_sl, PositionGetDouble(POSITION_TP));
               }
            }
         }
      }
   }
   else // No position
   {
      // Check time, confidence, and trend
      if(!valid_time || confidence < InpMinConf || !is_trending)
         return;

      // Determine trade direction based on prediction and trend alignment
      int trade_signal = -1; // -1 = none
      if(prediction == 1 && trend_up)
         trade_signal = 0; // buy
      else if(prediction == 2 && trend_down)
         trade_signal = 1; // sell

      if(trade_signal == -1)
         return; // no aligned signal

      // Calculate dynamic lot size based on risk %
      double balance = AccountInfoDouble(ACCOUNT_BALANCE);
      double risk_amount = balance * InpRiskPercent / 100.0;
      double sl_dist = current_atr * InpATRMultSL;
      double sl_pips = sl_dist / _Point;
      double tick_value = SymbolInfoDouble(_Symbol, SYMBOL_TRADE_TICK_VALUE);
      if(sl_pips <= 0 || tick_value <= 0) return;
      double lot = risk_amount / (sl_pips * tick_value);
      // Round to allowed step
      double step = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_STEP);
      lot = MathFloor(lot / step) * step;
      lot = MathMax(lot, SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MIN));
      lot = MathMin(lot, SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MAX));

      double tp_dist = sl_dist * InpRiskReward;
      double price, sl, tp;

      if(trade_signal == 0) // BUY
      {
         price = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
         sl = price - sl_dist;
         tp = price + tp_dist;
         m_trade.Buy(lot, _Symbol, price, sl, tp, "TrendFollowing_3class");
      }
      else // SELL
      {
         price = SymbolInfoDouble(_Symbol, SYMBOL_BID);
         sl = price + sl_dist;
         tp = price - tp_dist;
         m_trade.Sell(lot, _Symbol, price, sl, tp, "TrendFollowing_3class");
      }
   }
}
