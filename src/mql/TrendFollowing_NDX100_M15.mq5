//+------------------------------------------------------------------+
//|                                    TrendFollowing_NDX100_M15.mq5 |
//+------------------------------------------------------------------+
#property strict

#include <Trade\Trade.mqh>

#resource "\\Files\\ndx100_rates_m15_trend.onnx" as uchar ExtModel[]

//--- ENUMERATIONS
enum ENUM_LOGIC { LOGIC_NORMAL, LOGIC_MIRROR };

//--- INPUTS
input group "AI Config"
input ENUM_LOGIC InpLogic      = LOGIC_MIRROR;
input float      InpMinConf    = 0.52;
input int        InpStartHour  = 12;
input int        InpEndHour    = 18;
input int        InpCooldownCandles = 1;       // Cooldown candles before re-entry
input group "Risk"
input double     InpLot        = 1;
input int        InpMagic      = 123456;
input int        InpATR        = 5;
input double     InpMultiplier = 1.1;
input bool       InpCloseAtHalfTP = true;      // Close at half TP

//--- GLOBAL VARIABLES
long     onnx_handle = INVALID_HANDLE;
CTrade   m_trade;
const int WINDOW_SIZE = 20; // For M15 we use a window of 20
const int FEATURES    = 6;  // body, range, rsi, adx, plus_di, minus_di

int OnInit()
{
   onnx_handle = OnnxCreateFromBuffer(ExtModel, ONNX_DEFAULT);
   if(onnx_handle == INVALID_HANDLE) return(INIT_FAILED);

   long input_shape[] = {1, WINDOW_SIZE * FEATURES};
   if(!OnnxSetInputShape(onnx_handle, 0, input_shape)) return(INIT_FAILED);

   long out_shape_label[] = {1};
   OnnxSetOutputShape(onnx_handle, 0, out_shape_label);
   long out_shape_probs[] = {1, 2};
   OnnxSetOutputShape(onnx_handle, 1, out_shape_probs);

   m_trade.SetExpertMagicNumber(InpMagic);
   return(INIT_SUCCEEDED);
}

void OnDeinit(const int reason) { if(onnx_handle != INVALID_HANDLE) OnnxRelease(onnx_handle); }

void OnTick()
{
   // 1. CORRECT TIME FILTER
   MqlDateTime dt;
   TimeCurrent(dt);
   bool valid_time = (dt.hour >= InpStartHour && dt.hour < InpEndHour);

   // 2. BAR CONTROL
   static datetime last_bar = 0;
   static int last_entry_bar = -InpCooldownCandles - 1; // Track last entry bar index
   datetime current_bar = iTime(_Symbol, _Period, 0);
   if(current_bar == last_bar) return;
   last_bar = current_bar;

   int current_bar_index = 0; // Always 0 for the current bar
   int bars_since_entry = current_bar_index - last_entry_bar;

   // 3. DATA
   double close[], open[], high[], low[];
   ArraySetAsSeries(close, true); ArraySetAsSeries(open, true);
   ArraySetAsSeries(high, true);  ArraySetAsSeries(low, true);

   if(CopyClose(_Symbol, _Period, 0, WINDOW_SIZE + 15, close) < WINDOW_SIZE + 15 ||
      CopyOpen(_Symbol, _Period, 0, WINDOW_SIZE, open) < WINDOW_SIZE) return;

   // 4. INDICATORS
   int rsi_handle = iRSI(_Symbol, _Period, 14, PRICE_CLOSE);
   double rsi_buffer[];
   ArraySetAsSeries(rsi_buffer, true);
   CopyBuffer(rsi_handle, 0, 0, WINDOW_SIZE, rsi_buffer);

   int atr_handle = iATR(_Symbol, _Period, InpATR);
   double atr_buffer[];
   ArraySetAsSeries(atr_buffer, true);
   CopyBuffer(atr_handle, 0, 0, 1, atr_buffer);
   double current_atr = atr_buffer[0];

   int adx_handle = iADX(_Symbol, _Period, 14);
   double adx_buffer[], plus_di_buffer[], minus_di_buffer[];
   ArraySetAsSeries(adx_buffer, true);
   ArraySetAsSeries(plus_di_buffer, true);
   ArraySetAsSeries(minus_di_buffer, true);
   CopyBuffer(adx_handle, 0, 0, WINDOW_SIZE, adx_buffer);      // ADX
   CopyBuffer(adx_handle, 1, 0, WINDOW_SIZE, plus_di_buffer);  // +DI
   CopyBuffer(adx_handle, 2, 0, WINDOW_SIZE, minus_di_buffer); // -DI

   // 5. INPUT BUFFER WITH NORMALIZATION BY _Digits
   float input_buffer[];
   ArrayResize(input_buffer, WINDOW_SIZE * FEATURES);
   double pip_unit = _Point * (_Digits == 5 || _Digits == 3 ? 10 : 1);

   for(int i=0; i < WINDOW_SIZE; i++)
   {
      int mql_idx = WINDOW_SIZE - 1 - i;
      input_buffer[i * FEATURES + 0] = (float)((close[mql_idx] - open[mql_idx]) / pip_unit); // body
      input_buffer[i * FEATURES + 1] = (float)((iHigh(_Symbol, _Period, mql_idx) - iLow(_Symbol, _Period, mql_idx)) / pip_unit); // range
      input_buffer[i * FEATURES + 2] = (float)(rsi_buffer[mql_idx] / 100.0); // rsi
      input_buffer[i * FEATURES + 3] = (float)(adx_buffer[mql_idx]); // adx
      input_buffer[i * FEATURES + 4] = (float)(plus_di_buffer[mql_idx]); // +DI
      input_buffer[i * FEATURES + 5] = (float)(minus_di_buffer[mql_idx]); // -DI
   }

   // 6. INFERENCE
   long output_label[]; float output_probs[];
   ArrayResize(output_label, 1); ArrayResize(output_probs, 2);
   if(!OnnxRun(onnx_handle, ONNX_NO_CONVERSION, input_buffer, output_label, output_probs)) return;

   long  prediction = output_label[0];
   float confidence  = (prediction == 1) ? output_probs[1] : output_probs[0];

   // 7. CLOSE AT HALF TP (if enabled and position exists)
   if(InpCloseAtHalfTP && PositionSelect(_Symbol))
   {
      double entry_price = PositionGetDouble(POSITION_PRICE_OPEN);
      double sl_dist = current_atr * InpMultiplier;
      double tp_dist = sl_dist * 1.5;
      double half_tp = tp_dist / 2.0;
      bool should_close = false;

      if(PositionGetInteger(POSITION_TYPE) == POSITION_TYPE_BUY)
      {
         double ask = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
         if(ask >= entry_price + half_tp) should_close = true;
      }
      else if(PositionGetInteger(POSITION_TYPE) == POSITION_TYPE_SELL)
      {
         double bid = SymbolInfoDouble(_Symbol, SYMBOL_BID);
         if(bid <= entry_price - half_tp) should_close = true;
      }

      if(should_close)
      {
         ulong pos_ticket = PositionGetInteger(POSITION_TICKET);
         m_trade.PositionClose(pos_ticket);
         return;
      }
   }

   // 8. EXECUTION WITH TIME FILTER AND COOLDOWN
   if(!PositionSelect(_Symbol) && valid_time && confidence >= InpMinConf && bars_since_entry >= InpCooldownCandles)
   {
      double sl_dist = current_atr * InpMultiplier;
      double tp_dist = sl_dist * 1.5;

      if((InpLogic == LOGIC_MIRROR && prediction == 1) || (InpLogic == LOGIC_NORMAL && prediction == 0))
      {
         double price = SymbolInfoDouble(_Symbol, SYMBOL_BID);
         m_trade.Sell(InpLot, _Symbol, price, price + sl_dist, price - tp_dist, MQLInfoString(MQL_PROGRAM_NAME));
      }
      else
      {
         double price = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
         m_trade.Buy(InpLot, _Symbol, price, price - sl_dist, price + tp_dist, MQLInfoString(MQL_PROGRAM_NAME));
      }
      last_entry_bar = current_bar_index;
   }

   Comment("AI M15 Trend | Confidence: ", DoubleToString(confidence*100, 2), "%",
           "\nSchedule: ", (valid_time ? "ACTIVE" : "RESTRICTED"),
           "\nCooldown: ", (bars_since_entry < InpCooldownCandles ? "WAITING" : "READY"));
}
