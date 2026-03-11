//+------------------------------------------------------------------+
//|                                            StochasticADX_H1.mq5 |
//+------------------------------------------------------------------+
#property strict

#include <Trade\Trade.mqh>

//--- ENUMERATIONS
enum ENUM_LOGIC { LOGIC_NORMAL, LOGIC_MIRROR };

//--- INPUTS
input group "AI Config"
input ENUM_LOGIC InpLogic      = LOGIC_MIRROR;
input string     InpModelName = "stochastic_adx.onnx";
input float      InpMinConf    = 0.52;
input int        InpStartHour  = 12;
input int        InpEndHour    = 18;

input group "Risk"
input double     InpLot        = 1;
input int        InpMagic      = 123456;
input int        InpATR        = 5;
input double     InpMultiplier = 1.1;
input bool       InpUseProfitClose   = true;   // Close when profit reaches % of SL
input double     InpProfitPercentSL  = 0.30;   // % of SL as profit trigger

//--- INDICATOR PARAMETERS (matching Python training)
const int STOCH_K_PERIOD = 7;
const int STOCH_D_PERIOD = 3;
const int STOCH_SLOWING  = 3;
const int ADX_PERIOD     = 8;

//--- GLOBAL VARIABLES
long     onnx_handle = INVALID_HANDLE;
CTrade   m_trade;
const int WINDOW_SIZE = 20; // Same as Python training
const int FEATURES    = 7;  // body, range, stoch_k, stoch_d, adx, plus_di, minus_di

//--- STATUS CACHES FOR COMMENT
static float  g_confidence = 0;
static long   g_prediction = -1;
static double g_atr        = 0;

void ShowStatus()
{
   MqlDateTime dt;
   TimeCurrent(dt);
   bool valid_time = (dt.hour >= InpStartHour && dt.hour < InpEndHour);

   string info = "\n\n\n";
   info += MQLInfoString(MQL_PROGRAM_NAME) + " | " + _Symbol + " | " + EnumToString(_Period);
   info += "\nModel: " + InpModelName;
   info += "\nLogic: " + EnumToString(InpLogic) + " | Magic: " + IntegerToString(InpMagic) + " | Lot: " + DoubleToString(InpLot, 2);
   info += "\nATR(" + IntegerToString(InpATR) + "): " + DoubleToString(g_atr, _Digits) + " | Mult: " + DoubleToString(InpMultiplier, 1) +
           " | ProfitClose: " + (InpUseProfitClose ? "ON @" + DoubleToString(InpProfitPercentSL * 100, 0) + "%SL" : "OFF");
   info += "\nSpread: " + IntegerToString(SymbolInfoInteger(_Symbol, SYMBOL_SPREAD)) + " pts | MinConf: " + DoubleToString(InpMinConf * 100, 1) + "%";

   string signal = "WAITING";
   if(g_prediction >= 0)
   {
      long display_pred = ApplyLogic(g_prediction);
      if(display_pred == 1) signal = "BUY";
      else if(display_pred == 2) signal = "SELL";
      else signal = "HOLD";
   }
   info += "\nConfidence: " + DoubleToString(g_confidence * 100, 2) + "% | Signal: " + signal;
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

// Apply logic mirroring to prediction (0=hold, 1=buy, 2=sell)
long ApplyLogic(long prediction)
{
   if(InpLogic == LOGIC_MIRROR)
   {
      if(prediction == 1) return 2;
      if(prediction == 2) return 1;
   }
   return prediction;
}

int OnInit()
{
   onnx_handle = OnnxCreate(InpModelName, ONNX_DEFAULT);
   if(onnx_handle == INVALID_HANDLE) return(INIT_FAILED);

   long input_shape[] = {1, WINDOW_SIZE * FEATURES};
   if(!OnnxSetInputShape(onnx_handle, 0, input_shape)) return(INIT_FAILED);

   // Output shape: label (1) and probabilities (3) - assuming model outputs 3 classes
   long out_shape_label[] = {1};
   OnnxSetOutputShape(onnx_handle, 0, out_shape_label);
   long out_shape_probs[] = {1, 3};
   OnnxSetOutputShape(onnx_handle, 1, out_shape_probs);

   m_trade.SetExpertMagicNumber(InpMagic);
   EventSetTimer(5);
   return(INIT_SUCCEEDED);
}

void OnDeinit(const int reason) { if(onnx_handle != INVALID_HANDLE) OnnxRelease(onnx_handle); }
void OnTimer()
{
   ShowStatus();
}

void OnTick()
{
   // 1. CORRECT TIME FILTER
   MqlDateTime dt;
   TimeCurrent(dt);
   bool valid_time = (dt.hour >= InpStartHour && dt.hour < InpEndHour);

   // 2. BAR CONTROL
   static datetime last_bar = 0;
   datetime current_bar = iTime(_Symbol, _Period, 0);
   if(current_bar == last_bar) return;
   last_bar = current_bar;

   // 3. DATA
   double close[], open[], high[], low[];
   ArraySetAsSeries(close, true); ArraySetAsSeries(open, true);
   ArraySetAsSeries(high, true);  ArraySetAsSeries(low, true);

   if(CopyClose(_Symbol, _Period, 0, WINDOW_SIZE + 15, close) < WINDOW_SIZE + 15 ||
      CopyOpen(_Symbol, _Period, 0, WINDOW_SIZE, open) < WINDOW_SIZE) return;

   // 4. INDICATORS
   // ATR for stop loss
   int atr_handle = iATR(_Symbol, _Period, InpATR);
   double atr_buffer[];
   ArraySetAsSeries(atr_buffer, true);
   CopyBuffer(atr_handle, 0, 0, 1, atr_buffer);
   double current_atr = atr_buffer[0];
   g_atr = current_atr;

   // Stochastic (slow)
   int stoch_handle = iStochastic(_Symbol, _Period, STOCH_K_PERIOD, STOCH_D_PERIOD, STOCH_SLOWING, MODE_SMA, STO_LOWHIGH);
   double stoch_k_buffer[], stoch_d_buffer[];
   ArraySetAsSeries(stoch_k_buffer, true);
   ArraySetAsSeries(stoch_d_buffer, true);
   CopyBuffer(stoch_handle, 0, 0, WINDOW_SIZE, stoch_k_buffer); // %K line (slow)
   CopyBuffer(stoch_handle, 1, 0, WINDOW_SIZE, stoch_d_buffer); // %D line (slow)

   // ADX
   int adx_handle = iADX(_Symbol, _Period, ADX_PERIOD);
   double adx_buffer[], plus_di_buffer[], minus_di_buffer[];
   ArraySetAsSeries(adx_buffer, true);
   ArraySetAsSeries(plus_di_buffer, true);
   ArraySetAsSeries(minus_di_buffer, true);
   CopyBuffer(adx_handle, 0, 0, WINDOW_SIZE, adx_buffer);      // ADX
   CopyBuffer(adx_handle, 1, 0, WINDOW_SIZE, plus_di_buffer);  // +DI
   CopyBuffer(adx_handle, 2, 0, WINDOW_SIZE, minus_di_buffer); // -DI

   // 5. INPUT BUFFER WITH NORMALIZATION
   float input_buffer[];
   ArrayResize(input_buffer, WINDOW_SIZE * FEATURES);
   double pip_unit = _Point * (_Digits == 5 || _Digits == 3 ? 10 : 1);

   for(int i=0; i < WINDOW_SIZE; i++)
   {
      int mql_idx = WINDOW_SIZE - 1 - i;
      input_buffer[i * FEATURES + 0] = (float)((close[mql_idx] - open[mql_idx]) / pip_unit); // body
      input_buffer[i * FEATURES + 1] = (float)((iHigh(_Symbol, _Period, mql_idx) - iLow(_Symbol, _Period, mql_idx)) / pip_unit); // range
      input_buffer[i * FEATURES + 2] = (float)(stoch_k_buffer[mql_idx] / 100.0); // stoch_k normalized to 0-1
      input_buffer[i * FEATURES + 3] = (float)(stoch_d_buffer[mql_idx] / 100.0); // stoch_d normalized to 0-1
      input_buffer[i * FEATURES + 4] = (float)(adx_buffer[mql_idx]); // adx
      input_buffer[i * FEATURES + 5] = (float)(plus_di_buffer[mql_idx]); // +DI
      input_buffer[i * FEATURES + 6] = (float)(minus_di_buffer[mql_idx]); // -DI
   }

   // 6. INFERENCE
   long output_label[]; float output_probs[];
   ArrayResize(output_label, 1); ArrayResize(output_probs, 3);
   if(!OnnxRun(onnx_handle, ONNX_NO_CONVERSION, input_buffer, output_label, output_probs)) return;

   long  prediction = output_label[0];
   float confidence  = output_probs[prediction]; // probability of predicted class
   g_confidence = confidence;
   g_prediction = prediction;

   // Apply logic mirroring for trade decision
   long trade_prediction = ApplyLogic(prediction);

   // 7. EXECUTION WITH TIME FILTER
   if(!PositionSelect(_Symbol) && valid_time && confidence >= InpMinConf)
   {
      double sl_dist = current_atr * InpMultiplier;
      double tp_dist = InpUseProfitClose ? (sl_dist * InpProfitPercentSL) : (sl_dist * 1.5);

      if(trade_prediction == 1) // BUY
      {
         double price = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
         m_trade.Buy(InpLot, _Symbol, price, price - sl_dist, price + tp_dist, MQLInfoString(MQL_PROGRAM_NAME));
      }
      else if(trade_prediction == 2) // SELL
      {
         double price = SymbolInfoDouble(_Symbol, SYMBOL_BID);
         m_trade.Sell(InpLot, _Symbol, price, price + sl_dist, price - tp_dist, MQLInfoString(MQL_PROGRAM_NAME));
      }
      // trade_prediction == 0 (HOLD) does nothing
   }
}