//+------------------------------------------------------------------+
//|                              TrendFollowing_Up_Down_Neural.mq5   |
//|    Compatible con modelo ONNX triclase:                          |
//|    output_label → 0 = sin señal | 1 = BUY | 2 = SELL             |
//+------------------------------------------------------------------+
#property strict

#include <Trade\Trade.mqh>

#resource "\\Files\\ndx100_rates_h1_trend_up_down.onnx" as uchar ExtModel[];

//--- INPUTS
input group "AI Config"
input float      InpMinConf          = 0.52;
input int        InpStartHour        = 12;
input int        InpEndHour          = 18;
input group "Risk"
input double     InpLot              = 1;
input int        InpMagic            = 123456;
input int        InpATR              = 5;
input double     InpMultiplier       = 1.1;
input bool       InpUseProfitClose   = true;
input double     InpProfitPercentSL  = 0.30;

//--- GLOBAL VARIABLES
long     onnx_handle = INVALID_HANDLE;
CTrade   m_trade;
const int WINDOW_SIZE = 20;
const int FEATURES    = 6;  // body, range, rsi, adx, plus_di, minus_di
const int NUM_CLASSES = 3;  // 0=nada, 1=buy, 2=sell

//--- STATUS CACHE
static float  g_conf_buy  = 0;
static float  g_conf_sell = 0;
static long   g_prediction = -1;
static double g_atr = 0;

//+------------------------------------------------------------------+
void ShowStatus()
{
   MqlDateTime dt;
   TimeCurrent(dt);
   bool valid_time = (dt.hour >= InpStartHour && dt.hour < InpEndHour);

   string signal = "WAITING";
   if(g_prediction == 1) signal = "BUY";
   else if(g_prediction == 2) signal = "SELL";
   else if(g_prediction == 0) signal = "NO SIGNAL";

   string info = "\n\n\n";
   info += MQLInfoString(MQL_PROGRAM_NAME) + " | " + _Symbol + " | " + EnumToString(_Period);
   info += "\nMagic: " + IntegerToString(InpMagic) + " | Lot: " + DoubleToString(InpLot, 2);
   info += "\nATR(" + IntegerToString(InpATR) + "): " + DoubleToString(g_atr, _Digits) +
           " | Mult: " + DoubleToString(InpMultiplier, 1) +
           " | ProfitClose: " + (InpUseProfitClose ? "ON @" + DoubleToString(InpProfitPercentSL * 100, 0) + "%SL" : "OFF");
   info += "\nSpread: " + IntegerToString(SymbolInfoInteger(_Symbol, SYMBOL_SPREAD)) +
           " pts | MinConf: " + DoubleToString(InpMinConf * 100, 1) + "%";
   info += "\nSignal: " + signal +
           " | Conf BUY: "  + DoubleToString(g_conf_buy  * 100, 2) + "%" +
           " | Conf SELL: " + DoubleToString(g_conf_sell * 100, 2) + "%";
   info += "\nSchedule: " + StringFormat("%02d:00-%02d:00", InpStartHour, InpEndHour) +
           " > " + (valid_time ? "ACTIVE" : "RESTRICTED");

   if(PositionSelect(_Symbol))
   {
      long   pos_type   = PositionGetInteger(POSITION_TYPE);
      double pos_open   = PositionGetDouble(POSITION_PRICE_OPEN);
      double pos_profit = PositionGetDouble(POSITION_PROFIT);
      double pos_sl     = PositionGetDouble(POSITION_SL);
      double pos_tp     = PositionGetDouble(POSITION_TP);
      double pos_lots   = PositionGetDouble(POSITION_VOLUME);
      info += "\n--- Open Trade ---";
      info += "\nType: " + (pos_type == POSITION_TYPE_BUY ? "BUY" : "SELL") +
              " | Lots: " + DoubleToString(pos_lots, 2);
      info += "\nEntry: " + DoubleToString(pos_open, _Digits) +
              " | P/L: " + (pos_profit >= 0 ? "+" : "") + DoubleToString(pos_profit, 2);
      info += "\nSL: " + DoubleToString(pos_sl, _Digits) +
              " | TP: " + DoubleToString(pos_tp, _Digits);
   }
   else
      info += "\n--- No Open Trade ---";

   Comment(info);
}

//+------------------------------------------------------------------+
int OnInit()
{
   onnx_handle = OnnxCreateFromBuffer(ExtModel, ONNX_DEFAULT);
   if(onnx_handle == INVALID_HANDLE)
   {
      PrintFormat("ERROR: Cannot load model '%s'", ExtModel[0]);
      return(INIT_FAILED);
   }

   // Input: [1, WINDOW_SIZE * FEATURES]
   long input_shape[] = {1, WINDOW_SIZE * FEATURES};
   if(!OnnxSetInputShape(onnx_handle, 0, input_shape))
   {
      Print("ERROR: OnnxSetInputShape failed");
      return(INIT_FAILED);
   }

   // Output 0: label escalar → shape {1}
   long out_shape_label[] = {1};
   if(!OnnxSetOutputShape(onnx_handle, 0, out_shape_label))
   {
      Print("ERROR: OnnxSetOutputShape label failed");
      return(INIT_FAILED);
   }

   // Output 1: probabilidades de las 3 clases → shape {1, 3}
   long out_shape_probs[] = {1, NUM_CLASSES};
   if(!OnnxSetOutputShape(onnx_handle, 1, out_shape_probs))
   {
      Print("ERROR: OnnxSetOutputShape probs failed");
      return(INIT_FAILED);
   }

   m_trade.SetExpertMagicNumber(InpMagic);
   EventSetTimer(5);
   Print("Model loaded OK. Classes → 0:no signal | 1:BUY | 2:SELL");
   return(INIT_SUCCEEDED);
}

//+------------------------------------------------------------------+
void OnDeinit(const int reason)
{
   if(onnx_handle != INVALID_HANDLE) OnnxRelease(onnx_handle);
   Comment("");
}

void OnTimer() { ShowStatus(); }

//+------------------------------------------------------------------+
void OnTick()
{
   // 1. FILTRO HORARIO
   MqlDateTime dt;
   TimeCurrent(dt);
   bool valid_time = (dt.hour >= InpStartHour && dt.hour < InpEndHour);

   // 2. CONTROL DE BARRA
   static datetime last_bar = 0;
   datetime current_bar = iTime(_Symbol, _Period, 0);
   if(current_bar == last_bar) return;
   last_bar = current_bar;

   // 3. DATOS OHLC
   double close[], open[], high[], low[];
   ArraySetAsSeries(close, true); ArraySetAsSeries(open, true);
   ArraySetAsSeries(high,  true); ArraySetAsSeries(low,  true);

   if(CopyClose(_Symbol, _Period, 0, WINDOW_SIZE + 15, close) < WINDOW_SIZE + 15 ||
      CopyOpen (_Symbol, _Period, 0, WINDOW_SIZE,      open)  < WINDOW_SIZE ||
      CopyHigh (_Symbol, _Period, 0, WINDOW_SIZE,      high)  < WINDOW_SIZE ||
      CopyLow  (_Symbol, _Period, 0, WINDOW_SIZE,      low)   < WINDOW_SIZE) return;

   // 4. INDICADORES
   int rsi_handle = iRSI(_Symbol, _Period, 14, PRICE_CLOSE);
   double rsi_buffer[];
   ArraySetAsSeries(rsi_buffer, true);
   if(CopyBuffer(rsi_handle, 0, 0, WINDOW_SIZE, rsi_buffer) < WINDOW_SIZE) return;

   int atr_handle = iATR(_Symbol, _Period, InpATR);
   double atr_buffer[];
   ArraySetAsSeries(atr_buffer, true);
   if(CopyBuffer(atr_handle, 0, 0, 1, atr_buffer) < 1) return;
   g_atr = atr_buffer[0];

   int adx_handle = iADX(_Symbol, _Period, 14);
   double adx_buffer[], plus_di_buffer[], minus_di_buffer[];
   ArraySetAsSeries(adx_buffer,       true);
   ArraySetAsSeries(plus_di_buffer,   true);
   ArraySetAsSeries(minus_di_buffer,  true);
   if(CopyBuffer(adx_handle, 0, 0, WINDOW_SIZE, adx_buffer)      < WINDOW_SIZE) return;
   if(CopyBuffer(adx_handle, 1, 0, WINDOW_SIZE, plus_di_buffer)  < WINDOW_SIZE) return;
   if(CopyBuffer(adx_handle, 2, 0, WINDOW_SIZE, minus_di_buffer) < WINDOW_SIZE) return;

   // 5. BUFFER DE ENTRADA
   float input_buffer[];
   ArrayResize(input_buffer, WINDOW_SIZE * FEATURES);
   double pip_unit = _Point * (_Digits == 5 || _Digits == 3 ? 10 : 1);

   for(int i = 0; i < WINDOW_SIZE; i++)
   {
      int mql_idx = WINDOW_SIZE - 1 - i;  // más antiguo primero (igual que Python)
      input_buffer[i * FEATURES + 0] = (float)((close[mql_idx] - open[mql_idx]) / pip_unit); // body
      input_buffer[i * FEATURES + 1] = (float)((high[mql_idx]  - low[mql_idx])  / pip_unit); // range
      input_buffer[i * FEATURES + 2] = (float)(rsi_buffer[mql_idx]    / 100.0);              // rsi
      input_buffer[i * FEATURES + 3] = (float)(adx_buffer[mql_idx]);                         // adx
      input_buffer[i * FEATURES + 4] = (float)(plus_di_buffer[mql_idx]);                     // +DI
      input_buffer[i * FEATURES + 5] = (float)(minus_di_buffer[mql_idx]);                    // -DI
   }

   // 6. INFERENCIA TRICLASE
   // output_label : long[1]          → clase predicha (0, 1 o 2)
   // output_probs : float[3]         → P(clase0), P(clase1), P(clase2)
   long  output_label[];
   float output_probs[];
   ArrayResize(output_label, 1);
   ArrayResize(output_probs, NUM_CLASSES);  // ← 3 probabilidades, no 2

   if(!OnnxRun(onnx_handle, ONNX_NO_CONVERSION, input_buffer, output_label, output_probs)) return;

   long  prediction  = output_label[0];        // 0, 1 o 2
   float conf_no_sig = output_probs[0];        // P(sin señal)
   float conf_buy    = output_probs[1];        // P(BUY)
   float conf_sell   = output_probs[2];        // P(SELL)

   // Confianza de la clase ganadora
   float confidence = (prediction == 1) ? conf_buy :
                      (prediction == 2) ? conf_sell : conf_no_sig;

   g_prediction = prediction;
   g_conf_buy   = conf_buy;
   g_conf_sell  = conf_sell;

   // 7. EJECUCIÓN
   if(!PositionSelect(_Symbol) && valid_time && confidence >= InpMinConf)
   {
      double sl_dist = g_atr * InpMultiplier;
      double tp_dist = InpUseProfitClose ? (sl_dist * InpProfitPercentSL) : (sl_dist * 1.5);

      if(prediction == 1)   // BUY directo del modelo
      {
         double price = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
         m_trade.Buy(InpLot, _Symbol, price,
                     price - sl_dist, price + tp_dist,
                     MQLInfoString(MQL_PROGRAM_NAME));
      }
      else if(prediction == 2)   // SELL directo del modelo
      {
         double price = SymbolInfoDouble(_Symbol, SYMBOL_BID);
         m_trade.Sell(InpLot, _Symbol, price,
                      price + sl_dist, price - tp_dist,
                      MQLInfoString(MQL_PROGRAM_NAME));
      }
      // prediction == 0 → no hacer nada
   }
}