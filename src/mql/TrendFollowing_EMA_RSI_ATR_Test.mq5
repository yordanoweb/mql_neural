//+------------------------------------------------------------------+
//|                                   TrendFollowing_EMA_RSI_ATR.mq5 |
//+------------------------------------------------------------------+
#property strict

#include <Trade\Trade.mqh>

#resource "\\Files\\ndx100_rates_h1_ema_rsi_atr.onnx" as uchar ExtModel[];

//--- ENUMERATIONS
enum ENUM_LOGIC { LOGIC_NORMAL, LOGIC_MIRROR };

//--- INPUTS
input group "AI Config"
input ENUM_LOGIC InpLogic      = LOGIC_MIRROR;
input float      InpMinConf    = 0.66;
input int        InpStartHour  = 12;
input int        InpEndHour    = 20;
input group "Risk"
input double     InpLot        = 1;
input int        InpMagic      = 100654321;
input int        InpATR        = 4;
input double     InpMultiplier = 0.3;
input bool       InpUseProfitClose   = true;
input double     InpProfitPercentSL  = 0.30;

//--- GLOBAL VARIABLES
long     onnx_handle = INVALID_HANDLE;
CTrade   m_trade;
const int WINDOW_SIZE = 20; // Ventana de 20 velas
const int FEATURES    = 5;  // body, range, ema, rsi, atr

//--- STATUS CACHES
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
   info += "\nLogic: " + EnumToString(InpLogic) + " | Magic: " + IntegerToString(InpMagic) + " | Lot: " + DoubleToString(InpLot, 2);
   info += "\nATR(" + IntegerToString(InpATR) + "): " + DoubleToString(g_atr, _Digits) + " | Mult: " + DoubleToString(InpMultiplier, 1);
   info += "\nSpread: " + IntegerToString(SymbolInfoInteger(_Symbol, SYMBOL_SPREAD)) + " pts | MinConf: " + DoubleToString(InpMinConf * 100, 1) + "%";

   string signal = "WAITING";
   if(g_prediction >= 0)
   {
      bool is_sell = (InpLogic == LOGIC_MIRROR && g_prediction == 1) || (InpLogic == LOGIC_NORMAL && g_prediction == 0);
      signal = is_sell ? "SELL" : "BUY";
   }
   info += "\nConfidence: " + DoubleToString(g_confidence * 100, 2) + "% | Signal: " + signal;
   info += "\nSchedule: " + StringFormat("%02d:00-%02d:00", InpStartHour, InpEndHour) + " > " + (valid_time ? "ACTIVE" : "RESTRICTED");

   Comment(info);
}

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
   EventSetTimer(5);
   return(INIT_SUCCEEDED);
}

void OnDeinit(const int reason) { if(onnx_handle != INVALID_HANDLE) OnnxRelease(onnx_handle); }
void OnTimer() { ShowStatus(); }

void OnTick()
{
   // 1. FILTRO DE HORARIO
   MqlDateTime dt;
   TimeCurrent(dt);
   bool valid_time = (dt.hour >= InpStartHour && dt.hour < InpEndHour);

   // 2. CONTROL DE BARRA
   static datetime last_bar = 0;
   datetime current_bar = iTime(_Symbol, _Period, 0);
   if(current_bar == last_bar) return;
   last_bar = current_bar;

   // 3. DATOS
   double close[], open[], high[], low[];
   ArraySetAsSeries(close, true); ArraySetAsSeries(open, true);
   ArraySetAsSeries(high, true);  ArraySetAsSeries(low, true);

   if(CopyClose(_Symbol, _Period, 0, WINDOW_SIZE + 15, close) < WINDOW_SIZE + 15 ||
      CopyOpen(_Symbol, _Period, 0, WINDOW_SIZE, open) < WINDOW_SIZE) return;

   // 4. INDICADORES
   int rsi_handle = iRSI(_Symbol, _Period, 7, PRICE_CLOSE);
   double rsi_buffer[];
   ArraySetAsSeries(rsi_buffer, true);
   CopyBuffer(rsi_handle, 0, 0, WINDOW_SIZE, rsi_buffer);

   int ema_handle = iMA(_Symbol, _Period, 9, 0, MODE_EMA, PRICE_CLOSE);
   double ema_buffer[];
   ArraySetAsSeries(ema_buffer, true);
   CopyBuffer(ema_handle, 0, 0, WINDOW_SIZE, ema_buffer);

   int atr_handle = iATR(_Symbol, _Period, InpATR);
   double atr_buffer[];
   ArraySetAsSeries(atr_buffer, true);
   CopyBuffer(atr_handle, 0, 0, 1, atr_buffer);
   double current_atr = atr_buffer[0];
   g_atr = current_atr;

   // 5. INPUT BUFFER
   float input_buffer[];
   ArrayResize(input_buffer, WINDOW_SIZE * FEATURES);
   double pip_unit = _Point * (_Digits == 5 || _Digits == 3 ? 10 : 1);

   for(int i=0; i < WINDOW_SIZE; i++)
   {
      int mql_idx = WINDOW_SIZE - 1 - i;
      input_buffer[i * FEATURES + 0] = (float)((close[mql_idx] - open[mql_idx]) / pip_unit); // body
      input_buffer[i * FEATURES + 1] = (float)((iHigh(_Symbol, _Period, mql_idx) - iLow(_Symbol, _Period, mql_idx)) / pip_unit); // range
      input_buffer[i * FEATURES + 2] = (float)(ema_buffer[mql_idx]); // ema
      input_buffer[i * FEATURES + 3] = (float)(rsi_buffer[mql_idx] / 100.0); // rsi normalizado
      input_buffer[i * FEATURES + 4] = (float)(current_atr); // atr actual
   }

   // 6. INFERENCIA
   long output_label[]; float output_probs[];
   ArrayResize(output_label, 1); ArrayResize(output_probs, 2);
   if(!OnnxRun(onnx_handle, ONNX_NO_CONVERSION, input_buffer, output_label, output_probs)) return;

   long  prediction = output_label[0];
   float confidence  = (prediction == 1) ? output_probs[1] : output_probs[0];
   g_confidence = confidence;
   g_prediction = prediction;

   // 7. EJECUCIÓN
   if(!PositionSelect(_Symbol) && valid_time && confidence >= InpMinConf)
   {
      double sl_dist = current_atr * InpMultiplier;
      double tp_dist = InpUseProfitClose ? (sl_dist * InpProfitPercentSL) : (sl_dist * 1.5);

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
   }
}

