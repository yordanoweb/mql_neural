//+------------------------------------------------------------------+
//|                                              SimpleONNX_EA.mq5   |
//+------------------------------------------------------------------+
#property strict

#include <Trade\Trade.mqh>

#resource "\\Files\\US100.cash_M15_20220311_20251230.onnx" as uchar ExtModel[];

//--- ENUMERATIONS
enum ENUM_LOGIC { LOGIC_NORMAL, LOGIC_MIRROR };

//--- INPUTS
input group "AI Configuration"
input string     InpModelFile  = "US100.cash_M15_20220311_20251230.onnx";  // Dynamic model filename
input ENUM_LOGIC InpLogic      = LOGIC_MIRROR; 
input float      InpMinConf    = 0.62;         
input int        InpStartHour  = 9;            
input int        InpEndHour    = 18;           
input group "Risk Management"
input double     InpLot        = 1;          
input int        InpMagic      = 123456;       
input int        InpATR        = 6;           
input double     InpMultiplier = 1.5;          
input group "Timer Settings"
input int        InpTimerSeconds = 60;  // Timer interval in seconds          

//--- GLOBAL VARIABLES
long     onnx_handle = INVALID_HANDLE;
CTrade   m_trade;
const int WINDOW_SIZE = 20; // For M15 we use window of 20
const int FEATURES    = 3;
long     g_prediction = 0;  // Last inference prediction
float    g_confidence = 0.0;  // Last inference confidence
bool     g_valid_time = false;  // Last time filter result 

int OnInit()
{
   // Load ONNX model directly from file
   onnx_handle = OnnxCreateFromBuffer(ExtModel, ONNX_DEFAULT);
   
   if(onnx_handle == INVALID_HANDLE)
   {
      Print("ERROR: Failed to load ONNX model: ", InpModelFile);
      Print("Error Code: ", GetLastError());
      Print("Make sure the file is in: C:\\Program Files\\MetaTrader 5\\MQL5\\Files\\");
      return(INIT_FAILED);
   }

   long input_shape[] = {1, 60}; // 20 candles * 3 features
   if(!OnnxSetInputShape(onnx_handle, 0, input_shape)) return(INIT_FAILED);

   long out_shape_label[] = {1};
   OnnxSetOutputShape(onnx_handle, 0, out_shape_label);
   long out_shape_probs[] = {1, 2};
   OnnxSetOutputShape(onnx_handle, 1, out_shape_probs);

   m_trade.SetExpertMagicNumber(InpMagic);
   EventSetTimer(InpTimerSeconds);  // Set timer to configured interval
   return(INIT_SUCCEEDED);
}

void OnDeinit(const int reason) 
{ 
   EventKillTimer();  // Stop timer
   if(onnx_handle != INVALID_HANDLE) OnnxRelease(onnx_handle); 
}

void OnTick()
{
   // 1. CORRECT TIME FILTER
   MqlDateTime dt;
   TimeCurrent(dt); 
   g_valid_time = (dt.hour >= InpStartHour && dt.hour < InpEndHour);

   // 2. CANDLE CONTROL
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
   int rsi_handle = iRSI(_Symbol, _Period, 14, PRICE_CLOSE);
   double rsi_buffer[];
   ArraySetAsSeries(rsi_buffer, true);
   CopyBuffer(rsi_handle, 0, 0, WINDOW_SIZE, rsi_buffer);

   int atr_handle = iATR(_Symbol, _Period, InpATR);
   double atr_buffer[];
   ArraySetAsSeries(atr_buffer, true);
   CopyBuffer(atr_handle, 0, 0, 1, atr_buffer);
   double current_atr = atr_buffer[0];

   // 5. INPUT BUFFER WITH NORMALIZATION BY _Digits
   float input_buffer[];
   ArrayResize(input_buffer, WINDOW_SIZE * FEATURES);
   
   // _Digits is the correct variable. If 5 or 3 decimals, we adjust to pips (x10).
   double pip_unit = _Point * (_Digits == 5 || _Digits == 3 ? 10 : 1);

   for(int i=0; i < WINDOW_SIZE; i++)
   {
      int mql_idx = WINDOW_SIZE - 1 - i;
      input_buffer[i * 3 + 0] = (float)((close[mql_idx] - open[mql_idx]) / pip_unit);
      input_buffer[i * 3 + 1] = (float)((iHigh(_Symbol, _Period, mql_idx) - iLow(_Symbol, _Period, mql_idx)) / pip_unit);
      input_buffer[i * 3 + 2] = (float)(rsi_buffer[mql_idx] / 100.0);
   }
   
   // Store input buffer for OnTimer use
   static float s_input_buffer[];
   ArrayResize(s_input_buffer, ArraySize(input_buffer));
   ArrayCopy(s_input_buffer, input_buffer);
   
   // Store data for OnTimer use
   static double s_current_atr = 0;
   s_current_atr = current_atr;

   // 7. EXECUTION WITH TIME FILTER (using global inference results from OnTimer)
   if(!PositionSelect(_Symbol) && g_valid_time && g_confidence >= InpMinConf)
   {
      double sl_dist = s_current_atr * InpMultiplier;
      double tp_dist = sl_dist * 1.5;

      if((InpLogic == LOGIC_MIRROR && g_prediction == 1) || (InpLogic == LOGIC_NORMAL && g_prediction == 0))
      {
         double price = SymbolInfoDouble(_Symbol, SYMBOL_BID);
         m_trade.Sell(InpLot, _Symbol, price, price + sl_dist, price - tp_dist, "AI M15");
      }
      else
      {
         double price = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
         m_trade.Buy(InpLot, _Symbol, price, price - sl_dist, price + tp_dist, "AI M15");
      }
   }
}

string GetTimeframeString(ENUM_TIMEFRAMES tf)
{
   switch(tf)
   {
      case PERIOD_M1: return "M1";
      case PERIOD_M5: return "M5";
      case PERIOD_M15: return "M15";
      case PERIOD_M30: return "M30";
      case PERIOD_H1: return "H1";
      case PERIOD_H4: return "H4";
      case PERIOD_D1: return "D1";
      default: return "Unknown TF";
   }
}

void OnTimer()
{
   PerformInference();
   
   // Update display
   Comment("\n\n\nAI " + GetTimeframeString(_Period) + " | Confidence: ", DoubleToString(g_confidence*100, 2), "%",
           "\nTime: ", (g_valid_time ? "ACTIVE" : "RESTRICTED"));
}

void PerformInference()
{
   // 6. INFERENCE
   static float s_input_buffer[];
   
   long output_label[]; float output_probs[];
   ArrayResize(output_label, 1); ArrayResize(output_probs, 2);
   if(!OnnxRun(onnx_handle, ONNX_NO_CONVERSION, s_input_buffer, output_label, output_probs)) return;

   g_prediction = output_label[0];
   g_confidence = (g_prediction == 1) ? output_probs[1] : output_probs[0];
}