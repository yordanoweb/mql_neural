//+------------------------------------------------------------------+
//|                                              SimpleONNX_EA.mq5   |
//+------------------------------------------------------------------+
#property strict

#include <Trade\Trade.mqh>

#resource "\\Files\\USTEC_M15_oc_hl_rsi_w10_f8_mina0.5_rsi8_atr4.onnx" as uchar ExtModel[]

//--- ENUMERATIONS
enum ENUM_LOGIC { LOGIC_NORMAL, LOGIC_MIRROR };

//--- INPUTS
input group "AI Config"
input string     InpModelFile  = "USTEC_M15_oc_hl_rsi_w10_f8_mina0.5_rsi8_atr4.onnx";
input int        InpWindow     = 10;
input ENUM_LOGIC InpLogic      = LOGIC_MIRROR; 
input float      InpMinConf    = 0.55;         
input int        InpStartHour  = 0;            
input int        InpEndHour    = 23;           
input group "Risk"
input double     InpLot        = 1;          
input int        InpMagic      = 8812345688;
input int        InpATR        = 6;
input double     InpMultiplier = 1.1;          

//--- GLOBAL VARIABLES
long     onnx_handle = INVALID_HANDLE;
CTrade   m_trade;
const int FEATURES    = 3; 

int OnInit()
{
   if(MQLInfoInteger(MQL_TESTER))
      onnx_handle = OnnxCreateFromBuffer(ExtModel, ONNX_DEFAULT);
   else
      onnx_handle = OnnxCreate(InpModelFile, ONNX_DEFAULT);

   if(onnx_handle == INVALID_HANDLE) return(INIT_FAILED);

   long input_shape[] = {1, InpWindow * FEATURES};
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
   datetime current_bar = iTime(_Symbol, _Period, 0);
   if(current_bar == last_bar) return;
   last_bar = current_bar;

   // 3. DATA
   double close[], open[], high[], low[];
   ArraySetAsSeries(close, true); ArraySetAsSeries(open, true);
   ArraySetAsSeries(high, true);  ArraySetAsSeries(low, true);

   if(CopyClose(_Symbol, _Period, 0, InpWindow + 15, close) < InpWindow + 15 ||
      CopyOpen(_Symbol, _Period, 0, InpWindow, open) < InpWindow) return;

   // 4. INDICATORS
   int rsi_handle = iRSI(_Symbol, _Period, 14, PRICE_CLOSE);
   double rsi_buffer[];
   ArraySetAsSeries(rsi_buffer, true);
   CopyBuffer(rsi_handle, 0, 0, InpWindow, rsi_buffer);

   int atr_handle = iATR(_Symbol, _Period, InpATR);
   double atr_buffer[];
   ArraySetAsSeries(atr_buffer, true);
   CopyBuffer(atr_handle, 0, 0, 1, atr_buffer);
   double current_atr = atr_buffer[0];

   // 5. INPUT BUFFER WITH NORMALIZATION BY _Digits
   float input_buffer[];
   ArrayResize(input_buffer, InpWindow * FEATURES);
   
   // _Digits is the correct variable. If it's 5 or 3 decimals, adjust to pips (x10).
   double pip_unit = _Point * (_Digits == 5 || _Digits == 3 ? 10 : 1);

   for(int i=0; i < InpWindow; i++)
   {
      int mql_idx = InpWindow - 1 - i;
      input_buffer[i * 3 + 0] = (float)((close[mql_idx] - open[mql_idx]) / pip_unit);
      input_buffer[i * 3 + 1] = (float)((iHigh(_Symbol, _Period, mql_idx) - iLow(_Symbol, _Period, mql_idx)) / pip_unit);
      input_buffer[i * 3 + 2] = (float)(rsi_buffer[mql_idx] / 100.0);
   }

   // 6. INFERENCE
   long output_label[]; float output_probs[];
   ArrayResize(output_label, 1); ArrayResize(output_probs, 2);
   if(!OnnxRun(onnx_handle, ONNX_NO_CONVERSION, input_buffer, output_label, output_probs)) return;

   long  prediction = output_label[0];
   float confidence  = (prediction == 1) ? output_probs[1] : output_probs[0];

   // 7. EXECUTION WITH TIME FILTER
   if(!PositionSelect(_Symbol) && valid_time && confidence >= InpMinConf)
   {
      double sl_dist = current_atr * InpMultiplier;
      double tp_dist = sl_dist * 1.5;

      if((InpLogic == LOGIC_MIRROR && prediction == 1) || (InpLogic == LOGIC_NORMAL && prediction == 0))
      {
         double price = SymbolInfoDouble(_Symbol, SYMBOL_BID);
         m_trade.Sell(InpLot, _Symbol, price, price + sl_dist, price - tp_dist, "AI " + IntegerToString(_Period));
      }
      else
      {
         double price = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
         m_trade.Buy(InpLot, _Symbol, price, price - sl_dist, price + tp_dist, "AI " + IntegerToString(_Period));
      }
   }
   
   Comment("AI | Confidence: ", DoubleToString(confidence*100, 2), "%",
           "\nSchedule: ", (valid_time ? "ACTIVE" : "RESTRICTED"));
}