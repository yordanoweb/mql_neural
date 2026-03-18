//+------------------------------------------------------------------+
//|                              TrendFollowing_NoRSI_Puntos.mq5     |
//+------------------------------------------------------------------+
#property strict
#include <Trade\Trade.mqh>

input group "IA y Modelo"
input string     InpModelName        = "modelo_selectivo_puntos.onnx";
input float      InpMinConf          = 0.55; 
input int        WINDOW_SIZE         = 20;   

input group "Riesgo en PUNTOS NOMINALES"
input double     InpLot              = 1.0;
input int        InpMagic            = 888;
input double     InpStopPoints       = 30.0; 
input double     InpTakePoints       = 60.0; 

long     onnx_handle = INVALID_HANDLE;
CTrade   m_trade;
const int FEATURES    = 5; // body, range, adx, pdi, mdi

int OnInit() {
   onnx_handle = OnnxCreate(InpModelName, ONNX_DEFAULT);
   if(onnx_handle == INVALID_HANDLE) return(INIT_FAILED);

   long input_shape[] = {1, WINDOW_SIZE * FEATURES};
   OnnxSetInputShape(onnx_handle, 0, input_shape);
   
   long out_shape_label[] = {1};
   OnnxSetOutputShape(onnx_handle, 0, out_shape_label);
   
   long out_shape_probs[] = {1, 3};
   OnnxSetOutputShape(onnx_handle, 1, out_shape_probs);

   m_trade.SetExpertMagicNumber(InpMagic);
   return(INIT_SUCCEEDED);
}

void OnDeinit(const int reason) {
   if(onnx_handle != INVALID_HANDLE) OnnxRelease(onnx_handle);
}

void OnTick() {
   static datetime last_bar = 0;
   datetime current_bar = iTime(_Symbol, _Period, 0);
   if(current_bar == last_bar) return;
   last_bar = current_bar;

   double close[], open[], high[], low[], adx_buffer[], pdi_buffer[], mdi_buffer[];
   ArraySetAsSeries(close, true); ArraySetAsSeries(open, true);
   ArraySetAsSeries(high, true);  ArraySetAsSeries(low, true);
   ArraySetAsSeries(adx_buffer, true); ArraySetAsSeries(pdi_buffer, true); ArraySetAsSeries(mdi_buffer, true);

   if(CopyClose(_Symbol,_Period,0,WINDOW_SIZE,close) < WINDOW_SIZE) return;
   if(CopyOpen(_Symbol,_Period,0,WINDOW_SIZE,open) < WINDOW_SIZE) return;
   if(CopyHigh(_Symbol,_Period,0,WINDOW_SIZE,high) < WINDOW_SIZE) return;
   if(CopyLow(_Symbol,_Period,0,WINDOW_SIZE,low) < WINDOW_SIZE) return;

   int adx_h = iADX(_Symbol,_Period,14);
   CopyBuffer(adx_h, 0, 0, WINDOW_SIZE, adx_buffer);
   CopyBuffer(adx_h, 1, 0, WINDOW_SIZE, pdi_buffer);
   CopyBuffer(adx_h, 2, 0, WINDOW_SIZE, mdi_buffer);

   float input_buffer[];
   ArrayResize(input_buffer, WINDOW_SIZE * FEATURES);
   
   for(int i = 0; i < WINDOW_SIZE; i++) {
      int mql_idx = WINDOW_SIZE - 1 - i; 
      
      input_buffer[i * FEATURES + 0] = (float)(close[mql_idx] - open[mql_idx]); // Body
      input_buffer[i * FEATURES + 1] = (float)(high[mql_idx] - low[mql_idx]);   // Range
      input_buffer[i * FEATURES + 2] = (float)(adx_buffer[mql_idx]);           // ADX
      input_buffer[i * FEATURES + 3] = (float)(pdi_buffer[mql_idx]);           // +DI
      input_buffer[i * FEATURES + 4] = (float)(mdi_buffer[mql_idx]);           // -DI
   }

   long output_label[1];
   float output_probs[3];
   if(OnnxRun(onnx_handle, ONNX_NO_CONVERSION, input_buffer, output_label, output_probs)) {
      long prediction = output_label[0];
      float confidence = output_probs[prediction];

      if(prediction > 0 && confidence >= InpMinConf && !PositionSelect(_Symbol)) {
         if(prediction == 1) { // BUY
            double price = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
            m_trade.Buy(InpLot, _Symbol, price, price - InpStopPoints, price + InpTakePoints);
         }
         else if(prediction == 2) { // SELL
            double price = SymbolInfoDouble(_Symbol, SYMBOL_BID);
            m_trade.Sell(InpLot, _Symbol, price, price + InpStopPoints, price - InpTakePoints);
         }
      }
   }
}
