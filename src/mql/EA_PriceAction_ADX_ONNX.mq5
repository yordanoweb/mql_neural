//+------------------------------------------------------------------+
//|                              TrendFollowing_Status_ADX.mq5       |
//+------------------------------------------------------------------+
#property strict
#include <Trade\Trade.mqh>

input group "IA y Modelo"
input string     InpModelName        = "modelo_selectivo_puntos.onnx";
input float      InpMinConf          = 0.55; 
input int        WINDOW_SIZE         = 20;   

input group "Filtros de Estrategia"
input float      InpADXThresh        = 24.0; // Visualizar contra este umbral
input int        InpStartHour        = 13;   
input int        InpEndHour          = 21;   

input group "Riesgo en PUNTOS NOMINALES"
input double     InpLot              = 1.0;
input int        InpMagic            = 888;
input double     InpStopPoints       = 50.0; 
input double     InpTakePoints       = 100.0; 

// Variables Globales
long     onnx_handle = INVALID_HANDLE;
CTrade   m_trade;
const int FEATURES    = 5; 

// Variables para el Status
long     g_prediction = 0;
float    g_conf_buy   = 0;
float    g_conf_sell  = 0;
double   g_curr_adx   = 0;
double   g_curr_pdi   = 0;
double   g_curr_mdi   = 0;

//+------------------------------------------------------------------+
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
   EventSetTimer(5);
   return(INIT_SUCCEEDED);
}

void OnDeinit(const int reason) {
   if(onnx_handle != INVALID_HANDLE) OnnxRelease(onnx_handle);
   EventKillTimer();
   Comment("");
}

void OnTimer() { ShowStatus(); }

void OnTick() {
   static datetime last_bar = 0;
   datetime current_bar = iTime(_Symbol, _Period, 0);
   
   // Actualizamos indicadores en cada Tick para el Panel
   int adx_h = iADX(_Symbol, _Period, 14);
   double adx_v[1], pdi_v[1], mdi_v[1];
   if(CopyBuffer(adx_h, 0, 0, 1, adx_v) > 0) g_curr_adx = adx_v[0];
   if(CopyBuffer(adx_h, 1, 0, 1, pdi_v) > 0) g_curr_pdi = pdi_v[0];
   if(CopyBuffer(adx_h, 2, 0, 1, mdi_v) > 0) g_curr_mdi = mdi_v[0];

   if(current_bar == last_bar) {
      ShowStatus();
      return;
   }
   last_bar = current_bar;

   // --- LÓGICA DE INFERENCIA ---
   double close[], open[], high[], low[], adx_b[], pdi_b[], mdi_b[];
   ArraySetAsSeries(close,true); ArraySetAsSeries(open,true);
   ArraySetAsSeries(high,true); ArraySetAsSeries(low,true);
   ArraySetAsSeries(adx_b,true); ArraySetAsSeries(pdi_b,true); ArraySetAsSeries(mdi_b,true);

   if(CopyClose(_Symbol,_Period,0,WINDOW_SIZE,close) < WINDOW_SIZE) return;
   if(CopyOpen(_Symbol,_Period,0,WINDOW_SIZE,open) < WINDOW_SIZE) return;
   
   CopyBuffer(adx_h, 0, 0, WINDOW_SIZE, adx_b);
   CopyBuffer(adx_h, 1, 0, WINDOW_SIZE, pdi_b);
   CopyBuffer(adx_h, 2, 0, WINDOW_SIZE, mdi_b);
   CopyHigh(_Symbol,_Period,0,WINDOW_SIZE,high);
   CopyLow(_Symbol,_Period,0,WINDOW_SIZE,low);

   float input_buffer[];
   ArrayResize(input_buffer, WINDOW_SIZE * FEATURES);
   for(int i = 0; i < WINDOW_SIZE; i++) {
      int idx = WINDOW_SIZE - 1 - i; 
      input_buffer[i*5+0] = (float)(close[idx] - open[idx]); 
      input_buffer[i*5+1] = (float)(high[idx] - low[idx]);   
      input_buffer[i*5+2] = (float)(adx_b[idx]);           
      input_buffer[i*5+3] = (float)(pdi_b[idx]);           
      input_buffer[i*5+4] = (float)(mdi_b[idx]);           
   }

   long out_l[1]; float out_p[3];
   if(OnnxRun(onnx_handle, ONNX_NO_CONVERSION, input_buffer, out_l, out_p)) {
      g_prediction = out_l[0];
      g_conf_buy   = out_p[1];
      g_conf_sell  = out_p[2];

      MqlDateTime dt; TimeCurrent(dt);
      bool time_ok = (dt.hour >= InpStartHour && dt.hour < InpEndHour);
      float conf = (g_prediction == 1) ? g_conf_buy : (g_prediction == 2) ? g_conf_sell : 0;

      if(g_prediction > 0 && conf >= InpMinConf && time_ok && !PositionSelect(_Symbol)) {
         if(g_prediction == 1) {
            double p = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
            m_trade.Buy(InpLot,_Symbol,p,p-InpStopPoints,p+InpTakePoints);
         } else {
            double p = SymbolInfoDouble(_Symbol, SYMBOL_BID);
            m_trade.Sell(InpLot,_Symbol,p,p+InpStopPoints,p-InpTakePoints);
         }
      }
   }
   ShowStatus();
}

//+------------------------------------------------------------------+
void ShowStatus() {
   MqlDateTime dt; TimeCurrent(dt);
   bool valid_time = (dt.hour >= InpStartHour && dt.hour < InpEndHour);
   string signal = (g_prediction == 1) ? "BUY" : (g_prediction == 2) ? "SELL" : "NO SIGNAL";
   
   // Color visual simple para la tendencia
   string trend_status = (g_curr_adx >= InpADXThresh) ? "STRONG" : "WEAK/RANGE";

   string info = "\n\n\n=== " + MQLInfoString(MQL_PROGRAM_NAME) + " ===\n";
   info += "Instrumento: " + _Symbol + " [" + EnumToString(_Period) + "]\n";
   info += "Horario: " + StringFormat("%02d:00-%02d:00", InpStartHour, InpEndHour) + 
           " (" + (valid_time ? "ACTIVE" : "RESTRICTED") + ")\n";
   info += "------------------------------------------\n";
   info += "ADX ACTUAL: " + DoubleToString(g_curr_adx, 2) + " (" + trend_status + ")\n";
   info += "+DI: " + DoubleToString(g_curr_pdi, 2) + " | -DI: " + DoubleToString(g_curr_mdi, 2) + "\n";
   info += "Umbral Min: " + DoubleToString(InpADXThresh, 1) + "\n";
   info += "------------------------------------------\n";
   info += "PREDICCIÓN IA: " + signal + "\n";
   info += "Confianza BUY:  " + DoubleToString(g_conf_buy * 100, 2) + "%\n";
   info += "Confianza SELL: " + DoubleToString(g_conf_sell * 100, 2) + "%\n";
   info += "Confianza Mín:  " + DoubleToString(InpMinConf * 100, 1) + "%\n";
   info += "------------------------------------------\n";
   
   if(PositionSelect(_Symbol)) {
      info += "TRADE: " + (PositionGetInteger(POSITION_TYPE)==POSITION_TYPE_BUY?"BUY":"SELL") + 
              " | PnL: " + DoubleToString(PositionGetDouble(POSITION_PROFIT), 2) + " USD\n";
   } else {
      info += "Estado: Esperando señal...\n";
   }

   Comment(info);
}
