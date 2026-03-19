//+------------------------------------------------------------------+
//|                              TrendFollowing_Status_ADX.mq5       |
//+------------------------------------------------------------------+
#property strict
#include <Trade\Trade.mqh>

input group "IA y Modelo"
input string     InpModelName        = "modelo_selectivo_puntos.onnx";
input float      InpMinConf          = 0.55;
input int        WINDOW_SIZE         = 25; // Window usado en training

input group "Filtros de Estrategia"
input float      InpADXThresh        = 24.0; // Umbral visual ADX (solo panel)
input int        InpStartHour        = 13;
input int        InpEndHour          = 21;

input group "Stochastic"
input int        InpStochK           = 5;    // %K period
input int        InpStochD           = 3;    // %D period (smoothing)
input int        InpStochSlowing     = 3;    // Slowing
input double     InpStochOversold    = 30.0; // Oversold level (solo panel)
input double     InpStochOverbought  = 70.0; // Overbought level (solo panel)

input group "Riesgo"
input double     InpLot              = 1.0;
input int        InpMagic            = 888;
input double     InpStopPoints       = 50.0;  // SL en PUNTOS (SYMBOL_POINT * N, NO pips)
input double     InpTakePoints       = 100.0; // TP en PUNTOS (SYMBOL_POINT * N, NO pips)

// Variables Globales
long     onnx_handle = INVALID_HANDLE;
CTrade   m_trade;
const int FEATURES    = 7;  // feat_body, feat_range, feat_adx, feat_pdi, feat_mdi, feat_stoch_k, feat_stoch_d

// Variables para el Status
long     g_prediction = 0;
float    g_conf_buy   = 0;
float    g_conf_sell  = 0;
double   g_curr_adx   = 0;
double   g_curr_pdi   = 0;
double   g_curr_mdi   = 0;
double   g_stoch_k    = 0;
double   g_stoch_d    = 0;
bool     g_stoch_buy  = false;
bool     g_stoch_sell = false;

//+------------------------------------------------------------------+
int OnInit()
  {
   onnx_handle = OnnxCreate(InpModelName, ONNX_DEFAULT);
   if(onnx_handle == INVALID_HANDLE)
     {
      Print("Error OnnxCreate: ", GetLastError());
      return(INIT_FAILED);
     }

   long input_shape[] = {1, WINDOW_SIZE * FEATURES};
   PrintFormat("ONNX input esperado por EA: [1,%d] (WINDOW_SIZE=%d, FEATURES=%d)",
               WINDOW_SIZE * FEATURES, WINDOW_SIZE, FEATURES);
   if(!OnnxSetInputShape(onnx_handle, 0, input_shape))
      Print("Fallo inicial de Shape. Intentando autodefinicion...");

   long out_shape_label[] = {1};
   OnnxSetOutputShape(onnx_handle, 0, out_shape_label);
   long out_shape_probs[] = {1, 3};
   OnnxSetOutputShape(onnx_handle, 1, out_shape_probs);

   // Diagnostic: print point value so user can verify SL/TP math
   double pt     = SymbolInfoDouble(_Symbol, SYMBOL_POINT);
   int    digits = (int)SymbolInfoInteger(_Symbol, SYMBOL_DIGITS);
   PrintFormat("Simbolo: %s | SYMBOL_POINT=%.6f | Digits=%d", _Symbol, pt, digits);
   PrintFormat("SL=%.0f pts = %.6f precio | TP=%.0f pts = %.6f precio",
               InpStopPoints, InpStopPoints * pt,
               InpTakePoints, InpTakePoints * pt);

   m_trade.SetExpertMagicNumber(InpMagic);
   return(INIT_SUCCEEDED);
  }

//+------------------------------------------------------------------+
void OnDeinit(const int reason)
  {
   if(onnx_handle != INVALID_HANDLE)
      OnnxRelease(onnx_handle);
   EventKillTimer();
   Comment("");
  }

void OnTimer() { ShowStatus(); }

//+------------------------------------------------------------------+
void OnTick()
  {
   static datetime last_bar = 0;
   datetime current_bar = iTime(_Symbol, _Period, 0);

   // Handles de indicadores
   int adx_h   = iADX(_Symbol, _Period, 14);
   int stoch_h = iStochastic(_Symbol, _Period,
                              InpStochK, InpStochD, InpStochSlowing,
                              MODE_SMA, STO_LOWHIGH);

   // Actualizar globals de panel en cada tick
   double adx_v[1], pdi_v[1], mdi_v[1];
   if(CopyBuffer(adx_h, 0, 0, 1, adx_v) > 0) g_curr_adx = adx_v[0];
   if(CopyBuffer(adx_h, 1, 0, 1, pdi_v) > 0) g_curr_pdi = pdi_v[0];
   if(CopyBuffer(adx_h, 2, 0, 1, mdi_v) > 0) g_curr_mdi = mdi_v[0];

   double sk[1], sd[1];
   if(CopyBuffer(stoch_h, 0, 0, 1, sk) > 0) g_stoch_k = sk[0];
   if(CopyBuffer(stoch_h, 1, 0, 1, sd) > 0) g_stoch_d = sd[0];

   // Crossover informacional para el panel (no es filtro de entrada)
   double sk2[2], sd2[2];
   ArraySetAsSeries(sk2, true);
   ArraySetAsSeries(sd2, true);
   if(CopyBuffer(stoch_h, 0, 0, 2, sk2) == 2 && CopyBuffer(stoch_h, 1, 0, 2, sd2) == 2)
     {
      g_stoch_buy  = (sk2[1] <= sd2[1]) && (sk2[0] > sd2[0]) && (sk2[1] < InpStochOversold);
      g_stoch_sell = (sk2[1] >= sd2[1]) && (sk2[0] < sd2[0]) && (sk2[1] > InpStochOverbought);
     }

   if(current_bar == last_bar)
     {
      ShowStatus();
      return;
     }
   last_bar = current_bar;

   // --- INFERENCIA ---
   double close[], open[], high[], low[];
   double adx_b[], pdi_b[], mdi_b[];
   double stoch_k_b[], stoch_d_b[];

   ArraySetAsSeries(close,     true);
   ArraySetAsSeries(open,      true);
   ArraySetAsSeries(high,      true);
   ArraySetAsSeries(low,       true);
   ArraySetAsSeries(adx_b,     true);
   ArraySetAsSeries(pdi_b,     true);
   ArraySetAsSeries(mdi_b,     true);
   ArraySetAsSeries(stoch_k_b, true);
   ArraySetAsSeries(stoch_d_b, true);

   if(CopyClose(_Symbol, _Period, 0, WINDOW_SIZE, close)     < WINDOW_SIZE) return;
   if(CopyOpen (_Symbol, _Period, 0, WINDOW_SIZE, open)      < WINDOW_SIZE) return;
   if(CopyHigh (_Symbol, _Period, 0, WINDOW_SIZE, high)      < WINDOW_SIZE) return;
   if(CopyLow  (_Symbol, _Period, 0, WINDOW_SIZE, low)       < WINDOW_SIZE) return;
   if(CopyBuffer(adx_h,   0, 0, WINDOW_SIZE, adx_b)         < WINDOW_SIZE) return;
   if(CopyBuffer(adx_h,   1, 0, WINDOW_SIZE, pdi_b)         < WINDOW_SIZE) return;
   if(CopyBuffer(adx_h,   2, 0, WINDOW_SIZE, mdi_b)         < WINDOW_SIZE) return;
   if(CopyBuffer(stoch_h, 0, 0, WINDOW_SIZE, stoch_k_b)     < WINDOW_SIZE) return;
   if(CopyBuffer(stoch_h, 1, 0, WINDOW_SIZE, stoch_d_b)     < WINDOW_SIZE) return;

   // Construir input_buffer — orden idéntico al training:
   // body, range, adx, pdi, mdi, stoch_k, stoch_d
   // i=0 → barra más antigua; i=WINDOW_SIZE-1 → barra más reciente
   float input_buffer[];
   ArrayResize(input_buffer, WINDOW_SIZE * FEATURES);

   for(int i = 0; i < WINDOW_SIZE; i++)
     {
      int idx    = WINDOW_SIZE - 1 - i;
      int offset = i * FEATURES;
      input_buffer[offset + 0] = (float)(close[idx]     - open[idx]);   // feat_body
      input_buffer[offset + 1] = (float)(high[idx]      - low[idx]);    // feat_range
      input_buffer[offset + 2] = (float)(adx_b[idx]);                   // feat_adx
      input_buffer[offset + 3] = (float)(pdi_b[idx]);                   // feat_pdi
      input_buffer[offset + 4] = (float)(mdi_b[idx]);                   // feat_mdi
      input_buffer[offset + 5] = (float)(stoch_k_b[idx]);               // feat_stoch_k
      input_buffer[offset + 6] = (float)(stoch_d_b[idx]);               // feat_stoch_d
     }

   long  out_l[]; ArrayResize(out_l, 1);
   float out_p[]; ArrayResize(out_p, 3);

   if(OnnxRun(onnx_handle, ONNX_NO_CONVERSION, input_buffer, out_l, out_p))
     {
      g_prediction = out_l[0];
      g_conf_buy   = out_p[1];
      g_conf_sell  = out_p[2];

      MqlDateTime dt;
      TimeCurrent(dt);
      bool time_ok = (dt.hour >= InpStartHour && dt.hour < InpEndHour);
      float conf   = (g_prediction == 1) ? g_conf_buy
                   : (g_prediction == 2) ? g_conf_sell : 0;

      if(g_prediction > 0 && conf >= InpMinConf && time_ok && !PositionSelect(_Symbol))
        {
         // Convertir puntos → distancia de precio
         // SYMBOL_POINT es el incremento mínimo del instrumento
         // Ej: EURUSD 5 dígitos → SYMBOL_POINT = 0.00001
         //     US30   2 dígitos → SYMBOL_POINT = 0.01
         // Por lo tanto InpStopPoints=50 siempre significa 50 puntos del instrumento
         double pt      = SymbolInfoDouble(_Symbol, SYMBOL_POINT);
         int    digits  = (int)SymbolInfoInteger(_Symbol, SYMBOL_DIGITS);
         double sl_dist = InpStopPoints * pt;
         double tp_dist = InpTakePoints * pt;

         if(g_prediction == 1)
           {
            double p  = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
            double sl = NormalizeDouble(p - sl_dist, digits);
            double tp = NormalizeDouble(p + tp_dist, digits);
            m_trade.Buy(InpLot, _Symbol, p, sl, tp);
           }
         else
           {
            double p  = SymbolInfoDouble(_Symbol, SYMBOL_BID);
            double sl = NormalizeDouble(p + sl_dist, digits);
            double tp = NormalizeDouble(p - tp_dist, digits);
            m_trade.Sell(InpLot, _Symbol, p, sl, tp);
           }
        }
     }
   ShowStatus();
  }

//+------------------------------------------------------------------+
void ShowStatus()
  {
   MqlDateTime dt;
   TimeCurrent(dt);
   bool valid_time = (dt.hour >= InpStartHour && dt.hour < InpEndHour);
   string signal = (g_prediction == 1) ? "BUY" : (g_prediction == 2) ? "SELL" : "NO SIGNAL";

   string trend_status = (g_curr_adx >= InpADXThresh) ? "STRONG" : "WEAK/RANGE";

   string stoch_zone;
   if(g_stoch_k < InpStochOversold)       stoch_zone = "OVERSOLD";
   else if(g_stoch_k > InpStochOverbought) stoch_zone = "OVERBOUGHT";
   else                                    stoch_zone = "NEUTRAL";

   string stoch_cross = (g_stoch_k > g_stoch_d) ? "%K > %D ▲" : "%K < %D ▼";

   double pt = SymbolInfoDouble(_Symbol, SYMBOL_POINT);

   string info = "\n\n\n=== " + MQLInfoString(MQL_PROGRAM_NAME) + " ===\n";
   info += "Instrumento: " + _Symbol + " [" + EnumToString(_Period) + "]\n";
   info += "Horario: " + StringFormat("%02d:00-%02d:00", InpStartHour, InpEndHour) +
           " (" + (valid_time ? "ACTIVE" : "RESTRICTED") + ")\n";
   info += "------------------------------------------\n";
   info += "ADX ACTUAL: " + DoubleToString(g_curr_adx, 2) + " (" + trend_status + ")\n";
   info += "+DI: " + DoubleToString(g_curr_pdi, 2) + " | -DI: " + DoubleToString(g_curr_mdi, 2) + "\n";
   info += "------------------------------------------\n";
   info += "STOCHASTIC (" + (string)InpStochK + "," + (string)InpStochD + "," + (string)InpStochSlowing + ")\n";
   info += "%K: " + DoubleToString(g_stoch_k, 2) + " | %D: " + DoubleToString(g_stoch_d, 2) + "\n";
   info += "Zona: " + stoch_zone + " | " + stoch_cross + "\n";
   info += "Cross BUY: " + (g_stoch_buy  ? "YES" : "NO") +
           " | Cross SELL: " + (g_stoch_sell ? "YES" : "NO") + "\n";
   info += "------------------------------------------\n";
   info += "PREDICCION IA: " + signal + "\n";
   info += "Confianza BUY:  " + DoubleToString(g_conf_buy  * 100, 2) + "%\n";
   info += "Confianza SELL: " + DoubleToString(g_conf_sell * 100, 2) + "%\n";
   info += "Confianza Min:  " + DoubleToString(InpMinConf  * 100, 1) + "%\n";
   info += "------------------------------------------\n";
   info += "SL: " + DoubleToString(InpStopPoints, 0) + " pts (" +
           DoubleToString(InpStopPoints * pt, 5) + ")\n";
   info += "TP: " + DoubleToString(InpTakePoints, 0) + " pts (" +
           DoubleToString(InpTakePoints * pt, 5) + ")\n";
   info += "SYMBOL_POINT: " + DoubleToString(pt, 6) + "\n";
   info += "------------------------------------------\n";

   if(PositionSelect(_Symbol))
     {
      info += "TRADE: " + (PositionGetInteger(POSITION_TYPE) == POSITION_TYPE_BUY ? "BUY" : "SELL") +
              " | PnL: " + DoubleToString(PositionGetDouble(POSITION_PROFIT), 2) + " USD\n";
     }
   else
     {
      info += "Estado: Esperando senal...\n";
     }

   Comment(info);
  }
//+------------------------------------------------------------------+
