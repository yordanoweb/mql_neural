//+------------------------------------------------------------------+
//|                        EA_PriceAction_ADX_Stoch_ONNX.mq5        |
//+------------------------------------------------------------------+
#property strict
#include <Trade\Trade.mqh>

input group "AI Model"
input string     InpModelName        = "modelo_selectivo_puntos.onnx";
input float      InpMinConf          = 0.55;   // Minimum confidence threshold
input int        WINDOW_SIZE         = 25;     // Window size used in training

input group "Inference Frequency"
input int        InpInferSeconds     = 15;     // Run inference every N seconds (0 = new bar only)
input bool       InpOneTradePerBar   = true;   // Limit to 1 trade per bar even if signal repeats

input group "Strategy Filters"
input float      InpADXThresh        = 24.0;   // ADX threshold (display only)
input int        InpStartHour        = 0;     // Session start hour
input int        InpEndHour          = 24;     // Session end hour

input group "Stochastic"
input int        InpStochK           = 5;      // %K period
input int        InpStochD           = 3;      // %D smoothing period
input int        InpStochSlowing     = 3;      // Slowing period
input double     InpStochOversold    = 30.0;   // Oversold level (display only)
input double     InpStochOverbought  = 70.0;   // Overbought level (display only)

input group "Risk"
input double     InpLot              = 1.0;
input int        InpMagic            = 7780877;
input double     InpStopPoints       = 50.0;   // SL in POINTS (SYMBOL_POINT * N, NOT pips)
input double     InpTakePoints       = 100.0;  // TP in POINTS (SYMBOL_POINT * N, NOT pips)

// Global variables
long      onnx_handle       = INVALID_HANDLE;
CTrade    m_trade;
const int FEATURES          = 7;   // body, range, adx, pdi, mdi, stoch_k, stoch_d

// Inference timing control
datetime  g_last_infer      = 0;   // Timestamp of last inference run
datetime  g_last_traded_bar = 0;   // Bar on which the last trade was opened

// Panel variables
long     g_prediction  = 0;
float    g_conf_buy    = 0;
float    g_conf_sell   = 0;
double   g_curr_adx    = 0;
double   g_curr_pdi    = 0;
double   g_curr_mdi    = 0;
double   g_stoch_k     = 0;
double   g_stoch_d     = 0;
bool     g_stoch_buy   = false;
bool     g_stoch_sell  = false;
int      g_infer_count = 0;   // Inference counter (panel display)

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
   PrintFormat("ONNX input shape: [1,%d] (WINDOW_SIZE=%d, FEATURES=%d)",
               WINDOW_SIZE * FEATURES, WINDOW_SIZE, FEATURES);
   if(!OnnxSetInputShape(onnx_handle, 0, input_shape))
      Print("Initial shape definition failed. Attempting auto-definition...");

   long out_shape_label[] = {1};
   OnnxSetOutputShape(onnx_handle, 0, out_shape_label);
   long out_shape_probs[] = {1, 3};
   OnnxSetOutputShape(onnx_handle, 1, out_shape_probs);

   double pt     = SymbolInfoDouble(_Symbol, SYMBOL_POINT);
   int    digits = (int)SymbolInfoInteger(_Symbol, SYMBOL_DIGITS);
   PrintFormat("Symbol: %s | SYMBOL_POINT=%.6f | Digits=%d", _Symbol, pt, digits);
   PrintFormat("SL=%.0f pts = %.6f price | TP=%.0f pts = %.6f price",
               InpStopPoints, InpStopPoints * pt,
               InpTakePoints, InpTakePoints * pt);

   if(InpInferSeconds > 0)
     {
      EventSetTimer(InpInferSeconds);
      PrintFormat("Inference timer active: every %d seconds", InpInferSeconds);
     }
   else
      Print("Mode: inference on new bar only");

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

//+------------------------------------------------------------------+
// Updates indicator globals for the panel (runs on every tick)
//+------------------------------------------------------------------+
void UpdateIndicatorGlobals(int adx_h, int stoch_h)
  {
   double adx_v[1], pdi_v[1], mdi_v[1];
   if(CopyBuffer(adx_h, 0, 0, 1, adx_v) > 0) g_curr_adx = adx_v[0];
   if(CopyBuffer(adx_h, 1, 0, 1, pdi_v) > 0) g_curr_pdi = pdi_v[0];
   if(CopyBuffer(adx_h, 2, 0, 1, mdi_v) > 0) g_curr_mdi = mdi_v[0];

   double sk[1], sd[1];
   if(CopyBuffer(stoch_h, 0, 0, 1, sk) > 0) g_stoch_k = sk[0];
   if(CopyBuffer(stoch_h, 1, 0, 1, sd) > 0) g_stoch_d = sd[0];

   double sk2[2], sd2[2];
   ArraySetAsSeries(sk2, true);
   ArraySetAsSeries(sd2, true);
   if(CopyBuffer(stoch_h, 0, 0, 2, sk2) == 2 && CopyBuffer(stoch_h, 1, 0, 2, sd2) == 2)
     {
      g_stoch_buy  = (sk2[1] <= sd2[1]) && (sk2[0] > sd2[0]) && (sk2[1] < InpStochOversold);
      g_stoch_sell = (sk2[1] >= sd2[1]) && (sk2[0] < sd2[0]) && (sk2[1] > InpStochOverbought);
     }
  }

//+------------------------------------------------------------------+
// Inference core -- callable from OnTick or OnTimer
//+------------------------------------------------------------------+
void RunInference()
  {
   int adx_h   = iADX(_Symbol, _Period, 14);
   int stoch_h = iStochastic(_Symbol, _Period,
                              InpStochK, InpStochD, InpStochSlowing,
                              MODE_SMA, STO_LOWHIGH);

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

   if(CopyClose(_Symbol, _Period, 0, WINDOW_SIZE, close)    < WINDOW_SIZE) return;
   if(CopyOpen (_Symbol, _Period, 0, WINDOW_SIZE, open)     < WINDOW_SIZE) return;
   if(CopyHigh (_Symbol, _Period, 0, WINDOW_SIZE, high)     < WINDOW_SIZE) return;
   if(CopyLow  (_Symbol, _Period, 0, WINDOW_SIZE, low)      < WINDOW_SIZE) return;
   if(CopyBuffer(adx_h,   0, 0, WINDOW_SIZE, adx_b)        < WINDOW_SIZE) return;
   if(CopyBuffer(adx_h,   1, 0, WINDOW_SIZE, pdi_b)        < WINDOW_SIZE) return;
   if(CopyBuffer(adx_h,   2, 0, WINDOW_SIZE, mdi_b)        < WINDOW_SIZE) return;
   if(CopyBuffer(stoch_h, 0, 0, WINDOW_SIZE, stoch_k_b)    < WINDOW_SIZE) return;
   if(CopyBuffer(stoch_h, 1, 0, WINDOW_SIZE, stoch_d_b)    < WINDOW_SIZE) return;

   // Build input buffer -- exact same feature order as training:
   // body, range, adx, pdi, mdi, stoch_k, stoch_d
   // i=0 -> oldest bar; i=WINDOW_SIZE-1 -> most recent bar
   float input_buffer[];
   ArrayResize(input_buffer, WINDOW_SIZE * FEATURES);

   for(int i = 0; i < WINDOW_SIZE; i++)
     {
      int idx    = WINDOW_SIZE - 1 - i;
      int offset = i * FEATURES;
      input_buffer[offset + 0] = (float)(close[idx]     - open[idx]);
      input_buffer[offset + 1] = (float)(high[idx]      - low[idx]);
      input_buffer[offset + 2] = (float)(adx_b[idx]);
      input_buffer[offset + 3] = (float)(pdi_b[idx]);
      input_buffer[offset + 4] = (float)(mdi_b[idx]);
      input_buffer[offset + 5] = (float)(stoch_k_b[idx]);
      input_buffer[offset + 6] = (float)(stoch_d_b[idx]);
     }

   long  out_l[]; ArrayResize(out_l, 1);
   float out_p[]; ArrayResize(out_p, 3);

   if(!OnnxRun(onnx_handle, ONNX_NO_CONVERSION, input_buffer, out_l, out_p))
      return;

   g_prediction = out_l[0];
   g_conf_buy   = out_p[1];
   g_conf_sell  = out_p[2];
   g_infer_count++;
   g_last_infer = TimeCurrent();

   MqlDateTime dt;
   TimeCurrent(dt);
   bool time_ok = (dt.hour >= InpStartHour && dt.hour < InpEndHour);
   float conf   = (g_prediction == 1) ? g_conf_buy
                : (g_prediction == 2) ? g_conf_sell : 0;

   // Guard: if InpOneTradePerBar=true, skip entry if already traded this bar
   datetime current_bar = iTime(_Symbol, _Period, 0);
   bool bar_ok = (!InpOneTradePerBar || g_last_traded_bar != current_bar);

   if(g_prediction > 0 && conf >= InpMinConf && time_ok && bar_ok && !PositionSelect(_Symbol))
     {
      double pt      = SymbolInfoDouble(_Symbol, SYMBOL_POINT);
      int    digits  = (int)SymbolInfoInteger(_Symbol, SYMBOL_DIGITS);
      double sl_dist = InpStopPoints * pt;
      double tp_dist = InpTakePoints * pt;

      if(g_prediction == 1)
        {
         double p  = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
         double sl = NormalizeDouble(p - sl_dist, digits);
         double tp = NormalizeDouble(p + tp_dist, digits);
         if(m_trade.Buy(InpLot, _Symbol, p, sl, tp))
            g_last_traded_bar = current_bar;
        }
      else
        {
         double p  = SymbolInfoDouble(_Symbol, SYMBOL_BID);
         double sl = NormalizeDouble(p + sl_dist, digits);
         double tp = NormalizeDouble(p - tp_dist, digits);
         if(m_trade.Sell(InpLot, _Symbol, p, sl, tp))
            g_last_traded_bar = current_bar;
        }
     }
  }

//+------------------------------------------------------------------+
void OnTick()
  {
   int adx_h   = iADX(_Symbol, _Period, 14);
   int stoch_h = iStochastic(_Symbol, _Period,
                              InpStochK, InpStochD, InpStochSlowing,
                              MODE_SMA, STO_LOWHIGH);

   // Panel always updates on every tick
   UpdateIndicatorGlobals(adx_h, stoch_h);

   if(InpInferSeconds <= 0)
     {
      // Classic mode: inference on new bar only
      static datetime last_bar = 0;
      datetime current_bar = iTime(_Symbol, _Period, 0);
      if(current_bar != last_bar)
        {
         last_bar = current_bar;
         RunInference();
        }
     }
   // If InpInferSeconds > 0, inference is triggered by OnTimer only

   ShowStatus();
  }

//+------------------------------------------------------------------+
void OnTimer()
  {
   if(InpInferSeconds > 0)
      RunInference();

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
   if(g_stoch_k < InpStochOversold)        stoch_zone = "OVERSOLD";
   else if(g_stoch_k > InpStochOverbought) stoch_zone = "OVERBOUGHT";
   else                                     stoch_zone = "NEUTRAL";

   string stoch_cross = (g_stoch_k > g_stoch_d) ? "%K > %D ^" : "%K < %D v";
   double pt = SymbolInfoDouble(_Symbol, SYMBOL_POINT);

   string mode_str = (InpInferSeconds > 0)
                     ? "TIMER (" + (string)InpInferSeconds + "s) #" + (string)g_infer_count
                     : "NEW BAR ONLY";

   string info = "\n\n\n=== " + MQLInfoString(MQL_PROGRAM_NAME) + " ===\n";
   info += "Symbol : " + _Symbol + " [" + EnumToString(_Period) + "]\n";
   info += "Session: " + StringFormat("%02d:00-%02d:00", InpStartHour, InpEndHour) +
           " (" + (valid_time ? "ACTIVE" : "RESTRICTED") + ")\n";
   info += "Inference: " + mode_str + "\n";
   if(g_last_infer > 0)
      info += "Last run:  " + TimeToString(g_last_infer, TIME_SECONDS) + "\n";
   info += "------------------------------------------\n";
   info += "ADX: " + DoubleToString(g_curr_adx, 2) + " (" + trend_status + ")\n";
   info += "+DI: " + DoubleToString(g_curr_pdi, 2) + "  -DI: " + DoubleToString(g_curr_mdi, 2) + "\n";
   info += "------------------------------------------\n";
   info += "STOCHASTIC (" + (string)InpStochK + "," + (string)InpStochD + "," + (string)InpStochSlowing + ")\n";
   info += "%K: " + DoubleToString(g_stoch_k, 2) + "  %D: " + DoubleToString(g_stoch_d, 2) + "\n";
   info += "Zone : " + stoch_zone + "  " + stoch_cross + "\n";
   info += "Cross BUY: "  + (g_stoch_buy  ? "YES" : "NO") +
           "  Cross SELL: " + (g_stoch_sell ? "YES" : "NO") + "\n";
   info += "------------------------------------------\n";
   info += "AI PREDICTION: " + signal + "\n";
   info += "Conf BUY:  " + DoubleToString(g_conf_buy  * 100, 2) + "%\n";
   info += "Conf SELL: " + DoubleToString(g_conf_sell * 100, 2) + "%\n";
   info += "Conf Min:  " + DoubleToString(InpMinConf  * 100, 1) + "%\n";
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
              "  PnL: " + DoubleToString(PositionGetDouble(POSITION_PROFIT), 2) + " USD\n";
     }
   else
     {
      info += "Status: Waiting for signal...\n";
     }

   Comment(info);
  }
//+------------------------------------------------------------------+
