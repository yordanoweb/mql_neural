//+------------------------------------------------------------------+
//|                                          SimpleONNX_EMA_Cross.mq5|
//|                        Single-Feature EMA Cross Predictor        |
//+------------------------------------------------------------------+
#property strict

#include <Trade\Trade.mqh>

// IMPORTANTE: Actualizar con el nombre de tu modelo generado
#resource "\\Files\\ndx100_m5_ema9_f14_cls.onnx" as uchar ExtModel[];

//--- INPUTS
input group "AI Config"
input string     InpModelFile  = "ndx100_m5_ema9_f14_cls.onnx";
input float      InpMinConf    = 0.55;      // Min confidence
input int        InpStartHour  = 0;
input int        InpEndHour    = 23;
input bool       InpReverse    = false;       // Invert signal

input group "EMA & ATR - Match training values"
input int        InpEMAPeriod  = 9;
input int        InpATRPeriod  = 6;

input group "Risk Management"
input double     InpLot        = 1.0;
input int        InpMagic      = 8812345688;
input double     InpProfitATR  = 1.5;       // TP in ATRs
input double     InpStopATR    = 1.0;       // SL in ATRs
input int        InpMaxHoldBars= 14;        // Hold bars (match with --future)

//--- GLOBALS
long     onnx_handle = INVALID_HANDLE;
CTrade   m_trade;
int      g_ema_handle, g_atr_handle;
float    g_confidence = 0;
string   g_prediction_str = "WAITING...";
bool     g_valid_time = false;

//+------------------------------------------------------------------+
int OnInit()
  {
   onnx_handle = OnnxCreateFromBuffer(ExtModel, ONNX_DEFAULT);
   if(onnx_handle == INVALID_HANDLE) return(INIT_FAILED);

   // SHAPE CRÍTICO: [batch=1, features=1] - UNA SOLA FEATURE
   long input_shape[] = {1, 1};
   if(!OnnxSetInputShape(onnx_handle, 0, input_shape)) return(INIT_FAILED);

   long out_label[] = {1};
   long out_probs[] = {1, 2};
   OnnxSetOutputShape(onnx_handle, 0, out_label);
   OnnxSetOutputShape(onnx_handle, 1, out_probs);

   g_ema_handle = iMA(_Symbol, _Period, InpEMAPeriod, 0, MODE_EMA, PRICE_CLOSE);
   g_atr_handle = iATR(_Symbol, _Period, InpATRPeriod);
   
   if(g_ema_handle == INVALID_HANDLE || g_atr_handle == INVALID_HANDLE) 
      return(INIT_FAILED);

   m_trade.SetExpertMagicNumber(InpMagic);
   EventSetTimer(1);
   
   Print("[INIT] EMA Cross AI loaded | EMA:", InpEMAPeriod, " | MaxHold:", InpMaxHoldBars);
   return(INIT_SUCCEEDED);
  }

void OnDeinit(const int reason)
  {
   if(onnx_handle != INVALID_HANDLE) OnnxRelease(onnx_handle);
   IndicatorRelease(g_ema_handle);
   IndicatorRelease(g_atr_handle);
   Comment(""); 
   EventKillTimer();
  }

void OnTimer()
  {
   Comment("\n=== EMA(", IntegerToString(InpEMAPeriod), ") CROSS AI ===", 
           "\nPred: ", g_prediction_str,
           "\nConf: ", DoubleToString(g_confidence*100,1), "% / ", DoubleToString(InpMinConf*100,1), "%",
           "\nTime: ", (g_valid_time ? "OPEN" : "CLOSED"));
  }

void OnTick()
  {
   // 1. Gestionar posición abierta (time-based exit)
   if(PositionSelect(_Symbol))
     {
      ManageExit();
      return;
     }

   // 2. Filtro horario
   MqlDateTime dt; TimeCurrent(dt);
   g_valid_time = (dt.hour >= InpStartHour && dt.hour <= InpEndHour);
   if(!g_valid_time) return;

   // 3. Datos de indicadores (necesitamos 2 velas para detectar cruce)
   double ema[2], atr[1], close[2];
   ArraySetAsSeries(ema, true); ArraySetAsSeries(atr, true); ArraySetAsSeries(close, true);
   
   if(CopyBuffer(g_ema_handle, 0, 0, 2, ema) != 2) return;
   if(CopyBuffer(g_atr_handle, 0, 0, 1, atr) != 1) return;
   if(CopyClose(_Symbol, _Period, 0, 2, close) != 2) return;
   
   if(ema[0] == 0 || atr[0] == 0) return;

   // 4. DETECTAR CRUCE EXACTO
   bool above_now = close[0] > ema[0];
   bool above_prev = close[1] > ema[1];
   bool cross_up = !above_prev && above_now;
   bool cross_down = above_prev && !above_now;
   
   if(!cross_up && !cross_down) return; // Sin cruce, no operar

   // 5. FEATURE ÚNICA: (close - ema) / atr
   float feature = (float)((close[0] - ema[0]) / atr[0]);
   float input_buffer[1] = {feature};

   // 6. INFERENCIA
   long label[1]; 
   float probs[2];
   if(!OnnxRun(onnx_handle, ONNX_NO_CONVERSION, input_buffer, label, probs)) return;

   long pred = label[0];  // 0=fallo, 1=éxito
   float conf = probs[pred];
   
   if(InpReverse) 
     { 
      pred = 1 - pred; 
      conf = 1.0f - conf; 
     }

   g_confidence = conf;
   g_prediction_str = (pred == 1) ? (cross_up ? "BUY" : "SELL") : "SKIP";

   // 7. FILTRO DE CONFIANZA
   if(pred != 1 || conf < InpMinConf) 
     {
      Print("[SKIP] Cross ", (cross_up?"UP":"DOWN"), " | Conf:", DoubleToString(conf*100,1), "%");
      return;
     }

   // 8. EJECUTAR
   double sl_dist = atr[0] * InpStopATR;
   double tp_dist = atr[0] * InpProfitATR;
   
   if(cross_up) // BUY
     {
      double ask = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
      if(m_trade.Buy(InpLot, _Symbol, ask, ask-sl_dist, ask+tp_dist, "EMA_CROSS_BUY"))
         Print("[BUY] Conf:", DoubleToString(conf*100,1), "% | Feature:", DoubleToString(feature,4));
     }
   else // SELL
     {
      double bid = SymbolInfoDouble(_Symbol, SYMBOL_BID);
      if(m_trade.Sell(InpLot, _Symbol, bid, bid+sl_dist, bid-tp_dist, "EMA_CROSS_SELL"))
         Print("[SELL] Conf:", DoubleToString(conf*100,1), "% | Feature:", DoubleToString(feature,4));
     }
  }

void ManageExit()
  {
   datetime open_time = (datetime)PositionGetInteger(POSITION_TIME);
   int bars_held = Bars(_Symbol, _Period, open_time, TimeCurrent());
   
   if(bars_held >= InpMaxHoldBars)
     {
      m_trade.PositionClose(_Symbol);
      Print("[EXIT] Time-based close after ", bars_held, " bars");
     }
  }
//+------------------------------------------------------------------+