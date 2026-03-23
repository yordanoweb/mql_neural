//+------------------------------------------------------------------+
//|                        EA_SGRADT50_ONNX.mq5                      |
//|                        SGRADT 5.0 - 5 Features Version           |
//+------------------------------------------------------------------+
#property strict
#include <Trade\Trade.mqh>

input group "=== AI Model Configuration ==="
input string     InpModelName        = "EUR_USD_H1_SGRADT50.onnx";  // ONNX Model filename
input string     InpMetaFile         = "EUR_USD_H1_SGRADT50.meta.json"; // Metadata file (optional)
input float      InpMinConf          = 0.55;   // Minimum confidence threshold
input int        InpWindowSize       = 20;     // Window size (from training)
input int        InpFeaturesPerBar   = 5;      // Features per bar (FIXED: 5)

input group "=== Inference Timing ==="
input int        InpInferSeconds     = 15;     // Run inference every N seconds (0 = new bar only)
input bool       InpOneTradePerBar   = true;   // Limit to 1 trade per bar

input group "=== Trading Session ==="
input int        InpStartHour        = 0;      // Session start hour (0-23)
input int        InpEndHour          = 24;     // Session end hour (0-24)

input group "=== Indicator Parameters (SGRADT 5.0 Defaults) ==="
input int        InpStochK           = 7;      // Stochastic %K period
input int        InpStochD           = 3;      // Stochastic %D smoothing
input int        InpStochSlowing     = 3;      // Stochastic slowing
input int        InpADXPeriod        = 8;      // ADX period

input group "=== Risk Management ==="
input double     InpLot              = 1.0;    // Lot size
input int        InpMagic            = 5050;   // Magic number
input double     InpStopPoints       = 50.0;   // Stop Loss POINTS (used if ATR disabled)
input double     InpTakePoints       = 100.0;  // Take Profit POINTS (used if ATR disabled)

input group "=== ATR-Based SL/TP ==="
input bool       InpUseATR           = false;  // ATR-based SL/TP instead of points
input int        InpATRPeriod        = 14;     // ATR period
input double     InpATRSLMultiplier  = 1.5;   // SL = ATR × multiplier
input double     InpATRTPMultiplier  = 3.0;   // TP = ATR × multiplier

input group "=== Display Options ==="
input bool       InpShowPanel        = true;   // Show information panel

//--- Global variables
long      onnx_handle       = INVALID_HANDLE;
CTrade    m_trade;

//--- Inference control
datetime  g_last_infer      = 0;
datetime  g_last_traded_bar = 0;

//--- Panel display variables
long     g_prediction  = 0;    // 0=HOLD, 1=BUY, 2=SELL
float    g_conf_buy    = 0.0;
float    g_conf_sell   = 0.0;
float    g_conf_hold   = 0.0;
double   g_curr_adx    = 0.0;
double   g_curr_pdi    = 0.0;
double   g_curr_mdi    = 0.0;
double   g_stoch_k     = 0.0;
double   g_stoch_d     = 0.0;
int      g_infer_count = 0;

//--- Indicator handles (created once in OnInit)
int      g_adx_handle   = INVALID_HANDLE;
int      g_stoch_handle = INVALID_HANDLE;
int      g_atr_handle   = INVALID_HANDLE;

//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit()
{
   //--- Initialize ONNX model
   onnx_handle = OnnxCreate(InpModelName, ONNX_DEFAULT);
   if(onnx_handle == INVALID_HANDLE)
   {
      Print("ERROR: Cannot load ONNX model: ", InpModelName);
      Print("   GetLastError: ", GetLastError());
      Print("   Make sure the file is in MQL5/Files/ folder");
      return INIT_FAILED;
   }
   
   Print("ONNX model loaded: ", InpModelName);
   
   //--- Set input shape [1, window_size * features_per_bar]
   int num_inputs = InpWindowSize * InpFeaturesPerBar;
   long input_shape[] = {1, num_inputs};
   
   if(!OnnxSetInputShape(onnx_handle, 0, input_shape))
   {
      Print("Warning: Could not set input shape explicitly");
      Print("   The model will try to infer it automatically");
   }
   else
   {
      PrintFormat("Input shape set: [1, %d] (%d bars × %d features)", 
                  num_inputs, InpWindowSize, InpFeaturesPerBar);
   }
   
   //--- Set output shapes
   // Output 0: label [1]
   long out_shape_label[] = {1};
   OnnxSetOutputShape(onnx_handle, 0, out_shape_label);
   
   // Output 1: probabilities [1, 3] for classes [HOLD, BUY, SELL]
   long out_shape_probs[] = {1, 3};
   OnnxSetOutputShape(onnx_handle, 1, out_shape_probs);
   
   //--- Initialize indicators
   g_adx_handle = iADX(_Symbol, _Period, InpADXPeriod);
   if(g_adx_handle == INVALID_HANDLE)
   {
      Print("ERROR: Cannot create ADX indicator");
      return INIT_FAILED;
   }
   
   g_stoch_handle = iStochastic(_Symbol, _Period,
                                InpStochK, InpStochD, InpStochSlowing,
                                MODE_SMA, STO_LOWHIGH);
   if(g_stoch_handle == INVALID_HANDLE)
   {
      Print("ERROR: Cannot create Stochastic indicator");
      return INIT_FAILED;
   }

   //--- Initialize ATR indicator
   g_atr_handle = iATR(_Symbol, _Period, InpATRPeriod);
   if(g_atr_handle == INVALID_HANDLE)
   {
      Print("ERROR: Cannot create ATR indicator");
      return INIT_FAILED;
   }

   Print("Indicators created: ADX(", InpADXPeriod, ") + Stochastic(",
         InpStochK, ",", InpStochD, ",", InpStochSlowing, ") + ATR(", InpATRPeriod, ")");
   
   //--- Print symbol information
   double pt = SymbolInfoDouble(_Symbol, SYMBOL_POINT);
   int digits = (int)SymbolInfoInteger(_Symbol, SYMBOL_DIGITS);
   
   PrintFormat("\n=== SYMBOL INFORMATION ===");
   PrintFormat("Symbol: %s", _Symbol);
   PrintFormat("Timeframe: %s", EnumToString(_Period));
   PrintFormat("Digits: %d", digits);
   PrintFormat("Point: %.6f", pt);

   if(InpUseATR)
   {
      PrintFormat("SL/TP mode: ATR(%d) | SL mult: %.1f | TP mult: %.1f",
                  InpATRPeriod, InpATRSLMultiplier, InpATRTPMultiplier);
   }
   else
   {
      PrintFormat("SL/TP mode: Fixed points | SL: %.0f pts | TP: %.0f pts",
                  InpStopPoints, InpTakePoints);
   }
   
   //--- Setup timer if needed
   if(InpInferSeconds > 0)
   {
      EventSetTimer(InpInferSeconds);
      PrintFormat("Inference timer: every %d seconds", InpInferSeconds);
   }
   else
   {
      Print("Inference mode: New bar only");
   }
   
   //--- Setup trade object
   m_trade.SetExpertMagicNumber(InpMagic);
   m_trade.SetDeviationInPoints(10);
   m_trade.SetTypeFilling(ORDER_FILLING_FOK);
   
   PrintFormat("\n=== EA INITIALIZED SUCCESSFULLY ===\n");
   
   return INIT_SUCCEEDED;
}

//+------------------------------------------------------------------+
//| Expert deinitialization function                                 |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
{
   //--- Release ONNX handle
   if(onnx_handle != INVALID_HANDLE)
      OnnxRelease(onnx_handle);
   
   //--- Release indicator handles
   if(g_adx_handle != INVALID_HANDLE)
      IndicatorRelease(g_adx_handle);
   if(g_stoch_handle != INVALID_HANDLE)
      IndicatorRelease(g_stoch_handle);
   if(g_atr_handle != INVALID_HANDLE)
      IndicatorRelease(g_atr_handle);
   
   //--- Kill timer
   EventKillTimer();
   
   //--- Clear chart
   Comment("");
   
   PrintFormat("EA deinitialized. Reason: %s", GetDeinitReasonText(reason));
}

//+------------------------------------------------------------------+
//| Get deinit reason text                                           |
//+------------------------------------------------------------------+
string GetDeinitReasonText(int reason)
{
   switch(reason)
   {
      case REASON_PROGRAM:     return "Program terminated";
      case REASON_REMOVE:      return "EA removed from chart";
      case REASON_RECOMPILE:   return "EA recompiled";
      case REASON_CHARTCHANGE: return "Chart symbol/period changed";
      case REASON_CHARTCLOSE:  return "Chart closed";
      case REASON_PARAMETERS:  return "Input parameters changed";
      case REASON_ACCOUNT:     return "Account changed";
      default:                 return "Unknown reason";
   }
}

//+------------------------------------------------------------------+
//| Update global variables for panel display                        |
//+------------------------------------------------------------------+
void UpdateIndicatorGlobals()
{
   double adx_v[1], pdi_v[1], mdi_v[1];
   if(CopyBuffer(g_adx_handle, 0, 0, 1, adx_v) > 0) g_curr_adx = adx_v[0];
   if(CopyBuffer(g_adx_handle, 1, 0, 1, pdi_v) > 0) g_curr_pdi = pdi_v[0];
   if(CopyBuffer(g_adx_handle, 2, 0, 1, mdi_v) > 0) g_curr_mdi = mdi_v[0];
   
   double sk[1], sd[1];
   if(CopyBuffer(g_stoch_handle, 0, 0, 1, sk) > 0) g_stoch_k = sk[0];
   if(CopyBuffer(g_stoch_handle, 1, 0, 1, sd) > 0) g_stoch_d = sd[0];
}

//+------------------------------------------------------------------+
//| Prepare feature vector from recent bars                          |
//+------------------------------------------------------------------+
bool PrepareFeatures(float &features[])
{
   int total_features = InpWindowSize * InpFeaturesPerBar;
   ArrayResize(features, total_features);
   
   //--- Get historical data
   double adx_data[], pdi_data[], mdi_data[], stoch_k[], stoch_d[];
   
   ArraySetAsSeries(adx_data, true);
   ArraySetAsSeries(pdi_data, true);
   ArraySetAsSeries(mdi_data, true);
   ArraySetAsSeries(stoch_k, true);
   ArraySetAsSeries(stoch_d, true);
   
   if(CopyBuffer(g_adx_handle, 0, 0, InpWindowSize, adx_data) != InpWindowSize)
      return false;
   if(CopyBuffer(g_adx_handle, 1, 0, InpWindowSize, pdi_data) != InpWindowSize)
      return false;
   if(CopyBuffer(g_adx_handle, 2, 0, InpWindowSize, mdi_data) != InpWindowSize)
      return false;
   if(CopyBuffer(g_stoch_handle, 0, 0, InpWindowSize, stoch_k) != InpWindowSize)
      return false;
   if(CopyBuffer(g_stoch_handle, 1, 0, InpWindowSize, stoch_d) != InpWindowSize)
      return false;
   
   //--- Build feature vector: oldest to newest
   //--- Feature order: stoch_main, stoch_signal, adx, pdi, mdi
   for(int i = 0; i < InpWindowSize; i++)
   {
      int idx = i * InpFeaturesPerBar;
      int bar = InpWindowSize - 1 - i;  // Reverse: oldest first
      
      features[idx + 0] = (float)stoch_k[bar];    // feat_stoch_main
      features[idx + 1] = (float)stoch_d[bar];    // feat_stoch_signal
      features[idx + 2] = (float)adx_data[bar];   // feat_adx
      features[idx + 3] = (float)pdi_data[bar];   // feat_pdi
      features[idx + 4] = (float)mdi_data[bar];   // feat_mdi
   }
   
   return true;
}

//+------------------------------------------------------------------+
//| Calculate SL and TP distances based on ATR or fixed points       |
//+------------------------------------------------------------------+
bool GetSLTPDistance(double &sl_dist, double &tp_dist)
{
   double pt = SymbolInfoDouble(_Symbol, SYMBOL_POINT);

   if(InpUseATR)
   {
      double atr_buf[1];
      if(CopyBuffer(g_atr_handle, 0, 1, 1, atr_buf) != 1)
      {
         Print("ERROR: Cannot read ATR value");
         return false;
      }
      double atr = atr_buf[0];
      sl_dist = (InpATRSLMultiplier > 0) ? atr * InpATRSLMultiplier : 0;
      tp_dist = (InpATRTPMultiplier > 0) ? atr * InpATRTPMultiplier : 0;
      PrintFormat("ATR: %.5f | SL dist: %.5f | TP dist: %.5f", atr, sl_dist, tp_dist);
   }
   else
   {
      sl_dist = InpStopPoints * pt;
      tp_dist = InpTakePoints * pt;
   }

   return true;
}

//+------------------------------------------------------------------+
//| Run ONNX inference and execute trade if conditions met           |
//+------------------------------------------------------------------+
void RunInference()
{
   g_last_infer = TimeCurrent();
   g_infer_count++;
   
   //--- Prepare features
   float features[];
   if(!PrepareFeatures(features))
   {
      Print("ERROR: Cannot prepare features");
      return;
   }
   
   //--- Prepare output arrays (must be sized according to model output shapes)
   float output_label[1];     // Output 0: label [1]
   float output_probs[3];     // Output 1: probabilities [1, 3] for classes [HOLD, BUY, SELL]
   
   //--- Run ONNX inference - outputs are populated directly in the arrays
   if(!OnnxRun(onnx_handle, ONNX_DEFAULT, features, output_label, output_probs))
   {
      Print("ERROR: OnnxRun failed. Error: ", GetLastError());
      return;
   }
   
   //--- Update global state (outputs are already populated from OnnxRun)
   g_prediction = (long)output_label[0];
   g_conf_hold = output_probs[0];
   g_conf_buy = output_probs[1];
   g_conf_sell = output_probs[2];
   
   //--- Get active confidence
   float active_conf = 0.0;
   if(g_prediction == 1)
      active_conf = g_conf_buy;
   else if(g_prediction == 2)
      active_conf = g_conf_sell;
   
   //--- Check trading conditions
   MqlDateTime dt;
   TimeCurrent(dt);
   bool time_ok = (dt.hour >= InpStartHour && dt.hour < InpEndHour);
   
   datetime current_bar = iTime(_Symbol, _Period, 0);
   bool bar_ok = true;
   if(InpOneTradePerBar)
      bar_ok = (current_bar != g_last_traded_bar);
   
   bool no_position = !PositionSelect(_Symbol);
   
   //--- Log decision
   if(g_prediction > 0)
   {
      PrintFormat("Inference #%d: %s signal | Conf: %.2f%% | Time: %s | Bar: %s | Position: %s",
                  g_infer_count,
                  (g_prediction == 1) ? "BUY" : "SELL",
                  active_conf * 100,
                  time_ok ? "OK" : "CLOSED",
                  bar_ok ? "OK" : "SKIP",
                  no_position ? "NONE" : "OPEN");
   }
   
   //--- Execute trade based EXCLUSIVELY on ONNX inference
   if(g_prediction > 0 && active_conf >= InpMinConf && time_ok && bar_ok && no_position)
   {
      int digits = (int)SymbolInfoInteger(_Symbol, SYMBOL_DIGITS);
      double sl_dist, tp_dist;

      if(!GetSLTPDistance(sl_dist, tp_dist))
         return;

      if(g_prediction == 1)  // BUY
      {
         double ask = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
         double sl = NormalizeDouble(ask - sl_dist, digits);
         double tp = NormalizeDouble(ask + tp_dist, digits);
         
         PrintFormat("Opening BUY: Price=%.5f, SL=%.5f, TP=%.5f, Lot=%.2f",
                     ask, sl, tp, InpLot);
         
         if(m_trade.Buy(InpLot, _Symbol, ask, sl, tp, 
            "SGRADT50 BUY @" + DoubleToString(active_conf * 100, 1) + "%"))
         {
            g_last_traded_bar = current_bar;
            Print("BUY order executed successfully");
         }
         else
         {
            Print("BUY order failed. Error: ", m_trade.ResultRetcode(), 
                  " - ", m_trade.ResultRetcodeDescription());
         }
      }
      else if(g_prediction == 2)  // SELL
      {
         double bid = SymbolInfoDouble(_Symbol, SYMBOL_BID);
         double sl = NormalizeDouble(bid + sl_dist, digits);
         double tp = NormalizeDouble(bid - tp_dist, digits);
         
         PrintFormat("Opening SELL: Price=%.5f, SL=%.5f, TP=%.5f, Lot=%.2f",
                     bid, sl, tp, InpLot);
         
         if(m_trade.Sell(InpLot, _Symbol, bid, sl, tp,
            "SGRADT50 SELL @" + DoubleToString(active_conf * 100, 1) + "%"))
         {
            g_last_traded_bar = current_bar;
            Print("SELL order executed successfully");
         }
         else
         {
            Print("SELL order failed. Error: ", m_trade.ResultRetcode(),
                  " - ", m_trade.ResultRetcodeDescription());
         }
      }
   }
   else if(g_prediction > 0)
   {
      // Log why trade was rejected
      if(active_conf < InpMinConf)
         PrintFormat("Signal rejected: Confidence %.2f%% < threshold %.1f%%",
                     active_conf * 100, InpMinConf * 100);
      if(!time_ok)
         Print("Signal rejected: Outside trading hours");
      if(!bar_ok)
         Print("Signal rejected: Already traded this bar");
      if(!no_position)
         Print("Signal rejected: Position already open");
   }
}

//+------------------------------------------------------------------+
//| OnTick function                                                  |
//+------------------------------------------------------------------+
void OnTick()
{
   //--- Update indicators for display
   UpdateIndicatorGlobals();
   
   //--- Run inference based on mode
   if(InpInferSeconds <= 0)
   {
      // Mode: inference on new bar only
      static datetime last_bar = 0;
      datetime current_bar = iTime(_Symbol, _Period, 0);
      
      if(current_bar != last_bar)
      {
         last_bar = current_bar;
         RunInference();
      }
   }
   // If InpInferSeconds > 0, inference is handled by OnTimer
   
   //--- Update display
   if(InpShowPanel)
      ShowStatus();
}

//+------------------------------------------------------------------+
//| OnTimer function                                                 |
//+------------------------------------------------------------------+
void OnTimer()
{
   if(InpInferSeconds > 0)
   {
      RunInference();
      
      if(InpShowPanel)
         ShowStatus();
   }
}

//+------------------------------------------------------------------+
//| Display status panel (REDUCED VERSION)                           |
//+------------------------------------------------------------------+
void ShowStatus()
{
   MqlDateTime dt;
   TimeCurrent(dt);
   bool valid_time = (dt.hour >= InpStartHour && dt.hour < InpEndHour);
   
   string signal_text = "HOLD";
   if(g_prediction == 1)
      signal_text = "BUY";
   else if(g_prediction == 2)
      signal_text = "SELL";
   
   string info = "\n";
   
   info += "SESSION: " + StringFormat("%02d:00-%02d:00", InpStartHour, InpEndHour);
   info += " [" + (valid_time ? "ACTIVE" : "CLOSED") + "]\n";
   info += "MODE: " + ((InpInferSeconds > 0) ? "TIMER " + (string)InpInferSeconds + "s" : "NEW BAR");
   info += " | Runs: " + (string)g_infer_count + "\n";
   
   if(g_last_infer > 0)
      info += "Last: " + TimeToString(g_last_infer, TIME_SECONDS) + "\n";
   
   info += "---\n";
   info += "INDICATORS\n";
   info += "ADX: " + DoubleToString(g_curr_adx, 1);
   info += " | +DI: " + DoubleToString(g_curr_pdi, 1);
   info += " | -DI: " + DoubleToString(g_curr_mdi, 1) + "\n";
   info += "Stoch K: " + DoubleToString(g_stoch_k, 1);
   info += " | D: " + DoubleToString(g_stoch_d, 1) + "\n";
   
   info += "---\n";
   info += "AI SIGNAL: " + signal_text + "\n";
   info += "HOLD: " + DoubleToString(g_conf_hold * 100, 1) + "%";
   info += " | BUY: " + DoubleToString(g_conf_buy * 100, 1) + "%";
   info += " | SELL: " + DoubleToString(g_conf_sell * 100, 1) + "%\n";
   info += "Min Conf: " + DoubleToString(InpMinConf * 100, 1) + "%\n";
   
   info += "---\n";
   info += "LOT: " + DoubleToString(InpLot, 2);
   if(InpUseATR)
   {
      double atr_buf[1];
      string atr_str = "N/A";
      if(CopyBuffer(g_atr_handle, 0, 1, 1, atr_buf) == 1)
         atr_str = DoubleToString(atr_buf[0], _Digits);
      info += " | SL/TP: ATR(" + (string)InpATRPeriod + ")=" + atr_str;
      info += " x" + DoubleToString(InpATRSLMultiplier, 1);
      info += "/" + DoubleToString(InpATRTPMultiplier, 1) + "\n";
   }
   else
   {
      info += " | SL: " + DoubleToString(InpStopPoints, 0);
      info += " | TP: " + DoubleToString(InpTakePoints, 0) + "\n";
   }
   
   if(PositionSelect(_Symbol))
   {
      long pos_type = PositionGetInteger(POSITION_TYPE);
      double profit = PositionGetDouble(POSITION_PROFIT);
      string type_str = (pos_type == POSITION_TYPE_BUY) ? "BUY" : "SELL";
      string profit_str = (profit >= 0) ? "+" + DoubleToString(profit, 2) : DoubleToString(profit, 2);
      
      info += "---\n";
      info += "POSITION: " + type_str + " | PL: " + profit_str + " " + AccountInfoString(ACCOUNT_CURRENCY) + "\n";
   }
   else
   {
      info += "---\n";
      info += "Waiting for signal\n";
   }
   
   Comment(info);
}
//+------------------------------------------------------------------+