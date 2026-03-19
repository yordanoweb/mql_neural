//+------------------------------------------------------------------+
//|                        EA_SGRADT50_ONNX.mq5                      |
//|                        Compatible with SGRADT 5.0 Strategy       |
//+------------------------------------------------------------------+
#property strict
#include <Trade\Trade.mqh>

input group "=== AI Model Configuration ==="
input string     InpModelName        = "EUR_USD_H1_SGRADT50_combined.onnx";  // ONNX Model filename
input string     InpMetaFile         = "EUR_USD_H1_SGRADT50_combined.meta.json"; // Metadata file (optional)
input float      InpMinConf          = 0.55;   // Minimum confidence threshold
input int        InpWindowSize       = 20;     // Window size (from training)
input int        InpFeaturesPerBar   = 7;      // Features per bar (from training)

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
input double     InpStochOversold    = 20.0;   // Oversold level
input double     InpStochOverbought  = 80.0;   // Overbought level
input int        InpADXPeriod        = 8;      // ADX period
input double     InpADXLimit         = 32.0;   // ADX trend threshold

input group "=== Risk Management ==="
input double     InpLot              = 1.0;    // Lot size
input int        InpMagic            = 5050;   // Magic number
input double     InpStopPoints       = 50.0;   // Stop Loss in POINTS
input double     InpTakePoints       = 100.0;  // Take Profit in POINTS

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

//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit()
{
   //--- Initialize ONNX model
   onnx_handle = OnnxCreate(InpModelName, ONNX_DEFAULT);
   if(onnx_handle == INVALID_HANDLE)
   {
      Print("❌ ERROR: Cannot load ONNX model: ", InpModelName);
      Print("   GetLastError: ", GetLastError());
      Print("   Make sure the file is in MQL5/Files/ folder");
      return INIT_FAILED;
   }
   
   Print("✓ ONNX model loaded: ", InpModelName);
   
   //--- Set input shape [1, window_size * features_per_bar]
   int num_inputs = InpWindowSize * InpFeaturesPerBar;
   long input_shape[] = {1, num_inputs};
   
   if(!OnnxSetInputShape(onnx_handle, 0, input_shape))
   {
      Print("⚠ Warning: Could not set input shape explicitly");
      Print("   The model will try to infer it automatically");
   }
   else
   {
      PrintFormat("✓ Input shape set: [1, %d] (%d bars × %d features)", 
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
      Print("❌ ERROR: Cannot create ADX indicator");
      return INIT_FAILED;
   }
   
   g_stoch_handle = iStochastic(_Symbol, _Period,
                                InpStochK, InpStochD, InpStochSlowing,
                                MODE_SMA, STO_LOWHIGH);
   if(g_stoch_handle == INVALID_HANDLE)
   {
      Print("❌ ERROR: Cannot create Stochastic indicator");
      return INIT_FAILED;
   }
   
   Print("✓ Indicators created: ADX(", InpADXPeriod, ") + Stochastic(", 
         InpStochK, ",", InpStochD, ",", InpStochSlowing, ")");
   
   //--- Print symbol information
   double pt = SymbolInfoDouble(_Symbol, SYMBOL_POINT);
   int digits = (int)SymbolInfoInteger(_Symbol, SYMBOL_DIGITS);
   
   PrintFormat("\n=== SYMBOL INFORMATION ===");
   PrintFormat("Symbol: %s", _Symbol);
   PrintFormat("Timeframe: %s", EnumToString(_Period));
   PrintFormat("Digits: %d", digits);
   PrintFormat("Point: %.6f", pt);
   PrintFormat("SL: %.0f points = %.6f price", InpStopPoints, InpStopPoints * pt);
   PrintFormat("TP: %.0f points = %.6f price", InpTakePoints, InpTakePoints * pt);
   
   //--- Setup timer if needed
   if(InpInferSeconds > 0)
   {
      EventSetTimer(InpInferSeconds);
      PrintFormat("✓ Inference timer: every %d seconds", InpInferSeconds);
   }
   else
   {
      Print("✓ Inference mode: New bar only");
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
//| Main inference function                                          |
//+------------------------------------------------------------------+
void RunInference()
{
   //--- Prepare arrays for data collection
   double close[], open[], high[], low[];
   double adx_b[], pdi_b[], mdi_b[];
   double stoch_k_b[], stoch_d_b[];
   
   //--- Set arrays as series (index 0 = most recent)
   ArraySetAsSeries(close, true);
   ArraySetAsSeries(open, true);
   ArraySetAsSeries(high, true);
   ArraySetAsSeries(low, true);
   ArraySetAsSeries(adx_b, true);
   ArraySetAsSeries(pdi_b, true);
   ArraySetAsSeries(mdi_b, true);
   ArraySetAsSeries(stoch_k_b, true);
   ArraySetAsSeries(stoch_d_b, true);
   
   //--- Copy price data
   if(CopyClose(_Symbol, _Period, 0, InpWindowSize, close) < InpWindowSize)
   {
      Print("⚠ Warning: Not enough close data");
      return;
   }
   if(CopyOpen(_Symbol, _Period, 0, InpWindowSize, open) < InpWindowSize)
   {
      Print("⚠ Warning: Not enough open data");
      return;
   }
   if(CopyHigh(_Symbol, _Period, 0, InpWindowSize, high) < InpWindowSize)
   {
      Print("⚠ Warning: Not enough high data");
      return;
   }
   if(CopyLow(_Symbol, _Period, 0, InpWindowSize, low) < InpWindowSize)
   {
      Print("⚠ Warning: Not enough low data");
      return;
   }
   
   //--- Copy indicator data
   if(CopyBuffer(g_adx_handle, 0, 0, InpWindowSize, adx_b) < InpWindowSize)
   {
      Print("⚠ Warning: Not enough ADX data");
      return;
   }
   if(CopyBuffer(g_adx_handle, 1, 0, InpWindowSize, pdi_b) < InpWindowSize)
   {
      Print("⚠ Warning: Not enough +DI data");
      return;
   }
   if(CopyBuffer(g_adx_handle, 2, 0, InpWindowSize, mdi_b) < InpWindowSize)
   {
      Print("⚠ Warning: Not enough -DI data");
      return;
   }
   if(CopyBuffer(g_stoch_handle, 0, 0, InpWindowSize, stoch_k_b) < InpWindowSize)
   {
      Print("⚠ Warning: Not enough Stochastic %K data");
      return;
   }
   if(CopyBuffer(g_stoch_handle, 1, 0, InpWindowSize, stoch_d_b) < InpWindowSize)
   {
      Print("⚠ Warning: Not enough Stochastic %D data");
      return;
   }
   
   //--- Build input buffer
   // Feature order MUST match training: body, range, stoch_main, stoch_signal, adx, pdi, mdi
   // Array organized as: [bar0_feat0, bar0_feat1, ..., bar0_feat6, bar1_feat0, bar1_feat1, ...]
   // bar0 = oldest, bar(window_size-1) = most recent
   
   float input_buffer[];
   ArrayResize(input_buffer, InpWindowSize * InpFeaturesPerBar);
   
   for(int i = 0; i < InpWindowSize; i++)
   {
      // Convert array index: i=0 is most recent in series arrays
      // But we need oldest first in input_buffer
      int idx = InpWindowSize - 1 - i;  // idx=0 -> oldest bar
      int offset = idx * InpFeaturesPerBar;
      
      // Feature 0: body (close - open)
      input_buffer[offset + 0] = (float)(close[i] - open[i]);
      
      // Feature 1: range (high - low)
      input_buffer[offset + 1] = (float)(high[i] - low[i]);
      
      // Feature 2: stoch_main (%K)
      input_buffer[offset + 2] = (float)stoch_k_b[i];
      
      // Feature 3: stoch_signal (%D)
      input_buffer[offset + 3] = (float)stoch_d_b[i];
      
      // Feature 4: ADX
      input_buffer[offset + 4] = (float)adx_b[i];
      
      // Feature 5: +DI
      input_buffer[offset + 5] = (float)pdi_b[i];
      
      // Feature 6: -DI
      input_buffer[offset + 6] = (float)mdi_b[i];
   }
   
   //--- Prepare output arrays
   long out_label[];
   float out_probs[];
   ArrayResize(out_label, 1);
   ArrayResize(out_probs, 3);
   
   //--- Run ONNX inference
   if(!OnnxRun(onnx_handle, ONNX_NO_CONVERSION, input_buffer, out_label, out_probs))
   {
      Print("❌ ERROR: ONNX inference failed. Error: ", GetLastError());
      return;
   }
   
   //--- Store results
   g_prediction  = out_label[0];
   g_conf_hold   = out_probs[0];  // Class 0: HOLD
   g_conf_buy    = out_probs[1];  // Class 1: BUY
   g_conf_sell   = out_probs[2];  // Class 2: SELL
   g_infer_count++;
   g_last_infer  = TimeCurrent();
   
   //--- Determine confidence for current prediction
   float active_conf = 0.0;
   if(g_prediction == 1)
      active_conf = g_conf_buy;
   else if(g_prediction == 2)
      active_conf = g_conf_sell;
   else
      active_conf = g_conf_hold;
   
   //--- Check session time
   MqlDateTime dt;
   TimeCurrent(dt);
   bool time_ok = (dt.hour >= InpStartHour && dt.hour < InpEndHour);
   
   //--- Check if we can trade this bar
   datetime current_bar = iTime(_Symbol, _Period, 0);
   bool bar_ok = (!InpOneTradePerBar || g_last_traded_bar != current_bar);
   
   //--- Check if position already exists
   bool no_position = !PositionSelect(_Symbol);
   
   //--- Log inference result
   if(g_prediction > 0)
   {
      PrintFormat("🔍 Inference #%d: %s (conf: %.2f%%) | ADX: %.1f | Stoch: %.1f/%.1f",
                  g_infer_count,
                  (g_prediction == 1 ? "BUY" : "SELL"),
                  active_conf * 100,
                  g_curr_adx,
                  g_stoch_k,
                  g_stoch_d);
   }
   
   //--- Execute trade if all conditions met
   if(g_prediction > 0 && active_conf >= InpMinConf && time_ok && bar_ok && no_position)
   {
      double pt = SymbolInfoDouble(_Symbol, SYMBOL_POINT);
      int digits = (int)SymbolInfoInteger(_Symbol, SYMBOL_DIGITS);
      double sl_dist = InpStopPoints * pt;
      double tp_dist = InpTakePoints * pt;
      
      if(g_prediction == 1)  // BUY
      {
         double ask = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
         double sl = NormalizeDouble(ask - sl_dist, digits);
         double tp = NormalizeDouble(ask + tp_dist, digits);
         
         PrintFormat("📈 Opening BUY: Price=%.5f, SL=%.5f, TP=%.5f, Lot=%.2f",
                     ask, sl, tp, InpLot);
         
         if(m_trade.Buy(InpLot, _Symbol, ask, sl, tp, 
            "SGRADT50 BUY @" + DoubleToString(active_conf * 100, 1) + "%"))
         {
            g_last_traded_bar = current_bar;
            Print("✅ BUY order executed successfully");
         }
         else
         {
            Print("❌ BUY order failed. Error: ", m_trade.ResultRetcode(), 
                  " - ", m_trade.ResultRetcodeDescription());
         }
      }
      else if(g_prediction == 2)  // SELL
      {
         double bid = SymbolInfoDouble(_Symbol, SYMBOL_BID);
         double sl = NormalizeDouble(bid + sl_dist, digits);
         double tp = NormalizeDouble(bid - tp_dist, digits);
         
         PrintFormat("📉 Opening SELL: Price=%.5f, SL=%.5f, TP=%.5f, Lot=%.2f",
                     bid, sl, tp, InpLot);
         
         if(m_trade.Sell(InpLot, _Symbol, bid, sl, tp,
            "SGRADT50 SELL @" + DoubleToString(active_conf * 100, 1) + "%"))
         {
            g_last_traded_bar = current_bar;
            Print("✅ SELL order executed successfully");
         }
         else
         {
            Print("❌ SELL order failed. Error: ", m_trade.ResultRetcode(),
                  " - ", m_trade.ResultRetcodeDescription());
         }
      }
   }
   else if(g_prediction > 0)
   {
      // Log why we didn't trade
      if(active_conf < InpMinConf)
         PrintFormat("⚠ Signal ignored: Confidence %.2f%% < threshold %.1f%%",
                     active_conf * 100, InpMinConf * 100);
      if(!time_ok)
         Print("⚠ Signal ignored: Outside trading hours");
      if(!bar_ok)
         Print("⚠ Signal ignored: Already traded this bar");
      if(!no_position)
         Print("⚠ Signal ignored: Position already open");
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
//| Display status panel                                             |
//+------------------------------------------------------------------+
void ShowStatus()
{
   MqlDateTime dt;
   TimeCurrent(dt);
   bool valid_time = (dt.hour >= InpStartHour && dt.hour < InpEndHour);
   
   string signal_text = "NO SIGNAL";
   color signal_color = clrGray;
   
   if(g_prediction == 1)
   {
      signal_text = "🟢 BUY";
      signal_color = clrLime;
   }
   else if(g_prediction == 2)
   {
      signal_text = "🔴 SELL";
      signal_color = clrRed;
   }
   
   string trend_status = (g_curr_adx >= InpADXLimit) ? "TRENDING" : "RANGING";
   
   string stoch_zone;
   if(g_stoch_k < InpStochOversold)
      stoch_zone = "OVERSOLD ⬇";
   else if(g_stoch_k > InpStochOverbought)
      stoch_zone = "OVERBOUGHT ⬆";
   else
      stoch_zone = "NEUTRAL ─";
   
   string stoch_cross = (g_stoch_k > g_stoch_d) ? "%K ABOVE %D ↑" : "%K BELOW %D ↓";
   
   double pt = SymbolInfoDouble(_Symbol, SYMBOL_POINT);
   
   string mode_str = (InpInferSeconds > 0)
                     ? "TIMER (" + (string)InpInferSeconds + "s)"
                     : "NEW BAR";
   
   //--- Build panel
   string info = "\n\n\n";
   info += "╔════════════════════════════════════════════╗\n";
   info += "║      SGRADT 5.0 - AI TRADING SYSTEM      ║\n";
   info += "╚════════════════════════════════════════════╝\n\n";
   
   info += "📊 SYMBOL: " + _Symbol + " [" + EnumToString(_Period) + "]\n";
   info += "⏰ SESSION: " + StringFormat("%02d:00-%02d:00", InpStartHour, InpEndHour);
   info += " [" + (valid_time ? "✓ ACTIVE" : "✗ CLOSED") + "]\n";
   info += "🔄 MODE: " + mode_str + " | Inferences: " + (string)g_infer_count + "\n";
   
   if(g_last_infer > 0)
      info += "🕐 Last Run: " + TimeToString(g_last_infer, TIME_SECONDS) + "\n";
   
   info += "\n────────────────────────────────────────────\n";
   info += "📈 ADX INDICATOR (Period: " + (string)InpADXPeriod + ")\n";
   info += "────────────────────────────────────────────\n";
   info += "   ADX: " + DoubleToString(g_curr_adx, 2) + " [" + trend_status + "]\n";
   info += "   +DI: " + DoubleToString(g_curr_pdi, 2) + "\n";
   info += "   -DI: " + DoubleToString(g_curr_mdi, 2) + "\n";
   
   info += "\n────────────────────────────────────────────\n";
   info += "📊 STOCHASTIC (" + (string)InpStochK + "," + (string)InpStochD + "," + (string)InpStochSlowing + ")\n";
   info += "────────────────────────────────────────────\n";
   info += "   %K: " + DoubleToString(g_stoch_k, 2) + "\n";
   info += "   %D: " + DoubleToString(g_stoch_d, 2) + "\n";
   info += "   Zone: " + stoch_zone + "\n";
   info += "   Cross: " + stoch_cross + "\n";
   
   info += "\n════════════════════════════════════════════\n";
   info += "🤖 AI PREDICTION\n";
   info += "════════════════════════════════════════════\n";
   info += "   Signal: " + signal_text + "\n";
   info += "\n   Confidence Levels:\n";
   info += "   ├─ HOLD:  " + DoubleToString(g_conf_hold * 100, 2) + "%\n";
   info += "   ├─ BUY:   " + DoubleToString(g_conf_buy  * 100, 2) + "%\n";
   info += "   └─ SELL:  " + DoubleToString(g_conf_sell * 100, 2) + "%\n";
   info += "\n   Minimum Required: " + DoubleToString(InpMinConf * 100, 1) + "%\n";
   
   info += "\n────────────────────────────────────────────\n";
   info += "💰 RISK SETTINGS\n";
   info += "────────────────────────────────────────────\n";
   info += "   Lot Size: " + DoubleToString(InpLot, 2) + "\n";
   info += "   Stop Loss:   " + DoubleToString(InpStopPoints, 0) + " pts";
   info += " (" + DoubleToString(InpStopPoints * pt, 5) + ")\n";
   info += "   Take Profit: " + DoubleToString(InpTakePoints, 0) + " pts";
   info += " (" + DoubleToString(InpTakePoints * pt, 5) + ")\n";
   
   info += "\n════════════════════════════════════════════\n";
   
   if(PositionSelect(_Symbol))
   {
      long pos_type = PositionGetInteger(POSITION_TYPE);
      double profit = PositionGetDouble(POSITION_PROFIT);
      string type_str = (pos_type == POSITION_TYPE_BUY) ? "BUY 📈" : "SELL 📉";
      string profit_str = (profit >= 0) ? "+" + DoubleToString(profit, 2) : DoubleToString(profit, 2);
      
      info += "💼 ACTIVE POSITION: " + type_str + "\n";
      info += "   P&L: " + profit_str + " " + AccountInfoString(ACCOUNT_CURRENCY) + "\n";
   }
   else
   {
      info += "⏳ Status: Waiting for signal...\n";
   }
   
   info += "════════════════════════════════════════════\n";
   
   Comment(info);
}
//+------------------------------------------------------------------+
