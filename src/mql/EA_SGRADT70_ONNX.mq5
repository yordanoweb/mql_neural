//+------------------------------------------------------------------+
//|                                      EA_SGRADT70_ONNX_v3.mq5     |
//|                   SGRADT 7.0 v3 - NN-Driven Strategy (6 Features)|
//|              Neural Network Makes ALL Decisions (No EMA Gate)    |
//+------------------------------------------------------------------+
#property copyright "SGRADT 7.0 v3"
#property version   "7.03"
#property strict

#include <Trade\Trade.mqh>

//--- Input Parameters

//=== AI Model Configuration ===
input group "======== AI MODEL ========"
input string InpModelName = "USTEC_M5_SGRADT70_v3.onnx";  // Model filename
input string InpMetaFile  = "USTEC_M5_SGRADT70_v3.meta.json"; // Metadata file (optional)
input double InpMinConf   = 0.55;      // Minimum confidence (0.0-1.0)
input int    InpWindowSize = 20;       // Window size (must match training)
input int    InpFeaturesPerBar = 6;    // Features per bar (ALWAYS 6 for v3)

//=== Inference Timing ===
input group "======== INFERENCE ========"
input int  InpInferSeconds = 0;        // Inference frequency (0 = new bar only)
input bool InpOneTradePerBar = true;   // Limit to 1 trade per bar

//=== Trading Session ===
input group "======== SESSION ========"
input int InpStartHour = 0;            // Session start hour (0-23)
input int InpEndHour   = 24;           // Session end hour (0-24)

//=== Indicator Parameters ===
input group "======== INDICATORS ========"
input int InpStochK    = 7;            // Stochastic K period
input int InpStochD    = 3;            // Stochastic D smoothing
input int InpADXPeriod = 8;            // ADX period

//=== Risk Management (ATR-based) ===
input group "======== RISK (ATR-BASED) ========"
input double InpLot           = 1.0;   // Lot size
input int    InpMagic         = 7073;  // Magic number (v3)
input int    InpATRPeriod     = 14;    // ATR period
input double InpATRMultiplierSL = 2.0; // ATR multiplier for Stop Loss
input double InpATRMultiplierTP = 3.0; // ATR multiplier for Take Profit

//=== Display ===
input group "======== DISPLAY ========"
input bool InpShowPanel = true;        // Show information panel

//--- Global Variables
CTrade trade;
long   g_onnx_handle  = INVALID_HANDLE;
int    g_adx_handle   = INVALID_HANDLE;
int    g_stoch_handle = INVALID_HANDLE;
int    g_atr_handle   = INVALID_HANDLE;

datetime g_last_bar_time = 0;
datetime g_last_trade_bar = 0;
datetime g_last_inference_time = 0;
int      g_inference_count = 0;

double   g_last_probas[3];
int      g_last_prediction = -1;

//+------------------------------------------------------------------+
//| Expert initialization function                                    |
//+------------------------------------------------------------------+
int OnInit()
  {
   Print("\n", StringRepeat("-", 70));
   Print("    SGRADT 7.0 v3 - NN-DRIVEN STRATEGY (6 Features)");
   Print("    No EMA Gate - Volume Gate Only");
   Print(StringRepeat("-", 70), "\n");

//--- Load ONNX model
   string model_path = InpModelName;
   Print("Loading ONNX model: ", model_path);

   g_onnx_handle = OnnxCreate(InpModelName, ONNX_DEFAULT);

   if(g_onnx_handle == INVALID_HANDLE)
     {
      Print("[ERROR] Cannot load ONNX model");
      Print("        Make sure the model is in MQL5/Files/ folder");
      return INIT_FAILED;
     }

   Print("[OK] ONNX model loaded successfully");

//--- Set input shape
   int num_inputs = InpFeaturesPerBar * InpWindowSize;
   long input_shape[] = {1, num_inputs};

   if(!OnnxSetInputShape(g_onnx_handle, 0, input_shape))
     {
      Print("[ERROR] Cannot set input shape");
      return INIT_FAILED;
     }

   Print("[OK] Input shape set: [1, ", num_inputs, "] (", InpWindowSize, " bars x ", InpFeaturesPerBar, " features)");

//--- Create indicator handles
   Print("\nInitializing indicators...");

   g_adx_handle = iADX(_Symbol, _Period, InpADXPeriod);
   if(g_adx_handle == INVALID_HANDLE)
     {
      Print("[ERROR] Cannot create ADX indicator");
      return INIT_FAILED;
     }

   g_stoch_handle = iStochastic(_Symbol, _Period,
                                InpStochK, InpStochD, 3,
                                MODE_SMA, STO_LOWHIGH);
   if(g_stoch_handle == INVALID_HANDLE)
     {
      Print("[ERROR] Cannot create Stochastic indicator");
      return INIT_FAILED;
     }

   g_atr_handle = iATR(_Symbol, _Period, InpATRPeriod);
   if(g_atr_handle == INVALID_HANDLE)
     {
      Print("[ERROR] Cannot create ATR indicator");
      return INIT_FAILED;
     }

   Print("[OK] Indicators created:");
   Print("     - ADX(", InpADXPeriod, ")");
   Print("     - Stochastic(", InpStochK, ",", InpStochD, ")");
   Print("     - ATR(", InpATRPeriod, ")");

//--- Wait for indicators to calculate
   Print("\nWaiting for indicators to calculate data...");
   int attempts = 0;
   int max_attempts = 50;
   int required_bars = InpWindowSize + 30;
   bool indicators_ready = false;

   while(attempts < max_attempts)
     {
      double test_buf[];

      int adx_count = CopyBuffer(g_adx_handle, 0, 0, required_bars, test_buf);
      int stoch_count = CopyBuffer(g_stoch_handle, 0, 0, required_bars, test_buf);
      int atr_count = CopyBuffer(g_atr_handle, 0, 0, required_bars, test_buf);

      if(adx_count == required_bars &&
         stoch_count == required_bars &&
         atr_count == required_bars)
        {
         Print("[OK] All indicators ready with ", required_bars, " bars");
         indicators_ready = true;
         break;
        }

      Sleep(100);
      attempts++;
     }

   if(!indicators_ready)
     {
      Print("[WARNING] Indicators may not have enough data yet");
      Print("          This is normal on first run");
      Print("          EA will skip inference until data is ready");
     }

//--- Initialize trade
   trade.SetExpertMagicNumber(InpMagic);
   trade.SetDeviationInPoints(10);
   trade.SetTypeFilling(ORDER_FILLING_FOK);
   trade.SetAsyncMode(false);

//--- Print configuration
   PrintConfiguration();

   ArrayInitialize(g_last_probas, 0.0);

   Print("\n", StringRepeat("-", 70));
   Print("    EA INITIALIZED SUCCESSFULLY");
   Print(StringRepeat("-", 70), "\n");

   return INIT_SUCCEEDED;
  }

//+------------------------------------------------------------------+
//| Expert deinitialization function                                  |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
  {
   if(g_onnx_handle != INVALID_HANDLE)
      OnnxRelease(g_onnx_handle);

   if(g_adx_handle != INVALID_HANDLE)
      IndicatorRelease(g_adx_handle);

   if(g_stoch_handle != INVALID_HANDLE)
      IndicatorRelease(g_stoch_handle);

   if(g_atr_handle != INVALID_HANDLE)
      IndicatorRelease(g_atr_handle);

   Comment("");

   Print("\n", StringRepeat("-", 70));
   Print("    EA DEINITIALIZED");
   Print(StringRepeat("-", 70), "\n");
  }

//+------------------------------------------------------------------+
//| Expert tick function                                              |
//+------------------------------------------------------------------+
void OnTick()
  {
//--- Check session
   if(!IsWithinSession())
     {
      if(InpShowPanel)
         UpdatePanel();
      return;
     }

//--- Check for new bar or time-based inference
   datetime current_bar_time = iTime(_Symbol, _Period, 0);
   bool is_new_bar = (current_bar_time != g_last_bar_time);

   bool should_infer = false;

   if(InpInferSeconds == 0)
     {
      // New bar mode
      if(is_new_bar)
        {
         should_infer = true;
         g_last_bar_time = current_bar_time;
        }
     }
   else
     {
      // Time-based mode
      datetime current_time = TimeCurrent();
      if((current_time - g_last_inference_time) >= InpInferSeconds)
        {
         should_infer = true;
         g_last_inference_time = current_time;

         if(is_new_bar)
            g_last_bar_time = current_bar_time;
        }
     }

//--- Run inference if needed
   if(should_infer)
     {
      // Prepare input buffer
      float input_buffer[];
      if(!PrepareInput(input_buffer))
        {
         Print("[WARNING] Cannot prepare input buffer - skipping inference");
         if(InpShowPanel)
            UpdatePanel();
         return;
        }

      // Run ONNX inference
      float probas[];
      if(!OnnxRun(g_onnx_handle, ONNX_DEFAULT, input_buffer, probas))
        {
         Print("[ERROR] ONNX inference failed");
         if(InpShowPanel)
            UpdatePanel();
         return;
        }

      g_inference_count++;

      // Get predicted class and probabilities
      int predicted_class = 0;
      double max_proba = probas[0];

      for(int i = 1; i < 3; i++)
        {
         if(probas[i] > max_proba)
           {
            max_proba = probas[i];
            predicted_class = i;
           }
        }

      // Store for display
      g_last_prediction = predicted_class;
      ArrayCopy(g_last_probas, probas);

      // Check confidence threshold
      if(max_proba < InpMinConf)
        {
         if(InpShowPanel)
            UpdatePanel();
         return;
        }

      // Execute trades based on NN decision ONLY
      // No manual gates - neural network makes ALL decisions

      if(predicted_class == 1)  // BUY signal
        {
         if(!HasPosition(POSITION_TYPE_BUY))
           {
            // Check one-trade-per-bar rule
            if(InpOneTradePerBar && g_last_trade_bar == current_bar_time)
              {
               if(InpShowPanel)
                  UpdatePanel();
               return;
              }

            // Close any opposite position
            if(HasPosition(POSITION_TYPE_SELL))
              {
               for(int i = PositionsTotal() - 1; i >= 0; i--)
                 {
                  if(PositionGetSymbol(i) == _Symbol &&
                     PositionGetInteger(POSITION_MAGIC) == InpMagic &&
                     PositionGetInteger(POSITION_TYPE) == POSITION_TYPE_SELL)
                    {
                     ulong ticket = PositionGetInteger(POSITION_TICKET);
                     trade.PositionClose(ticket);
                    }
                 }
              }

            // Open BUY position with ATR-based SL/TP
            double ask = SymbolInfoDouble(_Symbol, SYMBOL_ASK);

            double atr_buf[];
            CopyBuffer(g_atr_handle, 0, 0, 1, atr_buf);
            double atr_value = atr_buf[0];

            double sl = ask - (atr_value * InpATRMultiplierSL);
            double tp = ask + (atr_value * InpATRMultiplierTP);

            if(trade.Buy(InpLot, _Symbol, ask, sl, tp, "SGRADT v3 BUY"))
              {
               Print("BUY opened | Conf: ", DoubleToString(max_proba * 100, 1), "% | SL: ",
                     DoubleToString(InpATRMultiplierSL, 1), "xATR | TP: ",
                     DoubleToString(InpATRMultiplierTP, 1), "xATR");
               g_last_trade_bar = current_bar_time;
              }
           }
        }
      else if(predicted_class == 2)  // SELL signal
        {
         if(!HasPosition(POSITION_TYPE_SELL))
           {
            // Check one-trade-per-bar rule
            if(InpOneTradePerBar && g_last_trade_bar == current_bar_time)
              {
               if(InpShowPanel)
                  UpdatePanel();
               return;
              }

            // Close any opposite position
            if(HasPosition(POSITION_TYPE_BUY))
              {
               for(int i = PositionsTotal() - 1; i >= 0; i--)
                 {
                  if(PositionGetSymbol(i) == _Symbol &&
                     PositionGetInteger(POSITION_MAGIC) == InpMagic &&
                     PositionGetInteger(POSITION_TYPE) == POSITION_TYPE_BUY)
                    {
                     ulong ticket = PositionGetInteger(POSITION_TICKET);
                     trade.PositionClose(ticket);
                    }
                 }
              }

            // Open SELL position with ATR-based SL/TP
            double bid = SymbolInfoDouble(_Symbol, SYMBOL_BID);

            double atr_buf[];
            CopyBuffer(g_atr_handle, 0, 0, 1, atr_buf);
            double atr_value = atr_buf[0];

            double sl = bid + (atr_value * InpATRMultiplierSL);
            double tp = bid - (atr_value * InpATRMultiplierTP);

            if(trade.Sell(InpLot, _Symbol, bid, sl, tp, "SGRADT v3 SELL"))
              {
               Print("SELL opened | Conf: ", DoubleToString(max_proba * 100, 1), "% | SL: ",
                     DoubleToString(InpATRMultiplierSL, 1), "xATR | TP: ",
                     DoubleToString(InpATRMultiplierTP, 1), "xATR");
               g_last_trade_bar = current_bar_time;
              }
           }
        }
      // predicted_class == 0 (HOLD) - do nothing
     }

//--- Update display
   if(InpShowPanel)
      UpdatePanel();
  }

//+------------------------------------------------------------------+
//| Prepare input buffer for ONNX (6 features per bar)               |
//+------------------------------------------------------------------+
bool PrepareInput(float &input_buffer[])
  {
   int window = InpWindowSize;
   int total_size = window * InpFeaturesPerBar;  // 6 features per bar

   ArrayResize(input_buffer, total_size);
   ArrayInitialize(input_buffer, 0.0);

//--- Get indicator buffers
   double stoch_k_b[], stoch_d_b[];
   double adx_b[], di_plus_b[], di_minus_b[];
   long volume_b[];

   int stoch_k_count = CopyBuffer(g_stoch_handle, 0, 0, window, stoch_k_b);
   int stoch_d_count = CopyBuffer(g_stoch_handle, 1, 0, window, stoch_d_b);
   int adx_count = CopyBuffer(g_adx_handle, 0, 0, window, adx_b);
   int di_plus_count = CopyBuffer(g_adx_handle, 1, 0, window, di_plus_b);
   int di_minus_count = CopyBuffer(g_adx_handle, 2, 0, window, di_minus_b);
   int volume_count = CopyTickVolume(_Symbol, _Period, 0, window + 10, volume_b);

   if(stoch_k_count < window || stoch_d_count < window ||
      adx_count < window || di_plus_count < window ||
      di_minus_count < window || volume_count < window + 10)
     {
      Print("[WARNING] Not enough indicator data yet");
      return false;
     }

//--- Set arrays as series (newest = 0)
   ArraySetAsSeries(stoch_k_b, true);
   ArraySetAsSeries(stoch_d_b, true);
   ArraySetAsSeries(adx_b, true);
   ArraySetAsSeries(di_plus_b, true);
   ArraySetAsSeries(di_minus_b, true);
   ArraySetAsSeries(volume_b, true);

//--- Fill input buffer
   // Loop through the lookback window
   // i=0 is the most recent bar (current), i=window-1 is the oldest
   for(int i = 0; i < window; i++)
     {
      int offset = i * 6;  // 6 features per bar (v3)

      // Features 1-5: Technical indicators
      input_buffer[offset + 0] = (float)stoch_k_b[i];    // feat_stoch_main
      input_buffer[offset + 1] = (float)stoch_d_b[i];    // feat_stoch_signal
      input_buffer[offset + 2] = (float)adx_b[i];        // feat_adx
      input_buffer[offset + 3] = (float)di_plus_b[i];    // feat_pdi
      input_buffer[offset + 4] = (float)di_minus_b[i];   // feat_mdi

      // Feature 6: Volume Gate (ratio vs 10-bar average)
      // CRITICAL: Must match Python's rolling(window=10).mean()
      // Python calculates: avg of [i-9, i-8, ..., i-1, i] (10 bars ending at current)
      // Since arrays are SetAsSeries=true: volume_b[0]=newest, volume_b[19]=oldest
      // For bar at index i, we need average of indices [i, i+1, ..., i+9]
      double vol_avg = 0.0;
      int vol_count = 0;

      // Average the 10 bars starting from current position i
      for(int j = i; j < MathMin(i + 10, ArraySize(volume_b)); j++)
        {
         vol_avg += (double)volume_b[j];
         vol_count++;
        }

      if(vol_count > 0)
         vol_avg /= (double)vol_count;

      double volume_gate = (vol_avg > 0) ? (double)volume_b[i] / vol_avg : 1.0;
      input_buffer[offset + 5] = (float)volume_gate;        // feat_volume_gate
     }

   if(ArraySize(input_buffer) != total_size)
     {
      Print("[ERROR] Buffer size mismatch: expected ", total_size, ", got ", ArraySize(input_buffer));
      return false;
     }

   return true;
  }

//+------------------------------------------------------------------+
//| Check if within trading session                                   |
//+------------------------------------------------------------------+
bool IsWithinSession()
  {
   MqlDateTime dt;
   TimeCurrent(dt);

   if(InpStartHour <= InpEndHour)
     {
      return (dt.hour >= InpStartHour && dt.hour < InpEndHour);
     }
   else
     {
      return (dt.hour >= InpStartHour || dt.hour < InpEndHour);
     }
  }

//+------------------------------------------------------------------+
//| Check if position exists                                          |
//+------------------------------------------------------------------+
bool HasPosition(ENUM_POSITION_TYPE type)
  {
   for(int i = PositionsTotal() - 1; i >= 0; i--)
     {
      if(PositionGetSymbol(i) == _Symbol &&
         PositionGetInteger(POSITION_MAGIC) == InpMagic &&
         PositionGetInteger(POSITION_TYPE) == type)
        {
         return true;
        }
     }
   return false;
  }

//+------------------------------------------------------------------+
//| Update information panel                                          |
//+------------------------------------------------------------------+
void UpdatePanel()
  {
   string panel = "\n\n";

   panel += "SESSION: " + IntegerToString(InpStartHour, 2, '0') + ":00-" +
            IntegerToString(InpEndHour, 2, '0') + ":00";
   panel += IsWithinSession() ? " [ACTIVE]\n" : " [CLOSED]\n";

   string mode = (InpInferSeconds == 0) ? "NEW BAR" : IntegerToString(InpInferSeconds) + "s";
   panel += "MODE: " + mode + " | Inferences: " + IntegerToString(g_inference_count) + "\n";

//--- Indicators
   double adx[], di_plus[], di_minus[], stoch_k[], stoch_d[], atr[];

   CopyBuffer(g_adx_handle, 0, 0, 1, adx);
   CopyBuffer(g_adx_handle, 1, 0, 1, di_plus);
   CopyBuffer(g_adx_handle, 2, 0, 1, di_minus);
   CopyBuffer(g_stoch_handle, 0, 0, 1, stoch_k);
   CopyBuffer(g_stoch_handle, 1, 0, 1, stoch_d);
   CopyBuffer(g_atr_handle, 0, 0, 1, atr);

   panel += StringRepeat("-", 52) + "\n";
   panel += "FEATURES (for Neural Network)\n";
   panel += StringRepeat("-", 52) + "\n";
   panel += "   Stoch K/D: " + DoubleToString(stoch_k[0], 2) + " / " + DoubleToString(stoch_d[0], 2) + "\n";
   panel += "   ADX: " + DoubleToString(adx[0], 2) + " | DI+: " + DoubleToString(di_plus[0], 2) +
            " | DI-: " + DoubleToString(di_minus[0], 2) + "\n";

   // Volume Gate
   long vol_buf[];
   CopyTickVolume(_Symbol, _Period, 0, 11, vol_buf);
   ArraySetAsSeries(vol_buf, true);
   double vol_avg = 0;
   for(int i = 1; i <= 10; i++)
      vol_avg += (double)vol_buf[i];
   vol_avg /= 10.0;
   double vol_ratio = (vol_avg > 0) ? (double)vol_buf[0] / vol_avg : 1.0;
   panel += "   Volume Gate: " + DoubleToString(vol_ratio, 2) + "x avg\n";

   panel += StringRepeat("-", 52) + "\n";
   panel += "ATR RISK MANAGEMENT\n";
   panel += StringRepeat("-", 52) + "\n";
   panel += "   ATR(" + IntegerToString(InpATRPeriod) + "): " + DoubleToString(atr[0], _Digits) + "\n";
   panel += "   SL: " + DoubleToString(InpATRMultiplierSL, 1) + " x ATR = " +
            DoubleToString(atr[0] * InpATRMultiplierSL, _Digits) + "\n";
   panel += "   TP: " + DoubleToString(InpATRMultiplierTP, 1) + " x ATR = " +
            DoubleToString(atr[0] * InpATRMultiplierTP, _Digits) + "\n";

//--- AI Prediction
   panel += StringRepeat("-", 52) + "\n";
   panel += "NEURAL NETWORK DECISION\n";
   panel += StringRepeat("-", 52) + "\n";

   if(g_last_prediction >= 0)
     {
      string signal_text = "";
      if(g_last_prediction == 0)
         signal_text = "HOLD";
      else if(g_last_prediction == 1)
         signal_text = "BUY";
      else
         signal_text = "SELL";

      panel += "   Signal: " + signal_text + "\n";
      panel += "   Confidence (Min: " + DoubleToString(InpMinConf * 100, 1) + "%)\n";
      panel += "   - H: " + DoubleToString(g_last_probas[0] * 100, 1) + "% | " +
               "B: " + DoubleToString(g_last_probas[1] * 100, 1) + "% | " +
               "S: " + DoubleToString(g_last_probas[2] * 100, 1) + "%\n";
     }
   else
     {
      panel += "   Waiting for first inference...\n";
     }

//--- Position Info
   double pnl = 0;
   string pos_info = "";

   for(int i = PositionsTotal() - 1; i >= 0; i--)
     {
      if(PositionGetSymbol(i) == _Symbol && PositionGetInteger(POSITION_MAGIC) == InpMagic)
        {
         pnl = PositionGetDouble(POSITION_PROFIT);
         ENUM_POSITION_TYPE type = (ENUM_POSITION_TYPE)PositionGetInteger(POSITION_TYPE);
         pos_info = (type == POSITION_TYPE_BUY) ? "BUY" : "SELL";
         break;
        }
     }

   if(pos_info != "")
     {
      panel += StringRepeat("-", 52) + "\n";
      panel += "ACTIVE POSITION: " + pos_info + "\n";
      string pnl_str = (pnl >= 0) ? "+" : "";
      panel += "   P&L: " + pnl_str + DoubleToString(pnl, 2) + " " + AccountInfoString(ACCOUNT_CURRENCY) + "\n";
      panel += StringRepeat("-", 52) + "\n";
     }

   Comment(panel);
  }

//+------------------------------------------------------------------+
//| Print configuration                                               |
//+------------------------------------------------------------------+
void PrintConfiguration()
  {
   Print("\n=== SYMBOL INFORMATION ===");
   Print("Symbol: ", _Symbol);
   Print("Timeframe: ", EnumToString(_Period));
   Print("Digits: ", _Digits);
   Print("Point: ", DoubleToString(_Point, _Digits));

   Print("\n=== INDICATOR PARAMETERS ===");
   Print("Stochastic: (", InpStochK, ",", InpStochD, ")");
   Print("ADX: Period=", InpADXPeriod);
   Print("ATR: Period=", InpATRPeriod);

   Print("\n=== STRATEGY ===");
   Print("Type: NN-Driven (Neural Network decides everything)");
   Print("Features: 6 (stoch_k, stoch_d, adx, pdi, mdi, volume_gate)");
   Print("Exit: ATR-based SL/TP (SL=", InpATRMultiplierSL, "xATR, TP=", InpATRMultiplierTP, "xATR)");
   Print("No EMA gate - removed in v3");
   Print("No manual gates - NN makes all decisions");

   string mode = (InpInferSeconds == 0) ? "New bar only" : StringFormat("Every %d seconds", InpInferSeconds);
   Print("\nInference mode: ", mode);
  }

//+------------------------------------------------------------------+
//| Helper: Repeat string                                             |
//+------------------------------------------------------------------+
string StringRepeat(string str, int count)
  {
   string result = "";
   for(int i = 0; i < count; i++)
      result += str;
   return result;
  }

//+------------------------------------------------------------------+
