//+------------------------------------------------------------------+
//|                                           EA_SGRADT70_ONNX.mq5   |
//|                          SGRADT 7.0 - AI Strategy (6 Features with DI+/DI-)|
//+------------------------------------------------------------------+
#property copyright "SGRADT 7.0"
#property version   "7.00"
#property strict

#include <Trade\Trade.mqh>

//--- Input Parameters

//=== AI Model Configuration ===
input group "======== AI MODEL ========"
input string InpModelName = "USTEC_M5_SGRADT70_ema9.onnx";  // Model filename
input double InpMinConf   = 0.55;      // Minimum confidence (0.0-1.0)
input int    InpWindowSize = 10;       // Window size (must match training)
input int    InpFeaturesPerBar = 6;    // Features per bar (6 = body, range, stoch_k, stoch_d, adx, di_diff)

//=== Inference Timing ===
input group "======== INFERENCE ========"
input int    InpInferSeconds = 0;        // Inference frequency (0 = new bar only)
input bool   InpOneTradePerBar = true;   // Limit to 1 trade per bar
input bool   InpReverseInfer = false;    // BUY is SELL and SELL is BUY
input string InpTrainParams = "";        // ONNX train params

//=== Trading Session ===
input group "======== SESSION ========"
input int InpStartHour = 0;            // Session start hour (0-23)
input int InpEndHour   = 24;           // Session end hour (0-24)

//=== Indicator Parameters ===
input group "======== STOCHASTIC ========"
input int    InpStochK          = 5;      // Stochastic K period
input int    InpStochD          = 3;      // Stochastic D smoothing
input double InpStochOversold   = 30.0;   // Oversold level
input double InpStochOverbought = 70.0;   // Overbought level

input group "======== ADX ========"
input int    InpADXPeriod = 8;      // ADX period
input double InpADXLimit  = 24.0;   // ADX trend threshold
input bool   InpUseADXGate = true;  // Use ADX gate

//=== Forecast Exit Configuration ===
input group "======== FORECAST EXIT ========"
input int    InpForecastHorizon = 2;    // Bars to wait for forecast fulfillment (0=disabled)
input bool   InpUseForecastExit = true;  // Close if forecast fulfilled at horizon

//=== Candle Direction Gate Configuration ===
input group "======== CANDLE DIRECTION GATE ========"
input bool   InpUseH1CandleGate = true;  // Use H1 candle direction

//=== Risk Management ===
input group "======== RISK ========"
input double InpLot        = 1.0;   // Lot size
input int    InpMagic      = 7070;  // Magic number
input int    InpATRPeriod  = 6;    // ATR Period for SL/TP
input double InpATRMultSL  = 1.5;   // ATR Multiplier for Stop Loss (0=disabled)
input double InpATRMultTP  = 2.0;   // ATR Multiplier for Take Profit (0=disabled)

//=== Display ===
input group "======== DISPLAY ========"
input bool InpShowPanel = true;     // Show information panel

//--- Global Variables
CTrade trade;
long   g_onnx_handle = INVALID_HANDLE;
int    g_adx_handle  = INVALID_HANDLE;
int    g_stoch_handle = INVALID_HANDLE;
int    g_atr_handle  = INVALID_HANDLE;  // ATR handle

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
   Print("    SGRADT 7.0 - AI STRATEGY (6 Features with DI+/DI-)");
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

//--- Set output shapes
// Output 0: label [1]
   long out_shape_label[] = {1};
   OnnxSetOutputShape(g_onnx_handle, 0, out_shape_label);

// Output 1: probabilities [1, 3] for classes [HOLD, BUY, SELL]
   long out_shape_probs[] = {1, 3};
   OnnxSetOutputShape(g_onnx_handle, 1, out_shape_probs);

   Print("[OK] Output shapes set: label[1], probabilities[1,3]");

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

   // Create ATR handle
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

bool CheckH1CandleGate(ENUM_POSITION_TYPE position_type)
{
   if(!InpUseH1CandleGate)
      return true;
      
   double open[], close[];
   int copied = CopyOpen(_Symbol, PERIOD_H1, 0, 2, open);
   if(copied != 2)
     {
      Print("[ERROR] CopyOpen failed: expected ", 2, ", got ", copied);
      return false;
     }
   copied = CopyClose(_Symbol, PERIOD_H1, 0, 2, close);
   if(copied != 2)
     {
      Print("[ERROR] CopyClose failed: expected ", 2, ", got ", copied);
      return false;
     }
   
   if(position_type == POSITION_TYPE_BUY)
   {
      if(open[1] < close[1])
         return true;
   }
   else if(position_type == POSITION_TYPE_SELL)
   {
      if(open[1] > close[1])
         return true;
   }
   
   return false;
}

bool CheckADXGate(ENUM_POSITION_TYPE position_type)
{
   if(!InpUseADXGate)
      return true;

   double adx_buf[];
   double diplus_buf[];
   double diminus_buf[];
   
   if(CopyBuffer(g_adx_handle, 0, 0, 2, adx_buf) < 2 ||
      CopyBuffer(g_adx_handle, 1, 0, 2, diplus_buf) < 2 ||
      CopyBuffer(g_adx_handle, 2, 0, 2, diminus_buf) < 2)
      return false;
      
   double adx = adx_buf[0];
   double diplus = diplus_buf[0];
   double diminus = diminus_buf[0];
   
   if(position_type == POSITION_TYPE_BUY)
   {
      if(adx > InpADXLimit && diplus > diminus)
         return true;
   }
   else if(position_type == POSITION_TYPE_SELL)
   {
      if(adx > InpADXLimit && diminus > diplus)
         return true;
   }
      
   return false;
}

//+------------------------------------------------------------------+
//| Check forecast-based exit                                         |
//| Closes position if price has moved favorably at forecast horizon  |
//+------------------------------------------------------------------+
void CheckForecastExit()
  {
   if(!InpUseForecastExit || InpForecastHorizon <= 0) return;
   
   for(int i = PositionsTotal() - 1; i >= 0; i--)
     {
      ulong ticket = PositionGetTicket(i);
      if(ticket == 0) continue;
      
      if(PositionGetSymbol(i) != _Symbol || 
         PositionGetInteger(POSITION_MAGIC) != InpMagic)
         continue;
      
      // Get position open time and calculate bars since entry
      datetime open_time = (datetime)PositionGetInteger(POSITION_TIME);
      int entry_bar = iBarShift(_Symbol, _Period, open_time, false);
      int current_bar = 0; // Current bar is always 0 in timeseries indexing
      int bars_elapsed = entry_bar - current_bar;
      
      // Check if we've reached the forecast horizon
      if(bars_elapsed < InpForecastHorizon) continue;
      
      ENUM_POSITION_TYPE type = (ENUM_POSITION_TYPE)PositionGetInteger(POSITION_TYPE);
      double open_price = PositionGetDouble(POSITION_PRICE_OPEN);
      double current_price = (type == POSITION_TYPE_BUY) ? 
                             SymbolInfoDouble(_Symbol, SYMBOL_BID) : 
                             SymbolInfoDouble(_Symbol, SYMBOL_ASK);
      
      bool forecast_fulfilled = false;
      
      if(type == POSITION_TYPE_BUY)
        {
         // For BUY: forecast was for price to go up
         if(current_price > open_price)
           {
            forecast_fulfilled = true;
            if(trade.PositionClose(ticket))
               Print("[EXIT BUY] Forecast fulfilled at horizon ", InpForecastHorizon, 
                     " | Entry: ", DoubleToString(open_price, _Digits),
                     " | Current: ", DoubleToString(current_price, _Digits),
                     " | Profit: ", DoubleToString(current_price - open_price, _Digits));
           }
        }
      else if(type == POSITION_TYPE_SELL)
        {
         // For SELL: forecast was for price to go down
         if(current_price < open_price)
           {
            forecast_fulfilled = true;
            if(trade.PositionClose(ticket))
               Print("[EXIT SELL] Forecast fulfilled at horizon ", InpForecastHorizon,
                     " | Entry: ", DoubleToString(open_price, _Digits),
                     " | Current: ", DoubleToString(current_price, _Digits),
                     " | Profit: ", DoubleToString(open_price - current_price, _Digits));
           }
        }
     }
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

//--- Check for forecast-based exit
   if(InpUseForecastExit) CheckForecastExit();

//--- Determine if should run inference
   bool should_infer = false;
   datetime current_bar = iTime(_Symbol, _Period, 0);

   if(InpInferSeconds == 0)
     {
      // New bar only mode
      if(current_bar != g_last_bar_time)
        {
         should_infer = true;
         g_last_bar_time = current_bar;
        }
     }
   else
     {
      // Time-based mode
      if(TimeCurrent() >= g_last_inference_time + InpInferSeconds)
        {
         should_infer = true;
         g_last_inference_time = TimeCurrent();
        }
     }

//--- Run inference
   if(should_infer)
     {
      RunInference();
      g_inference_count++;
     }

//--- Update panel
   if(InpShowPanel)
      UpdatePanel();
  }

//+------------------------------------------------------------------+
//| Run ONNX inference                                                |
//+------------------------------------------------------------------+
void RunInference()
  {
//--- Check if already traded this bar
   datetime current_bar = iTime(_Symbol, _Period, 0);
   if(InpOneTradePerBar && current_bar == g_last_trade_bar)
     {
      return;
     }

//--- Prepare input
   float input_buffer[];
   if(!PrepareInput(input_buffer))
     {
      Print("[WARNING] Cannot prepare input data - skipping inference");
      return;
     }

//--- Validate buffer before ONNX call
   int expected_size = InpWindowSize * InpFeaturesPerBar;
   if(ArraySize(input_buffer) != expected_size)
     {
      Print("[ERROR] Input buffer size incorrect: expected ", expected_size, ", got ", ArraySize(input_buffer));
      return;
     }

//--- Run inference
   float output_label[1];
   float output_proba[3];

   if(!OnnxRun(g_onnx_handle, ONNX_DEFAULT, input_buffer, output_label, output_proba))
     {
      Print("[ERROR] ONNX inference failed");
      Print("        Input buffer size: ", ArraySize(input_buffer));
      Print("        Expected: ", expected_size);
      Print("        GetLastError: ", GetLastError());
      return;
     }

//--- Validate output
   if(ArraySize(output_proba) != 3)
     {
      Print("[ERROR] Invalid output probabilities: expected 3, got ", ArraySize(output_proba));
      return;
     }

//--- Store results
   for(int i = 0; i < 3; i++)
      g_last_probas[i] = output_proba[i];

   g_last_prediction = (int)output_label[0];

//--- Get max probability
   double max_prob = g_last_probas[0];
   int predicted_class = 0;

   for(int i = 1; i < 3; i++)
     {
      if(g_last_probas[i] > max_prob)
        {
         max_prob = g_last_probas[i];
         predicted_class = i;
        }
     }

   if(InpReverseInfer)
     {
      if(predicted_class == 1)
         predicted_class = 2;
      else if(predicted_class == 2)
         predicted_class = 1;
      PrintFormat("Reversed Inference Active: %.1f", predicted_class);
     }

//--- Show inference values for debug purposes
   double adx_buf[], stoch_buf[], stochd_buf[];
   CopyBuffer(g_adx_handle, 0, 0, 1, adx_buf);
   CopyBuffer(g_stoch_handle, 0, 0, 1, stoch_buf);
   CopyBuffer(g_stoch_handle, 1, 0, 1, stochd_buf);
   string class_name = "HOLD";
   if(predicted_class == 1) class_name = "BUY";
   else if(predicted_class == 2) class_name = "SELL";
   
   PrintFormat("%s | Inference: %.2f (%s) | ADX: %.1f | Stoch: %.1f/%.1f",
               _Symbol,
               max_prob,
               class_name,
               adx_buf[0],
               stoch_buf[0],
               stochd_buf[0]);

//--- Check confidence
   if(max_prob < InpMinConf)
     {
      return;
     }

   // Get current ATR value for SL/TP calculation
   double atr_val[], current_atr = 0;
   if(CopyBuffer(g_atr_handle, 0, 0, 1, atr_val) == 1)
      current_atr = atr_val[0];

//--- Execute trade
   if(predicted_class == 1)    // BUY
     {
      bool has_buy_pos = HasPosition(POSITION_TYPE_BUY);
      
      if(has_buy_pos)
        {
         Print("[SKIP BUY] Already have BUY position");
        }
      else
        {
         double ask = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
         double sl  = (InpATRMultSL > 0 && current_atr > 0) ? ask - current_atr * InpATRMultSL : 0;
         double tp  = (InpATRMultTP > 0 && current_atr > 0) ? ask + current_atr * InpATRMultTP : 0;
         
         if(!CheckADXGate(POSITION_TYPE_BUY))
         {
            Print("[SKIP BUY] ADX gate is not satisfied");
            return;
         }

         if(!CheckH1CandleGate(POSITION_TYPE_BUY))
         {
            Print("[SKIP BUY] H1 candle gate is not satisfied");
            return;
         }

         if(trade.Buy(InpLot, _Symbol, 0, sl, tp, "SGRADT70 BUY @" + DoubleToString(max_prob, 2) + "%"))
           {
            Print("[BUY] Order opened | Confidence: ", DoubleToString(max_prob * 100, 2), 
                  "% | SL: ", DoubleToString(sl, _Digits), " | TP: ", DoubleToString(tp, _Digits));
            g_last_trade_bar = current_bar;
           }
        }
     }
   else if(predicted_class == 2)    // SELL
     {
      bool has_sell_pos = HasPosition(POSITION_TYPE_SELL);
      
      if(has_sell_pos)
        {
         Print("[SKIP SELL] Already have SELL position");
        }
      else
        {
         double bid = SymbolInfoDouble(_Symbol, SYMBOL_BID);
         double sl  = (InpATRMultSL > 0 && current_atr > 0) ? bid + current_atr * InpATRMultSL : 0;
         double tp  = (InpATRMultTP > 0 && current_atr > 0) ? bid - current_atr * InpATRMultTP : 0;
         
         if(!CheckADXGate(POSITION_TYPE_SELL))
         {
            Print("[SKIP SELL] ADX gate is not satisfied");
            return;
         }

         if(!CheckH1CandleGate(POSITION_TYPE_SELL))
         {
            Print("[SKIP SELL] H1 candle gate is not satisfied");
            return;
         }

         if(trade.Sell(InpLot, _Symbol, 0, sl, tp, "SGRADT70 SELL @" + DoubleToString(max_prob, 2) + "%"))
           {
            Print("[SELL] Order opened | Confidence: ", DoubleToString(max_prob * 100, 2), 
                  "% | SL: ", DoubleToString(sl, _Digits), " | TP: ", DoubleToString(tp, _Digits));
            g_last_trade_bar = current_bar;
           }
        }
     }
  }

//+------------------------------------------------------------------+
//| Prepare input data for ONNX model                                 |
//+------------------------------------------------------------------+
bool PrepareInput(float &input_buffer[])
  {
   int window = InpWindowSize;
   int total_size = window * InpFeaturesPerBar;

   ArrayFree(input_buffer);
   if(ArrayResize(input_buffer, total_size) != total_size)
     {
      Print("[ERROR] Cannot resize input buffer to ", total_size);
      return false;
     }
   ArrayInitialize(input_buffer, 0.0);

//--- Get price data
   double open[], high[], low[], close[];
   ArraySetAsSeries(open, true);
   ArraySetAsSeries(high, true);
   ArraySetAsSeries(low, true);
   ArraySetAsSeries(close, true);

   int copied;
   copied = CopyOpen(_Symbol, _Period, 0, window, open);
   if(copied != window)
     {
      Print("[ERROR] CopyOpen failed: expected ", window, ", got ", copied);
      return false;
     }

   copied = CopyHigh(_Symbol, _Period, 0, window, high);
   if(copied != window)
     {
      Print("[ERROR] CopyHigh failed: expected ", window, ", got ", copied);
      return false;
     }

   copied = CopyLow(_Symbol, _Period, 0, window, low);
   if(copied != window)
     {
      Print("[ERROR] CopyLow failed: expected ", window, ", got ", copied);
      return false;
     }

   copied = CopyClose(_Symbol, _Period, 0, window, close);
   if(copied != window)
     {
      Print("[ERROR] CopyClose failed: expected ", window, ", got ", copied);
      return false;
     }

//--- Get indicator data
   double adx_b[], plus_di_b[], minus_di_b[];
   double stoch_k_b[], stoch_d_b[];

   ArraySetAsSeries(adx_b, true);
   ArraySetAsSeries(plus_di_b, true);
   ArraySetAsSeries(minus_di_b, true);
   ArraySetAsSeries(stoch_k_b, true);
   ArraySetAsSeries(stoch_d_b, true);

   // ADX buffer 0 = ADX
   copied = CopyBuffer(g_adx_handle, 0, 0, window, adx_b);
   if(copied != window)
     {
      Print("[ERROR] CopyBuffer ADX failed: expected ", window, ", got ", copied);
      return false;
     }

   // ADX buffer 1 = DI+
   copied = CopyBuffer(g_adx_handle, 1, 0, window, plus_di_b);
   if(copied != window)
     {
      Print("[ERROR] CopyBuffer PLUS_DI failed: expected ", window, ", got ", copied);
      return false;
     }

   // ADX buffer 2 = DI-
   copied = CopyBuffer(g_adx_handle, 2, 0, window, minus_di_b);
   if(copied != window)
     {
      Print("[ERROR] CopyBuffer MINUS_DI failed: expected ", window, ", got ", copied);
      return false;
     }

   copied = CopyBuffer(g_stoch_handle, 0, 0, window, stoch_k_b);
   if(copied != window)
     {
      Print("[ERROR] CopyBuffer Stoch K failed: expected ", window, ", got ", copied);
      return false;
     }

   copied = CopyBuffer(g_stoch_handle, 1, 0, window, stoch_d_b);
   if(copied != window)
     {
      Print("[ERROR] CopyBuffer Stoch D failed: expected ", window, ", got ", copied);
      return false;
     }

//--- Fill buffer (Order: body, range, stoch_k, stoch_d, adx, di_diff)
   for(int i = 0; i < window; i++)
     {
      int offset = i * InpFeaturesPerBar;

      input_buffer[offset + 0] = (float)(close[i] - open[i]);        // body
      input_buffer[offset + 1] = (float)(high[i] - low[i]);          // range
      input_buffer[offset + 2] = (float)stoch_k_b[i];                // stoch_main
      input_buffer[offset + 3] = (float)stoch_d_b[i];                // stoch_signal
      input_buffer[offset + 4] = (float)adx_b[i];                    // adx
      input_buffer[offset + 5] = (float)(plus_di_b[i] - minus_di_b[i]); // di_diff (+ for BUY, - for SELL)
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
   double adx[], stoch_k[], stoch_d[], atr[];

   CopyBuffer(g_adx_handle, 0, 0, 1, adx);
   CopyBuffer(g_stoch_handle, 0, 0, 1, stoch_k);
   CopyBuffer(g_stoch_handle, 1, 0, 1, stoch_d);
   CopyBuffer(g_atr_handle, 0, 0, 1, atr);

   panel += StringRepeat("=", 25) + "\n";
   panel += "ADX (Period: " + IntegerToString(InpADXPeriod) + ")\n";
   panel += StringRepeat("=", 25) + "\n";
   panel += "   ADX: " + DoubleToString(adx[0], 2);
   panel += (adx[0] > InpADXLimit) ? " [TRENDING]\n" : " [RANGING]\n";

   panel += StringRepeat("=", 25) + "\n";
   panel += "STOCHASTIC (" + IntegerToString(InpStochK) + "," + IntegerToString(InpStochD) + ")\n";
   panel += StringRepeat("=", 25) + "\n";
   panel += "   K: " + DoubleToString(stoch_k[0], 2) + " | " + " D: " + DoubleToString(stoch_d[0], 2) + "\n";

   string zone = "";
   if(stoch_k[0] <= InpStochOversold)
      zone = "OVERSOLD";
   else if(stoch_k[0] >= InpStochOverbought)
      zone = "OVERBOUGHT";
   else
      zone = "NEUTRAL";
   panel += "   Zone: " + zone + "\n";

//--- AI Prediction
   panel += StringRepeat("=", 25) + "\n";
   panel += "AI PREDICTION\n";
   panel += StringRepeat("=", 25) + "\n";

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

      panel += "   Confidence (Min: " + DoubleToString(InpMinConf * 100, 1) + ")\n";
      panel += "   - H: " + DoubleToString(g_last_probas[0] * 100, 1) + "% | " +
               "B: " + DoubleToString(g_last_probas[1] * 100, 1) + "% | " +
               "S: " + DoubleToString(g_last_probas[2] * 100, 1) + "%\n";

     }
   else
     {
      panel += "   Waiting for first inference...\n";
     }

//--- Risk Settings
   panel += StringRepeat("=", 25) + "\n";
   panel += "RISK SETTINGS\n";
   panel += StringRepeat("=", 25) + "\n";
   panel += "   Lot: " + DoubleToString(InpLot, 2) + "\n";
   panel += "   ATR(" + IntegerToString(InpATRPeriod) + "): " + DoubleToString(atr[0], _Digits) + "\n";
   panel += "   SL: " + DoubleToString(InpATRMultSL, 1) + "xATR | TP: " + DoubleToString(InpATRMultTP, 1) + "xATR\n";
   
   // Forecast exit info
   if(InpUseForecastExit && InpForecastHorizon > 0)
     {
      panel += "   Exit: Forecast@" + IntegerToString(InpForecastHorizon) + " bars\n";
     }
   else
     {
      panel += "   Exit: SL/TP Only\n";
     }

//--- Position Info
   double pnl = 0;
   string pos_info = "";
   int bars_held = 0;

   for(int i = PositionsTotal() - 1; i >= 0; i--)
     {
      if(PositionGetSymbol(i) == _Symbol && PositionGetInteger(POSITION_MAGIC) == InpMagic)
        {
         pnl = PositionGetDouble(POSITION_PROFIT);
         ENUM_POSITION_TYPE type = (ENUM_POSITION_TYPE)PositionGetInteger(POSITION_TYPE);
         pos_info = (type == POSITION_TYPE_BUY) ? "BUY" : "SELL";
         
         datetime open_time = (datetime)PositionGetInteger(POSITION_TIME);
         int entry_bar = iBarShift(_Symbol, _Period, open_time, false);
         bars_held = entry_bar;
         break;
        }
     }

   if(pos_info != "")
     {
      panel += StringRepeat("=", 25) + "\n";
      panel += "ACTIVE POSITION: " + pos_info + "\n";
      string pnl_str = (pnl >= 0) ? "+" : "";
      panel += "   P&L: " + pnl_str + DoubleToString(pnl, 2) + " " + AccountInfoString(ACCOUNT_CURRENCY) + "\n";
      if(InpForecastHorizon > 0)
         panel += "   Bars: " + IntegerToString(bars_held) + "/" + IntegerToString(InpForecastHorizon) + "\n";
      panel += StringRepeat("=", 25) + "\n";
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
   Print("Stochastic: (", InpStochK, ",", InpStochD, ") [", InpStochOversold, "/", InpStochOverbought, "]");
   Print("ADX: Period=", InpADXPeriod, ", Limit=", InpADXLimit);
   Print("ATR: Period=", InpATRPeriod);

   Print("\n=== STRATEGY ===");
   Print("Entry: AI Inference (ADX + Stochastic + DI features)");
   if(InpUseForecastExit && InpForecastHorizon > 0)
      Print("Exit: Forecast fulfillment at ", InpForecastHorizon, " bars");
   else
      Print("Exit: SL/TP Only");
   Print("SL/TP: ATR(", InpATRPeriod, ") x ", InpATRMultSL, " / x ", InpATRMultTP);

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