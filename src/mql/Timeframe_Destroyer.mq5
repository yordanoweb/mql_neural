//+------------------------------------------------------------------+
//|                                       Timeframe_Destroyer.mq5    |
//|                                  Timeframe Entry Strategy         |
//+------------------------------------------------------------------+
#property strict

#include <Trade\Trade.mqh>

#resource "\\Files\\xauusd_m15_w20_f8_pct0.5_h1-3.onnx" as uchar ExtModel[]

//--- INPUTS
input group "AI Config"
input int        InpWindowSize = 20;           // Window size (must match --window from training)
input float      InpMinConf    = 0.55;         // Minimum confidence threshold
input int        InpStartHour  = 1;            // Start hour of trading window
input int        InpEndHour    = 3;            // End hour of trading window
input double     InpTargetPct  = 0.5;          // Target percentage move (for display only)
input string     InpModelFile  = "xauusd_m15_w20_f8_pct0.5_h1-3.onnx";   // ONNX File (embed and recompile for backtest)

input group "Risk Management"
input double     InpLot        = 1.0;          // Lot size
input int        InpMagic      = 88122188;     // Magic number
input int        InpATRPeriod  = 14;           // ATR Period
input double     InpATRMultSL  = 1.5;          // ATR Multiplier for SL
input double     InpATRMultTP  = 1.5;          // ATR Multiplier for TP

input group "Trade Management"
input bool       InpOneTradePerWindow = true;  // Allow only one trade per time window

input group "Debug"
input bool       InpDebugPrint = false;        // Print debug info

//--- GLOBAL VARIABLES
long     onnx_handle = INVALID_HANDLE;
CTrade   m_trade;
const int FEATURES = 2;  // Only body and range
datetime last_trade_time = 0;
string program_name = MQLInfoString(MQL_PROGRAM_NAME);

int OnInit()
{
   // Load ONNX model
   if(MQLInfoInteger(MQL_TESTER)) {
      Print("Running in Strategy Tester - Loading embedded model");
      onnx_handle = OnnxCreateFromBuffer(ExtModel, ONNX_DEFAULT);
   } else {
      Print("Running Live/Demo | Model: ", InpModelFile);
      onnx_handle = OnnxCreate(InpModelFile, ONNX_DEFAULT);
   }
   
   if(onnx_handle == INVALID_HANDLE) 
   {
      Print("Failed to load ONNX model");
      return(INIT_FAILED);
   }

   // Set input shape: window * 2 features
   int input_size = InpWindowSize * FEATURES;
   long input_shape[] = {1, input_size};
   if(!OnnxSetInputShape(onnx_handle, 0, input_shape)) 
   {
      Print("Failed to set input shape [1, ", input_size, "]");
      return(INIT_FAILED);
   }

   // Set output shapes for 3-class classification
   long out_shape_label[] = {1};
   OnnxSetOutputShape(onnx_handle, 0, out_shape_label);
   long out_shape_probs[] = {1, 3};  // 3 classes: no_trade, long, short
   OnnxSetOutputShape(onnx_handle, 1, out_shape_probs);

   m_trade.SetExpertMagicNumber(InpMagic);
   
   Print("=== TIMEFRAME DESTROYER INITIALIZED ===");
   Print("Trading Window: ", InpStartHour, ":00 - ", InpEndHour, ":00");
   Print("Target Move: ±", DoubleToString(InpTargetPct, 2), "%");
   Print("Window Size: ", InpWindowSize, " | Features: ", FEATURES);
   Print("Input Shape: [1, ", input_size, "]");
   
   return(INIT_SUCCEEDED);
}

void OnDeinit(const int reason) 
{ 
   if(onnx_handle != INVALID_HANDLE) 
      OnnxRelease(onnx_handle); 
}

void OnTick()
{
   // 1. TIME FILTER - Only operate within specified hours
   MqlDateTime dt;
   TimeCurrent(dt); 
   bool valid_time = false;
   
   // Handle time window (including wrap-around midnight)
   if(InpStartHour <= InpEndHour)
      valid_time = (dt.hour >= InpStartHour && dt.hour < InpEndHour);
   else  // Wrap around midnight (e.g., 22-2)
      valid_time = (dt.hour >= InpStartHour || dt.hour < InpEndHour);

   // 2. BAR CONTROL - Only run on new bar
   static datetime last_bar = 0;
   datetime current_bar = iTime(_Symbol, _Period, 0);
   if(current_bar == last_bar) return;
   last_bar = current_bar;

   // 3. ONE TRADE PER WINDOW CHECK
   if(InpOneTradePerWindow && valid_time)
   {
      // Check if we already traded in this time window today
      MqlDateTime last_trade_dt;
      TimeToStruct(last_trade_time, last_trade_dt);
      
      if(last_trade_dt.year == dt.year && 
         last_trade_dt.mon == dt.mon && 
         last_trade_dt.day == dt.day)
      {
         // Already traded today in this window
         Comment("AI | Status: Already traded today in window ",
                 InpStartHour, ":00-", InpEndHour, ":00");
         return;
      }
   }

   // 4. LOAD PRICE DATA
   double close[], open[], high[], low[];
   ArraySetAsSeries(close, true); 
   ArraySetAsSeries(open, true);
   ArraySetAsSeries(high, true);  
   ArraySetAsSeries(low, true);

   if(CopyClose(_Symbol, _Period, 0, InpWindowSize, close) < InpWindowSize ||
      CopyOpen(_Symbol, _Period, 0, InpWindowSize, open) < InpWindowSize ||
      CopyHigh(_Symbol, _Period, 0, InpWindowSize, high) < InpWindowSize ||
      CopyLow(_Symbol, _Period, 0, InpWindowSize, low) < InpWindowSize) 
   {
      Print("Failed to copy price data");
      return;
   }

   // 5. PREPARE INPUT BUFFER
   // CRITICAL: Match Python training order exactly
   // Python loops: for i in range(window, len(df) - future):
   //     window_data = df[features].iloc[i-window:i].values.flatten()
   // This means: [oldest_candle_features, ..., newest_candle_features]
   
   float input_buffer[];
   ArrayResize(input_buffer, InpWindowSize * FEATURES);

   // MT5 arrays are in reverse (index 0 = newest)
   // So we need to reverse the order when building the buffer
   for(int i = 0; i < InpWindowSize; i++)
   {
      // Go from oldest to newest to match Python
      int mql_idx = InpWindowSize - 1 - i;  // oldest first
      
      // Features in same order as Python: body, range
      input_buffer[i * FEATURES + 0] = (float)(close[mql_idx] - open[mql_idx]);  // body
      input_buffer[i * FEATURES + 1] = (float)(high[mql_idx] - low[mql_idx]);     // range
   }
   
   // Debug: Print first and last candle features
   if(InpDebugPrint)
   {
      Print("=== INPUT DEBUG ===");
      Print("First candle (oldest): body=", input_buffer[0], " range=", input_buffer[1]);
      int last_idx = (InpWindowSize - 1) * FEATURES;
      Print("Last candle (newest): body=", input_buffer[last_idx], " range=", input_buffer[last_idx + 1]);
   }

   // 6. RUN INFERENCE
   long output_label[]; 
   float output_probs[];
   ArrayResize(output_label, 1); 
   ArrayResize(output_probs, 3);  // 3 classes
   
   if(!OnnxRun(onnx_handle, ONNX_NO_CONVERSION, input_buffer, output_label, output_probs)) 
   {
      Print("ONNX inference failed");
      return;
   }

   long  prediction = output_label[0];  // 0=no_trade, 1=long, 2=short
   float confidence = output_probs[prediction];

   if(InpDebugPrint)
   {
      Print("Prediction: ", prediction, " | Confidence: ", confidence);
      Print("Probs: [", output_probs[0], ", ", output_probs[1], ", ", output_probs[2], "]");
   }

   // 7. DISPLAY INFO
   string status_msg = "";
   if(!valid_time)
      status_msg = "OUTSIDE WINDOW";
   else if(PositionSelect(_Symbol))
      status_msg = "POSITION OPEN";
   else if(confidence < InpMinConf)
      status_msg = "LOW CONFIDENCE";
   else
   {
      if(prediction == 0) status_msg = "NO TRADE SIGNAL";
      else if(prediction == 1) status_msg = "LONG SIGNAL";
      else if(prediction == 2) status_msg = "SHORT SIGNAL";
   }

   Comment("=== TIMEFRAME DESTROYER ===",
           "\nWindow: ", InpStartHour, ":00 - ", InpEndHour, ":00",
           "\nCurrent Time: ", IntegerToString(dt.hour, 2, '0'), ":", IntegerToString(dt.min, 2, '0'),
           "\nStatus: ", status_msg,
           "\n---",
           "\nPrediction: ", (prediction == 0 ? "NO TRADE" : (prediction == 1 ? "LONG" : "SHORT")),
           "\nConfidence: ", DoubleToString(confidence * 100, 2), "%",
           "\nProbs [No/Long/Short]: [", 
           DoubleToString(output_probs[0]*100, 1), "%, ",
           DoubleToString(output_probs[1]*100, 1), "%, ",
           DoubleToString(output_probs[2]*100, 1), "%]");

   // 8. EXECUTE TRADES (only if prediction is 1 or 2)
   if(!PositionSelect(_Symbol) && valid_time && confidence >= InpMinConf && prediction != 0)
   {
      // Calculate SL and TP using ATR
      int atr_handle = iATR(_Symbol, _Period, InpATRPeriod);
      double atr_buffer[1];
      
      if(CopyBuffer(atr_handle, 0, 0, 1, atr_buffer) <= 0)
      {
         Print("Failed to get ATR value");
         return;
      }
      
      double sl_dist = atr_buffer[0] * InpATRMultSL;
      double tp_dist = atr_buffer[0] * InpATRMultTP;

      bool trade_executed = false;
      
      if(prediction == 1)  // LONG
      {
         double price = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
         double sl = price - sl_dist;
         double tp = price + tp_dist;
         
         if(m_trade.Buy(InpLot, _Symbol, price, sl, tp, 
            program_name + " LONG " + DoubleToString(confidence*100, 1) + "%"))
         {
            Print(">>> LONG EXECUTED | Conf: ", confidence*100, "% | SL: ", sl_dist/_Point, " pts | TP: ", tp_dist/_Point, " pts");
            trade_executed = true;
         }
         else
         {
            Print(">>> LONG FAILED | Error: ", GetLastError());
         }
      }
      else if(prediction == 2)  // SHORT
      {
         double price = SymbolInfoDouble(_Symbol, SYMBOL_BID);
         double sl = price + sl_dist;
         double tp = price - tp_dist;
         
         if(m_trade.Sell(InpLot, _Symbol, price, sl, tp, 
            program_name + " SHORT " + DoubleToString(confidence*100, 1) + "%"))
         {
            Print(">>> SHORT EXECUTED | Conf: ", confidence*100, "% | SL: ", sl_dist/_Point, " pts | TP: ", tp_dist/_Point, " pts");
            trade_executed = true;
         }
         else
         {
            Print(">>> SHORT FAILED | Error: ", GetLastError());
         }
      }
      
      if(trade_executed)
         last_trade_time = TimeCurrent();
   }
}
