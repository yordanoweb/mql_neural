//+------------------------------------------------------------------+
//|                                       Nasdaq_Destroyer_v2.mq5    |
//|                                  Timeframe Entry Strategy         |
//+------------------------------------------------------------------+
#property strict

#include <Trade\Trade.mqh>

#resource "\\Files\\xauusd_m15_w20_f8_pct0.5_h1-3.onnx" as uchar ExtModel[]

//--- INPUTS
input group "AI Config"
input float      InpMinConf    = 0.55;         // Minimum confidence threshold
input int        InpStartHour  = 1;            // Start hour of trading window
input int        InpEndHour    = 3;            // End hour of trading window
input double     InpTargetPct  = 0.5;          // Target percentage move (for display only)
input int        InpFuture     = 20;           // Number of bars to learn (match --window)
input string     InpModelFile  = "xauusd_m15_w20_f8_pct0.5_h1-3.onnx";   // ONNX File (embed and recompile for backtest)

input group "Risk Management"
input double     InpLot        = 1.0;          // Lot size
input int        InpMagic      = 88122188;     // Magic number
input int        InpATRPeriod  = 14;           // ATR Period
input double     InpATRMultSL  = 1.5;          // ATR Multiplier for SL
input double     InpATRMultTP  = 1.5;          // ATR Multiplier for TP

input group "Trade Management"
input bool       InpOneTradePerWindow = true;  // Allow only one trade per time window

//--- GLOBAL VARIABLES
long     onnx_handle = INVALID_HANDLE;
CTrade   m_trade;
const int WINDOW_SIZE = InpFuture;
const int FEATURES    = 2;  // Only body and range
datetime last_trade_time = 0;

int OnInit()
{
   // Load ONNX model
   if(MQLInfoInteger(MQL_TESTER)) {
      Print("Running in Strategy Tester");
      onnx_handle = OnnxCreateFromBuffer(ExtModel, ONNX_DEFAULT);
   } else {
      Print("Running Live/Demo | Model: ", InpModelFile);
      onnx_handle = OnnxCreate(InpModelFile, ONNX_DEFAULT);
   }

   // Set input shape: 20 candles * 2 features = 40
   long input_shape[] = {1, 40};
   if(!OnnxSetInputShape(onnx_handle, 0, input_shape)) 
   {
      Print("Failed to set input shape");
      return(INIT_FAILED);
   }

   // Set output shapes for 3-class classification
   long out_shape_label[] = {1};
   OnnxSetOutputShape(onnx_handle, 0, out_shape_label);
   long out_shape_probs[] = {1, 3};  // 3 classes: no_trade, long, short
   OnnxSetOutputShape(onnx_handle, 1, out_shape_probs);

   m_trade.SetExpertMagicNumber(InpMagic);
   
   Print("=== Nasdaq Destroyer v2 Initialized ===");
   Print("Trading Window: ", InpStartHour, ":00 - ", InpEndHour, ":00");
   Print("Target Move: ±", DoubleToString(InpTargetPct, 2), "%");
   Print("Window Size: ", WINDOW_SIZE, " | Features: ", FEATURES);
   
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

   if(CopyClose(_Symbol, _Period, 0, WINDOW_SIZE, close) < WINDOW_SIZE ||
      CopyOpen(_Symbol, _Period, 0, WINDOW_SIZE, open) < WINDOW_SIZE ||
      CopyHigh(_Symbol, _Period, 0, WINDOW_SIZE, high) < WINDOW_SIZE ||
      CopyLow(_Symbol, _Period, 0, WINDOW_SIZE, low) < WINDOW_SIZE) 
   {
      Print("Failed to copy price data");
      return;
   }

   // 5. PREPARE INPUT BUFFER (Only body and range features)
   float input_buffer[];
   ArrayResize(input_buffer, WINDOW_SIZE * FEATURES);

   for(int i = 0; i < WINDOW_SIZE; i++)
   {
      int mql_idx = WINDOW_SIZE - 1 - i;
      input_buffer[i * 2 + 0] = (float)(close[mql_idx] - open[mql_idx]);  // body
      input_buffer[i * 2 + 1] = (float)(high[mql_idx] - low[mql_idx]);    // range
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

   Comment("=== NASDAQ DESTROYER v2 ===",
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

   // 8. EXECUTE TRADES
   if(!PositionSelect(_Symbol) && valid_time && confidence >= InpMinConf)
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
            "AI_LONG " + DoubleToString(confidence*100, 1) + "%"))
         {
            Print("LONG executed | Conf: ", confidence*100, "% | Target: +", InpTargetPct, "%");
            trade_executed = true;
         }
      }
      else if(prediction == 2)  // SHORT
      {
         double price = SymbolInfoDouble(_Symbol, SYMBOL_BID);
         double sl = price + sl_dist;
         double tp = price - tp_dist;
         
         if(m_trade.Sell(InpLot, _Symbol, price, sl, tp, 
            "AI_SHORT " + DoubleToString(confidence*100, 1) + "%"))
         {
            Print("SHORT executed | Conf: ", confidence*100, "% | Target: -", InpTargetPct, "%");
            trade_executed = true;
         }
      }
      // prediction == 0 (no_trade) -> do nothing
      
      if(trade_executed)
         last_trade_time = TimeCurrent();
   }
}
