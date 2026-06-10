//+------------------------------------------------------------------+
//|                                              SimpleONNX_EA.mq5   |
//+------------------------------------------------------------------+
#property strict

#include <Trade\Trade.mqh>

//--- ENUMS
enum ENUM_LOGIC { LOGIC_NORMAL, LOGIC_MIRROR };

#resource "\\Files\\USTEC_M15_202202211330_202412312345.onnx" as uchar ExtModel[]

//--- INPUTS
input group "AI Configuration"
input string     InpModelFile  = "USTEC_M15_202202211330_202412312345.onnx";  // Dynamic model filename
input ENUM_LOGIC InpLogic      = LOGIC_MIRROR;
input float      InpMinConf    = 0.62;
input int        InpStartHour  = 9;
input int        InpEndHour    = 18;
input int        InpWindow     = 20;           // Must match training --window
input int        InpRSIPeriod  = 14;           // Must match training --rsi_period
input int        InpFeatATR    = 14;           // Must match training --atr_period
input bool       InpVerboseLogs = true;        // Print heartbeat and decision logs
input group "Risk Management"
input double     InpLot        = 1;
input int        InpMagic      = 123456;
input int        InpATR        = 6;            // ATR period for SL/TP only
input double     InpMultiplier = 1.5;

//--- GLOBAL VARIABLES
long     onnx_handle = INVALID_HANDLE;
CTrade   m_trade;
const int FEATURES    = 3;
int      rsi_handle   = INVALID_HANDLE;
int      feat_atr_handle = INVALID_HANDLE;
int      risk_atr_handle = INVALID_HANDLE;

void LogInfo(const string msg)
  {
   if(InpVerboseLogs)
      Print("[AI_EA] ", msg);
  }

string PredDirection(const long prediction)
  {
   return (prediction == 1 ? "UP" : "DOWN");
  }

string EntrySideFromPrediction(const long prediction)
  {
   bool should_sell = ((InpLogic == LOGIC_MIRROR && prediction == 1) || (InpLogic == LOGIC_NORMAL && prediction == 0));
   return (should_sell ? "SELL" : "BUY");
  }

void RunInferenceCycle(const string trigger)
  {
// 1. TRADING HOURS FILTER
   MqlDateTime dt;
   TimeCurrent(dt);
   bool valid_hours = (dt.hour >= InpStartHour && dt.hour < InpEndHour);

// 2. BAR CONTEXT (last closed bar)
   datetime current_bar = iTime(_Symbol, _Period, 1);
   if(current_bar <= 0)
      return;
   LogInfo(StringFormat("Heartbeat | trigger=%s | bar=%s | symbol=%s | tf=%s",
                        trigger, TimeToString(current_bar, TIME_DATE|TIME_MINUTES), _Symbol, EnumToString(_Period)));

// 3. DATA
   double close[], open[], high[], low[];
   ArraySetAsSeries(close, true);
   ArraySetAsSeries(open, true);
   ArraySetAsSeries(high, true);
   ArraySetAsSeries(low, true);

   if(CopyClose(_Symbol, _Period, 1, InpWindow, close) < InpWindow ||
      CopyOpen(_Symbol, _Period, 1, InpWindow, open) < InpWindow ||
      CopyHigh(_Symbol, _Period, 1, InpWindow, high) < InpWindow ||
      CopyLow(_Symbol, _Period, 1, InpWindow, low) < InpWindow)
      return;

// 4. INDICATORS
   double rsi_buffer[];
   ArraySetAsSeries(rsi_buffer, true);
   if(CopyBuffer(rsi_handle, 0, 1, InpWindow, rsi_buffer) < InpWindow)
      return;

   double feat_atr_buffer[];
   ArraySetAsSeries(feat_atr_buffer, true);
   if(CopyBuffer(feat_atr_handle, 0, 1, InpWindow, feat_atr_buffer) < InpWindow)
      return;

   double risk_atr_buffer[];
   ArraySetAsSeries(risk_atr_buffer, true);
   if(CopyBuffer(risk_atr_handle, 0, 1, 1, risk_atr_buffer) < 1)
      return;
   double current_atr = risk_atr_buffer[0];
   if(current_atr <= 0.0 || current_atr == EMPTY_VALUE)
      return;

// 5. INPUT BUFFER ALIGNED TO train_onnx_from_csv.py:
// feat_body=(close-open)/ATR, feat_range=(high-low)/ATR, feat_rsi=RSI/100
   float input_buffer[];
   ArrayResize(input_buffer, InpWindow * FEATURES);

   for(int i=0; i < InpWindow; i++)
     {
      int mql_idx = InpWindow - 1 - i;
      double atr_val = feat_atr_buffer[mql_idx];
      if(atr_val <= 0.0 || atr_val == EMPTY_VALUE)
         return;

      input_buffer[i * 3 + 0] = (float)((close[mql_idx] - open[mql_idx]) / atr_val);
      input_buffer[i * 3 + 1] = (float)((high[mql_idx] - low[mql_idx]) / atr_val);
      input_buffer[i * 3 + 2] = (float)(rsi_buffer[mql_idx] / 100.0);
     }

// 6. INFERENCE
   long output_label[];
   float output_probs[];
   ArrayResize(output_label, 1);
   ArrayResize(output_probs, 2);
   if(!OnnxRun(onnx_handle, ONNX_NO_CONVERSION, input_buffer, output_label, output_probs))
      return;

   long  prediction = output_label[0];
   float confidence  = (prediction == 1) ? output_probs[1] : output_probs[0];
   string pred_dir = PredDirection(prediction);
   string possible_entry_side = EntrySideFromPrediction(prediction);
   float prob_down = output_probs[0];
   float prob_up   = output_probs[1];
   LogInfo(StringFormat("Inference | pred=%s | p_up=%.4f | p_down=%.4f | conf=%.4f | candidate=%s",
                        pred_dir, prob_up, prob_down, confidence, possible_entry_side));

// 7. EXECUTION WITH TRADING HOURS FILTER
   bool has_position = PositionSelect(_Symbol);
   string robot_status = "WAITING";
   string decision_reason = "No decision yet";

   if(!has_position && valid_hours && confidence >= InpMinConf)
     {
      double sl_dist = current_atr * InpMultiplier;
      double tp_dist = sl_dist * 1.5;

      if((InpLogic == LOGIC_MIRROR && prediction == 1) || (InpLogic == LOGIC_NORMAL && prediction == 0))
        {
         double price = SymbolInfoDouble(_Symbol, SYMBOL_BID);
         if(m_trade.Sell(InpLot, _Symbol, price, price + sl_dist, price - tp_dist, "AI Sell"))
           {
            robot_status = "ORDER_SENT";
            decision_reason = "SELL order sent";
            LogInfo(StringFormat("Order sent | side=SELL | lot=%.2f | conf=%.4f | sl=%.5f | tp=%.5f",
                                 InpLot, confidence, price + sl_dist, price - tp_dist));
           }
         else
           {
            robot_status = "ORDER_ERROR";
            decision_reason = StringFormat("SELL failed (error=%d)", GetLastError());
            Print("[AI_EA] ERROR: ", decision_reason);
           }
        }
      else
        {
         double price = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
         if(m_trade.Buy(InpLot, _Symbol, price, price - sl_dist, price + tp_dist, "AI Buy"))
           {
            robot_status = "ORDER_SENT";
            decision_reason = "BUY order sent";
            LogInfo(StringFormat("Order sent | side=BUY | lot=%.2f | conf=%.4f | sl=%.5f | tp=%.5f",
                                 InpLot, confidence, price - sl_dist, price + tp_dist));
           }
         else
           {
            robot_status = "ORDER_ERROR";
            decision_reason = StringFormat("BUY failed (error=%d)", GetLastError());
            Print("[AI_EA] ERROR: ", decision_reason);
           }
        }
     }
   else
    {
     if(has_position)
       {
        robot_status = "HOLDING_POSITION";
        decision_reason = "Existing position on symbol";
       }
     else
        if(!valid_hours)
          {
           robot_status = "WAITING_SESSION";
           decision_reason = "Outside trading hours";
          }
        else
           if(confidence < InpMinConf)
             {
              robot_status = "WAITING_CONFIDENCE";
              decision_reason = StringFormat("Confidence %.2f%% below min %.2f%%",
                                             confidence * 100.0, InpMinConf * 100.0);
             }
    }

   Comment("\n\n\nAI ROBOT STATUS: ", robot_status,
           "\nUpdate trigger: ", trigger, " (every 60s)",
           "\nMode: ", (MQLInfoInteger(MQL_TESTER) ? "BACKTEST" : "LIVE"),
           " | Symbol/TF: ", _Symbol, " ", EnumToString(_Period),
           "\nModel: ", InpModelFile,
           "\nAnalyzed bar: ", TimeToString(current_bar, TIME_DATE|TIME_MINUTES),
           "\nSession: ", (valid_hours ? "ACTIVE" : "RESTRICTED"),
           " (", IntegerToString(InpStartHour), ":00-", IntegerToString(InpEndHour), ":00)",
           "\nPrediction: ", pred_dir,
           " | Candidate side: ", possible_entry_side,
           "\nProbabilities: UP=", DoubleToString(prob_up * 100.0, 2),
           "% | DOWN=", DoubleToString(prob_down * 100.0, 2), "%",
           "\nConfidence: ", DoubleToString(confidence * 100.0, 2),
           "% (min ", DoubleToString(InpMinConf * 100.0, 2), "%)",
           "\nRisk ATR: ", DoubleToString(current_atr, _Digits),
           " | Position: ", (has_position ? "OPEN" : "NONE"),
           "\nDecision: ", decision_reason);
  }

//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
int OnInit()
  {
   if(InpWindow <= 0 || InpRSIPeriod <= 0 || InpFeatATR <= 0 || InpATR <= 0)
     {
      Print("ERROR: Invalid input parameters. Window/periods must be > 0.");
      return(INIT_FAILED);
     }

   if(MQLInfoInteger(MQL_TESTER))
     {
      // Running in Strategy Tester
      onnx_handle = OnnxCreateFromBuffer(ExtModel, ONNX_DEFAULT);
      LogInfo("Execution mode: BACKTEST (embedded ONNX resource)");

     }
   else
     {
      // Load ONNX model directly from file
      onnx_handle = OnnxCreate(InpModelFile, ONNX_DEFAULT);
      LogInfo("Execution mode: LIVE/DEMO (loading ONNX from file)");
     }

   if(onnx_handle == INVALID_HANDLE)
     {
      Print("ERROR: Failed to load ONNX model: ", InpModelFile);
      Print("Error Code: ", GetLastError());
      Print("Make sure the file is in: C:\\Program Files\\MetaTrader 5\\MQL5\\Files\\");
      return(INIT_FAILED);
     }

   long input_shape[] = {1, InpWindow * FEATURES}; // window * features
   if(!OnnxSetInputShape(onnx_handle, 0, input_shape))
      return(INIT_FAILED);

   long out_shape_label[] = {1};
   OnnxSetOutputShape(onnx_handle, 0, out_shape_label);
   long out_shape_probs[] = {1, 2};
   OnnxSetOutputShape(onnx_handle, 1, out_shape_probs);

// Build indicator handles once so feature generation stays aligned with training.
   rsi_handle = iRSI(_Symbol, _Period, InpRSIPeriod, PRICE_CLOSE);
   if(rsi_handle == INVALID_HANDLE)
     {
      Print("ERROR: Failed to create RSI handle.");
      return(INIT_FAILED);
     }

   feat_atr_handle = iATR(_Symbol, _Period, InpFeatATR);
   if(feat_atr_handle == INVALID_HANDLE)
     {
      Print("ERROR: Failed to create feature ATR handle.");
      return(INIT_FAILED);
     }

   risk_atr_handle = iATR(_Symbol, _Period, InpATR);
   if(risk_atr_handle == INVALID_HANDLE)
     {
      Print("ERROR: Failed to create risk ATR handle.");
      return(INIT_FAILED);
     }

   m_trade.SetExpertMagicNumber(InpMagic);
   EventSetTimer(60);
   LogInfo(StringFormat("Initialized | model=%s | window=%d | rsi=%d | feat_atr=%d | risk_atr=%d | min_conf=%.2f",
                       InpModelFile, InpWindow, InpRSIPeriod, InpFeatATR, InpATR, InpMinConf));
   Comment("\n\n\nAI ROBOT STATUS: INITIALIZING\nWaiting for first 60-second inference cycle...");
   return(INIT_SUCCEEDED);
  }

//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
  {
   LogInfo(StringFormat("Deinitializing EA (reason=%d)", reason));
   EventKillTimer();
   if(onnx_handle != INVALID_HANDLE)
      OnnxRelease(onnx_handle);
   if(rsi_handle != INVALID_HANDLE)
      IndicatorRelease(rsi_handle);
   if(feat_atr_handle != INVALID_HANDLE)
      IndicatorRelease(feat_atr_handle);
   if(risk_atr_handle != INVALID_HANDLE)
      IndicatorRelease(risk_atr_handle);
   Comment("");
  }

//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void OnTick()
  {
// Inference and status updates are timer-driven (every 60 seconds).
  }

void OnTimer()
  {
   RunInferenceCycle("timer");
  }
//+------------------------------------------------------------------+
