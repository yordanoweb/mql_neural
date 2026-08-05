//+------------------------------------------------------------------+
//|                                          BuyOnly_ONNX_EA.mq5     |
//+------------------------------------------------------------------+
#property strict

#include <Trade\Trade.mqh>

#resource "\\Files\\XAU_M15_buy_only.onnx" as uchar ExtModel[];

//--- INPUTS
input group "======== AI Configuration ========"
input string  InpModelFile  = "BTCUSD_M15_202606121618.onnx";  // Dynamic model filename
input float   InpMinConf    = 0.62;                            // Min buy probability to enter
input int     InpStartHour  = 9;
input int     InpEndHour    = 18;
input int     InpWindow     = 20;           // Must match training --window
input group "======== Risk Management ========"
input double  InpLot        = 1;
input int     InpMagic      = 123456;
input int     InpSLPoints   = 600;          // Stop Loss distance in points
input int     InpTPPoints   = 600;          // Take Profit distance in points
input int     InpMinDollars = 5;            // Minimum money to close trade (0=disabled)
input bool    InpUseSL      = true;        // Use Stop Loss
input group "======== Entry Protection ========"
input int     InpCooldownBars      = 2;     // Bars to wait after position close
input bool    InpRequirePrevCandleDir = true; // Require previous candle bullish
input bool    InpRequireCurrCandleDir = true; // Require current candle bullish
input group "======== Timer Settings ========"
input int     InpTimerSeconds = 60;  // Timer interval in seconds
input group "======== Debug / Test ========="
input bool    InpDebug     = false; // Log Debug Info

//--- GLOBAL VARIABLES
long     onnx_handle = INVALID_HANDLE;
CTrade   m_trade;
const int FEATURES    = 3;
long     g_prediction = 0;      // Last inference label (0=no_buy, 1=buy)
float    g_confidence = 0.0;    // Last buy probability
bool     g_valid_time = false;  // Last time filter result
double   g_rsi_buffer[];        // RSI values
double   g_close[];             // Close prices
double   g_open[];              // Open prices
double   g_high[];              // High prices
double   g_low[];               // Low prices
float    g_input_buffer[];      // Input buffer for inference
string   pred_text = "";        // Prediction text for display
datetime g_last_position_close_time = 0;
bool     g_prev_position_open = false;
int      g_rsi_handle = INVALID_HANDLE;

//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool RefreshMarketSnapshot();
bool GetData();
bool GetIndicators();
bool BuildInputBuffer();
bool PerformInference();
void TryExecuteBuyEntry();
string GetTimeframeString(ENUM_TIMEFRAMES tf);

//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void UpdateTimeFilter()
  {
   MqlDateTime dt;
   TimeCurrent(dt);
   g_valid_time = (dt.hour >= InpStartHour && dt.hour < InpEndHour);
  }

//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void UpdatePositionState()
  {
   bool has_position = PositionSelect(_Symbol);
   if(g_prev_position_open && !has_position)
      g_last_position_close_time = TimeCurrent();
   g_prev_position_open = has_position;
  }

//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool IsCooldownFinished()
  {
   if(InpCooldownBars <= 0 || g_last_position_close_time <= 0)
      return true;

   int bars_since_close = iBarShift(_Symbol, _Period, g_last_position_close_time, false);
   if(bars_since_close < 0)
      return false;

   return (bars_since_close >= InpCooldownBars);
  }

//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool IsEntryPriceProfitable()
  {
   if(InpMinDollars <= 0)
      return false;

   if(!PositionSelect(_Symbol))
      return false;

   double profit = PositionGetDouble(POSITION_PROFIT);
   double commission = PositionGetDouble(POSITION_COMMISSION);
   double swap = PositionGetDouble(POSITION_SWAP);
   double net_profit = profit + commission + swap;

   return (net_profit >= InpMinDollars);
  }

//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
string GetDealReasonText(const long reason)
  {
   switch((ENUM_DEAL_REASON)reason)
     {
      case DEAL_REASON_SL:
         return "STOP_LOSS";
      case DEAL_REASON_TP:
         return "TAKE_PROFIT";
      case DEAL_REASON_SO:
         return "STOP_OUT";
      case DEAL_REASON_CLIENT:
         return "MANUAL_CLIENT";
      case DEAL_REASON_MOBILE:
         return "MANUAL_MOBILE";
      case DEAL_REASON_WEB:
         return "MANUAL_WEB";
      case DEAL_REASON_EXPERT:
         return "EXPERT";
      default:
         return "OTHER";
     }
  }

//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void ReportExitInfo(const ulong deal_ticket)
  {
   if(deal_ticket == 0 || !HistoryDealSelect(deal_ticket))
      return;

   string symbol = HistoryDealGetString(deal_ticket, DEAL_SYMBOL);
   long magic = HistoryDealGetInteger(deal_ticket, DEAL_MAGIC);
   long entry = HistoryDealGetInteger(deal_ticket, DEAL_ENTRY);
   if(symbol != _Symbol || magic != InpMagic || (entry != DEAL_ENTRY_OUT && entry != DEAL_ENTRY_OUT_BY))
      return;

   datetime deal_time = (datetime)HistoryDealGetInteger(deal_ticket, DEAL_TIME);
   long reason = HistoryDealGetInteger(deal_ticket, DEAL_REASON);
   long deal_type = HistoryDealGetInteger(deal_ticket, DEAL_TYPE);
   double price = HistoryDealGetDouble(deal_ticket, DEAL_PRICE);
   double volume = HistoryDealGetDouble(deal_ticket, DEAL_VOLUME);
   double profit = HistoryDealGetDouble(deal_ticket, DEAL_PROFIT);
   double commission = HistoryDealGetDouble(deal_ticket, DEAL_COMMISSION);
   double swap = HistoryDealGetDouble(deal_ticket, DEAL_SWAP);
   double net = profit + commission + swap;
   string comment = HistoryDealGetString(deal_ticket, DEAL_COMMENT);
   string side = (deal_type == DEAL_TYPE_BUY ? "BUY_DEAL" : (deal_type == DEAL_TYPE_SELL ? "SELL_DEAL" : "OTHER"));

   Print("\n--- Exit Report at ", TimeToString(deal_time, TIME_SECONDS), " ---");
   Print("Ticket: ", deal_ticket, " | Symbol: ", symbol, " | Magic: ", magic, " | Side: ", side);
   Print("Reason: ", GetDealReasonText(reason), " | Price: ", DoubleToString(price, _Digits),
         " | Volume: ", DoubleToString(volume, 2));
   Print("P/L: ", DoubleToString(profit, 2), " | Commission: ", DoubleToString(commission, 2),
         " | Swap: ", DoubleToString(swap, 2), " | Net: ", DoubleToString(net, 2));
   Print("Comment: ", comment);
  }

//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
int OnInit()
  {
   if(InpWindow < 2)
     {
      Print("ERROR: InpWindow must be >= 2. Current value: ", InpWindow);
      return(INIT_PARAMETERS_INCORRECT);
     }

   bool loaded_from_resource = false;

// Check if we are in backtesting environment
   if(MQLInfoInteger(MQL_TESTER))
     {
      // In backtesting - load ONNX from resource buffer (ExtModel)
      onnx_handle = OnnxCreateFromBuffer(ExtModel, ONNX_DEFAULT);
      loaded_from_resource = true;
     }
   else
     {
      // Live trading - load ONNX dynamically from InpModelFile
      onnx_handle = OnnxCreate(InpModelFile, ONNX_DEFAULT);
     }

   if(onnx_handle == INVALID_HANDLE)
     {
      Print("ERROR: Failed to load ONNX model: ", InpModelFile);
      Print("Error Code: ", GetLastError());
      Print("Make sure the file is in: C:\\Program Files\\MetaTrader 5\\MQL5\\Files\\");
      return(INIT_FAILED);
     }

   Print("ONNX loaded successfully: ", (loaded_from_resource ? "[RESOURCE] ExtModel" : InpModelFile));

   g_rsi_handle = iRSI(_Symbol, _Period, 14, PRICE_CLOSE);
   if(g_rsi_handle == INVALID_HANDLE)
     {
      Print("ERROR: Failed to create RSI handle. Error Code: ", GetLastError());
      return(INIT_FAILED);
     }

   long input_shape[] = {1, InpWindow * FEATURES};
   if(!OnnxSetInputShape(onnx_handle, 0, input_shape))
      return(INIT_FAILED);

   long out_shape_label[] = {1};
   OnnxSetOutputShape(onnx_handle, 0, out_shape_label);
   long out_shape_probs[] = {1, 2};
   OnnxSetOutputShape(onnx_handle, 1, out_shape_probs);

   m_trade.SetExpertMagicNumber(InpMagic);
   EventSetTimer(InpTimerSeconds);

   UpdateTimeFilter();
   UpdatePositionState();
   if(RefreshMarketSnapshot() && PerformInference())
      UpdateComment();

   return(INIT_SUCCEEDED);
  }

//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
  {
   EventKillTimer();
   if(onnx_handle != INVALID_HANDLE)
      OnnxRelease(onnx_handle);
   if(g_rsi_handle != INVALID_HANDLE)
      IndicatorRelease(g_rsi_handle);
   Comment("");
  }

//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void OnTick()
  {
// 1. Time filter and position tracking
   UpdateTimeFilter();
   UpdatePositionState();

// Optional: close profitable position early
   if(PositionSelect(_Symbol) && IsEntryPriceProfitable())
     {
      if(m_trade.PositionClose(_Symbol))
         return;
     }
  }

//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void OnTradeTransaction(const MqlTradeTransaction &trans,
                        const MqlTradeRequest &request,
                        const MqlTradeResult &result)
  {
   if(trans.type == TRADE_TRANSACTION_DEAL_ADD)
      ReportExitInfo(trans.deal);
  }

//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool RefreshMarketSnapshot()
  {
   return (GetData() && GetIndicators() && BuildInputBuffer());
  }

//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool GetData()
  {
   ArraySetAsSeries(g_close, true);
   ArraySetAsSeries(g_open, true);
   ArraySetAsSeries(g_high, true);
   ArraySetAsSeries(g_low, true);

   if(CopyClose(_Symbol, _Period, 0, InpWindow + 15, g_close) < InpWindow + 15)
      return false;
   if(CopyOpen(_Symbol, _Period, 0, InpWindow, g_open) < InpWindow)
      return false;
   if(CopyHigh(_Symbol, _Period, 0, InpWindow, g_high) < InpWindow)
      return false;
   if(CopyLow(_Symbol, _Period, 0, InpWindow, g_low) < InpWindow)
      return false;

   return true;
  }

//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool GetIndicators()
  {
   if(g_rsi_handle == INVALID_HANDLE)
      return false;

   ArraySetAsSeries(g_rsi_buffer, true);
   if(CopyBuffer(g_rsi_handle, 0, 0, InpWindow, g_rsi_buffer) < InpWindow)
      return false;

   return true;
  }

//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool BuildInputBuffer()
  {
// Build input buffer matching train_buy_only.py: raw price body/range + RSI/100
   if(ArraySize(g_close) < InpWindow || ArraySize(g_open) < InpWindow ||
      ArraySize(g_high) < InpWindow || ArraySize(g_low) < InpWindow ||
      ArraySize(g_rsi_buffer) < InpWindow)
      return false;

   ArrayResize(g_input_buffer, InpWindow * FEATURES);

   for(int i=0; i < InpWindow; i++)
     {
      int mql_idx = InpWindow - 1 - i;
      g_input_buffer[i * 3 + 0] = (float)(g_close[mql_idx] - g_open[mql_idx]);
      g_input_buffer[i * 3 + 1] = (float)(g_high[mql_idx] - g_low[mql_idx]);
      g_input_buffer[i * 3 + 2] = (float)(g_rsi_buffer[mql_idx] / 100.0);
     }

   return true;
  }

//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
string GetTimeframeString(ENUM_TIMEFRAMES tf)
  {
   switch(tf)
     {
      case PERIOD_M1:
         return "M1";
      case PERIOD_M5:
         return "M5";
      case PERIOD_M15:
         return "M15";
      case PERIOD_M30:
         return "M30";
      case PERIOD_H1:
         return "H1";
      case PERIOD_H4:
         return "H4";
      case PERIOD_D1:
         return "D1";
      default:
         return "Unknown TF";
     }
  }

//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void OnTimer()
  {
   Print("\n--- Timer Triggered at ", TimeToString(TimeCurrent(), TIME_SECONDS), " ---");
   UpdateTimeFilter();
   UpdatePositionState();
   if(!RefreshMarketSnapshot())
      return;
   if(!PerformInference())
      return;
   TryExecuteBuyEntry();
   UpdateComment();
  }

//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void UpdateComment()
  {
   pred_text = (g_prediction == 1 ? "BUY" : "NO_BUY");
   string prev_candle_txt = "N/A";
   string curr_candle_txt = "N/A";
   if(ArraySize(g_open) > 1 && ArraySize(g_close) > 1)
     {
      prev_candle_txt = (g_open[1] < g_close[1] ? "BULLISH" : "BEARISH");
      curr_candle_txt = (g_open[0] < g_close[0] ? "BULLISH" : "BEARISH");
     }
   Comment("\n\n\nAI BUY-ONLY ", GetTimeframeString(_Period), " | Confidence: ", DoubleToString(g_confidence*100, 2), "%",
           "\nModel: ", InpModelFile,
           "\nTime: ", (g_valid_time ? "ACTIVE" : "RESTRICTED"),
           "\nWindow: ", InpWindow,
           "\nPrediction: ", pred_text,
           "\nPrev: ", prev_candle_txt, " (req=", InpRequirePrevCandleDir, ")",
           "\nCurr: ", curr_candle_txt, " (req=", InpRequireCurrCandleDir, ")");
  }

//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool PerformInference()
  {
   long output_label[];
   float output_probs[];
   ArrayResize(output_label, 1);
   ArrayResize(output_probs, 2);
   if(!OnnxRun(onnx_handle, ONNX_NO_CONVERSION, g_input_buffer, output_label, output_probs))
      return false;

   g_prediction = output_label[0];
   g_confidence = output_probs[1];  // buy probability

   Print("Inference Result: Prediction = ", g_prediction, ", Buy Confidence = ", DoubleToString(g_confidence*100, 2), "%");
   Print("Prediction: ", (g_prediction == 1 ? "BUY" : "NO_BUY"));
   Print("Probabilities: [no_buy=", DoubleToString(output_probs[0]*100, 2), "%, buy=", DoubleToString(output_probs[1]*100, 2), "%]");
   return true;
  }

//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void TryExecuteBuyEntry()
  {
   if(ArraySize(g_open) < 2 || ArraySize(g_close) < 2)
      return;

   bool prev_candle_bullish = (g_open[1] < g_close[1]);
   bool curr_candle_bullish = (g_open[0] < g_close[0]);
   bool prev_candle_ok = !InpRequirePrevCandleDir || prev_candle_bullish;
   bool curr_candle_ok = !InpRequireCurrCandleDir || curr_candle_bullish;

   bool no_open_pos = !PositionSelect(_Symbol);
   bool cooldown_ok = IsCooldownFinished();
   bool prediction_ok = (g_prediction == 1);
   bool confidence_ok = (g_confidence >= InpMinConf);

   bool entry_allowed = no_open_pos && g_valid_time && cooldown_ok && prev_candle_ok && curr_candle_ok && prediction_ok;

   if(entry_allowed)
     {
      double sl_dist = InpSLPoints * _Point;
      double tp_dist = InpTPPoints * _Point;
      double price = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
      double sl = InpUseSL ? (price - sl_dist) : 0;
      double tp = price + tp_dist;
      if(m_trade.Buy(InpLot, _Symbol, price, sl, tp, "AI BUY"))
         Print("=== Buy Executed @ ", DoubleToString(price, _Digits),
               " | sl: ", DoubleToString(sl, _Digits),
               " | tp: ", DoubleToString(tp, _Digits));
      else
         Print("BUY ORDER FAILED | Retcode: ", m_trade.ResultRetcode(), " ", m_trade.ResultRetcodeDescription());
      return;
     }

   Print("BUY ENTRY BYPASSED | PredOK: ", prediction_ok,
         " ConfOK: ", confidence_ok,
         " TimeOK: ", g_valid_time,
         " CooldownOK: ", cooldown_ok,
         " NoPos: ", no_open_pos,
         " PrevBull: ", prev_candle_bullish,
         " CurrBull: ", curr_candle_bullish,
         " PrevOK: ", prev_candle_ok,
         " CurrOK: ", curr_candle_ok);
  }
//+------------------------------------------------------------------+
