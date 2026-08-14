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
input bool    InpMirrorEntryOperation = false; // Execute the opposite side when a buy signal is detected
input group "======== Risk Management ========"
input double  InpLot        = 1.0;          // Margin Percent (1.0=1.0%)
input int     InpMagic      = 123456;
input int     InpSLATR      = 14;          // Stop Loss ATR
input int     InpTPATR      = 14;          // Take Profit ATR
input int     InpMinDollars = 5;            // Minimum money to close trade (0=disabled)
input bool    InpUseSL      = true;        // Use Stop Loss
input group "======== Entry Protection ========"
input int     InpCooldownBars      = 2;     // Bars to wait after position close
input bool    InpRequirePrevCandleDir = true; // Require previous candle bullish
input bool    InpRequireCurrCandleDir = true; // Require current candle bullish
input bool    InpRequireVolatility = true; // Require volatility check (ATR)
input int     InpVolatilityPeriod  = 14;   // Amount of Candle for Volatility
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
bool     g_runtime_mirror_entry_operation = false;

#include "Utils.mqh"

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
ENUM_ORDER_TYPE GetEntryOrderType();
string GetEntryActionText();

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
   g_runtime_mirror_entry_operation = InpMirrorEntryOperation;

   UpdateTimeFilter();
   UpdatePositionState();
   if(RefreshMarketSnapshot() && PerformInference())
      UpdateComment();

   double lot_size = CalculateVolumeByPercent(InpLot, GetEntryOrderType());
   Print("Initial Lot Size Calculated: ", DoubleToString(lot_size, 2), " lots for InpLot=", InpLot, "%");

   SaveCurrentExperAdvisorInputs(MQLInfoString(MQL_PROGRAM_NAME) + ".set");

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

// Triggers specifically when inputs are changed via the GUI
   if(reason == REASON_PARAMETERS)
      SaveCurrentExperAdvisorInputs(MQLInfoString(MQL_PROGRAM_NAME) + ".set");

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

// 1. Validate the transaction is a deal addition
   if(trans.type == TRADE_TRANSACTION_DEAL_ADD)
     {
      // 2. Filter only deals for the current symbol
      if(trans.symbol == _Symbol)
        {
         // Load info of recent deal added to history
         if(HistoryDealSelect(trans.deal))
           {
            ENUM_DEAL_ENTRY dealEntry   = (ENUM_DEAL_ENTRY)HistoryDealGetInteger(trans.deal, DEAL_ENTRY);
            ENUM_DEAL_REASON dealReason = (ENUM_DEAL_REASON)HistoryDealGetInteger(trans.deal, DEAL_REASON);

            // --- CASE 1: OPENING A POSITION ---
            if(dealEntry == DEAL_ENTRY_IN)
               PlaySound("ok.wav"); // Sound for Opening Position

            // --- CASE 2 and 3: CLOSE BY TP OR SL ---
            else
               if(dealEntry == DEAL_ENTRY_OUT || dealEntry == DEAL_ENTRY_INOUT)
                 {
                  // Verify the close was for Take Profit
                  if(dealReason == DEAL_REASON_TP)
                     PlaySound("news.wav"); // Sound for Take Profit
                  // Verify the close was for Stop Loss
                  else
                     if(dealReason == DEAL_REASON_SL)
                        PlaySound("timeout.wav"); // Sound for Stop Loss
                     // Manual or Expert close
                     else
                        if(dealReason == DEAL_REASON_CLIENT || dealReason == DEAL_REASON_EXPERT)
                           PlaySound("alert.wav");
                 }
           }
        }
     }

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
void OnTimer()
  {
   Print("\n--- " + _Symbol + " Timer Triggered at ", TimeToString(TimeCurrent(), TIME_SECONDS), " ---");
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
   string execute_text = (g_prediction == 1 ? GetEntryActionText() : "NONE");
   string prev_candle_txt = "N/A";
   string curr_candle_txt = "N/A";
   if(ArraySize(g_open) > 1 && ArraySize(g_close) > 1)
     {
      prev_candle_txt = (g_open[1] < g_close[1] ? "BULLISH" : "BEARISH");
      curr_candle_txt = (g_open[0] < g_close[0] ? "BULLISH" : "BEARISH");
     }
   Comment("\n\n\nAI BUY-ONLY ", GetTimeframeString(_Period),
           "\nMirror Input: ", (InpMirrorEntryOperation ? "ENABLED" : "DISABLED"),
           "\nMirror Runtime: ", (g_runtime_mirror_entry_operation ? "ENABLED" : "DISABLED"),
           "\nConfidence: ", DoubleToString(g_confidence*100, 2), "% | Expected: ", DoubleToString(InpMinConf*100, 2), "%",
           "\nModel: ", InpModelFile,
           "\nTime: ", (g_valid_time ? "ACTIVE" : "RESTRICTED"),
           "\nWindow: ", InpWindow,
           "\nPrediction: ", pred_text,
           "\nExecute: ", execute_text,
           "\nPrev: ", prev_candle_txt, " (req=", InpRequirePrevCandleDir, ")",
           "\nCurr: ", curr_candle_txt, " (req=", InpRequireCurrCandleDir, ")");
  }

//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
ENUM_ORDER_TYPE GetEntryOrderType()
  {
   return (g_runtime_mirror_entry_operation ? ORDER_TYPE_SELL : ORDER_TYPE_BUY);
  }

//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
string GetEntryActionText()
  {
   return (GetEntryOrderType() == ORDER_TYPE_BUY ? "BUY" : "SELL");
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

   Print("Inference Result: Prediction = ", g_prediction,
         ", Buy Confidence = ", DoubleToString(g_confidence*100, 2), "% / ",
         DoubleToString(InpMinConf*100, 2), "%");
   Print("Prediction: ", (g_prediction == 1 ? "BUY" : "NO_BUY") +
         " | Probabilities: [no_buy=", DoubleToString(output_probs[0]*100, 2), "%, buy=", DoubleToString(output_probs[1]*100, 2), "%]");
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

   EVOLATILITY volatility = GetCurrentVolatility(InpVolatilityPeriod);
   bool volatility_ok = !InpRequireVolatility || (volatility == VOLATILITY_HIGH || volatility == VOLATILITY_VERY_HIGH);

   bool entry_allowed = no_open_pos && g_valid_time && cooldown_ok && prev_candle_ok && curr_candle_ok && prediction_ok && confidence_ok && volatility_ok;

   if(entry_allowed)
     {
      ENUM_ORDER_TYPE entry_order_type = GetEntryOrderType();
      bool is_buy_order = (entry_order_type == ORDER_TYPE_BUY);
      double sl_price = GetStopLoss(InpSLATR, entry_order_type);
      double tp_price = GetTakeProfit(InpTPATR, entry_order_type);
      double price = SymbolInfoDouble(_Symbol, is_buy_order ? SYMBOL_ASK : SYMBOL_BID);
      double sl = InpUseSL ? sl_price : 0;
      double tp = tp_price;
      double lot = CalculateVolumeByPercent(InpLot, entry_order_type);
      string entry_text = GetEntryActionText();
      Print("=== Attempting ", entry_text, " Entry === | Price: ", DoubleToString(price, _Digits),
            " | SL: ", DoubleToString(sl, _Digits),
            " | TP: ", DoubleToString(tp, _Digits),
            " | Lot: ", DoubleToString(lot, 2),
            " | Volatility: ", EnumToString(volatility));
      bool trade_ok = (is_buy_order
                       ? m_trade.Buy(lot, _Symbol, price, sl, tp, "AI BUY@" + DoubleToString(g_confidence, 2))
                       : m_trade.Sell(lot, _Symbol, price, sl, tp, "AI SELL@" + DoubleToString(g_confidence, 2)));
      if(trade_ok)
         Print("=== ", entry_text, " Executed @ ", DoubleToString(price, _Digits),
               " | sl: ", DoubleToString(sl, _Digits),
               " | tp: ", DoubleToString(tp, _Digits));
      else
        {
         bool previous_runtime_mirror = g_runtime_mirror_entry_operation;
         g_runtime_mirror_entry_operation = !g_runtime_mirror_entry_operation;
         Alert(entry_text, " ORDER FAILED | Retcode: ", m_trade.ResultRetcode(), " ", m_trade.ResultRetcodeDescription(),
               " | Mirror Runtime Flipped: ", (previous_runtime_mirror ? "ENABLED" : "DISABLED"),
               " -> ", (g_runtime_mirror_entry_operation ? "ENABLED" : "DISABLED"));
         Print("ORDER FAILURE MIRROR FLIP | Previous Runtime Mirror: ",
               (previous_runtime_mirror ? "ENABLED" : "DISABLED"),
               " | Current Runtime Mirror: ", (g_runtime_mirror_entry_operation ? "ENABLED" : "DISABLED"));
        }
      return;
     }

   Print("BUY ENTRY BYPASSED | PredOK: ", prediction_ok,
         " | ConfOK: ", confidence_ok,
         " | TimeOK: ", g_valid_time,
         " | CooldownOK: ", cooldown_ok);
   Print(" | NoPos: ", no_open_pos,
         " | PrevBull: ", prev_candle_bullish,
         " | CurrBull: ", curr_candle_bullish,
         " | PrevOK: ", prev_candle_ok);
   Print(" | CurrOK: ", curr_candle_ok,
         " | Volatility: ", EnumToString(volatility), " (", g_score, ")");
  }
//+------------------------------------------------------------------+
