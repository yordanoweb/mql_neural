//+------------------------------------------------------------------+
//|                                              SimpleONNX_EA.mq5   |
//+------------------------------------------------------------------+
#property strict

#include <Trade\Trade.mqh>

#resource "\\Files\\US100.cash_M15_20220311_20251230.onnx" as uchar ExtModel[];

//--- ENUMERATIONS
enum ENUM_LOGIC { LOGIC_NORMAL, LOGIC_MIRROR };

//--- INPUTS
input group "======== AI Configuration ========"
input string     InpModelFile  = "US100.cash_M15_20220311_20251230.onnx";  // Dynamic model filename
input ENUM_LOGIC InpLogic      = LOGIC_MIRROR; 
input float      InpMinConf    = 0.62;         
input int        InpStartHour  = 9;            
input int        InpEndHour    = 18;           
input int        InpWindow     = 20;           // Must match training --window
input group "======== Risk Management ========"
input double     InpLot        = 1;          
input int        InpMagic      = 123456;       
input int        InpATR        = 6;           
input double     InpMultiplier = 1.5;          
input group "======== Entry Protection ========"
input double     InpMinBodyATR        = 0.35;  // Min candle body / ATR on bar[1]
input double     InpMinRangeATR       = 0.60;  // Min candle range / ATR on bar[1]
input double     InpMinBodyRatio      = 0.55;  // Min body/range ratio on bar[1]
input double     InpMaxSpreadATRRatio = 0.15;  // Max spread / ATR allowed to enter
input int        InpCooldownBars      = 2;     // Bars to wait after position close
input group "======== Timer Settings ========"
input int        InpTimerSeconds = 60;  // Timer interval in seconds          

//--- GLOBAL VARIABLES
long     onnx_handle = INVALID_HANDLE;
CTrade   m_trade;
const int FEATURES    = 3;
long     g_prediction = 0;  // Last inference prediction
float    g_confidence = 0.0;  // Last inference confidence
bool     g_valid_time = false;  // Last time filter result
double   g_rsi_buffer[];  // RSI values
double   g_current_atr = 0;  // Current ATR value
double   g_close[];  // Close prices
double   g_open[];   // Open prices
double   g_high[];   // High prices
double   g_low[];    // Low prices
float    g_input_buffer[];  // Input buffer for inference
string   pred_text = "";  // Prediction text for display
datetime g_last_position_close_time = 0;
bool     g_prev_position_open = false;
double   g_last_body_atr = 0.0;
double   g_last_range_atr = 0.0;
double   g_last_body_ratio = 0.0;
bool     g_last_strong_move = false;

void UpdatePositionState()
{
   bool has_position = PositionSelect(_Symbol);
   if(g_prev_position_open && !has_position)
      g_last_position_close_time = TimeCurrent();
   g_prev_position_open = has_position;
}

bool IsCooldownFinished()
{
   if(InpCooldownBars <= 0 || g_last_position_close_time <= 0)
      return true;

   int bars_since_close = iBarShift(_Symbol, _Period, g_last_position_close_time, false);
   if(bars_since_close < 0)
      return false;

   return (bars_since_close >= InpCooldownBars);
}

bool IsSpreadAcceptable(double &spread, double &spread_atr)
{
   spread = 0.0;
   spread_atr = 0.0;
   if(g_current_atr <= 0.0)
      return false;

   double ask = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
   double bid = SymbolInfoDouble(_Symbol, SYMBOL_BID);
   if(ask <= 0.0 || bid <= 0.0 || ask < bid)
      return false;

   spread = ask - bid;
   spread_atr = spread / g_current_atr;
   return (spread_atr <= InpMaxSpreadATRRatio);
}

bool HasStrongMovement(double &body_atr, double &range_atr, double &body_ratio)
{
   body_atr = 0.0;
   range_atr = 0.0;
   body_ratio = 0.0;
   if(g_current_atr <= 0.0)
      return false;

   double body  = MathAbs(g_close[1] - g_open[1]);
   double range = g_high[1] - g_low[1];
   if(range <= 0.0)
      return false;

   body_atr = body / g_current_atr;
   range_atr = range / g_current_atr;
   body_ratio = body / range;

   return (body_atr >= InpMinBodyATR &&
           range_atr >= InpMinRangeATR &&
           body_ratio >= InpMinBodyRatio);
}

string GetDealReasonText(const long reason)
{
   switch((ENUM_DEAL_REASON)reason)
   {
      case DEAL_REASON_SL:     return "STOP_LOSS";
      case DEAL_REASON_TP:     return "TAKE_PROFIT";
      case DEAL_REASON_SO:     return "STOP_OUT";
      case DEAL_REASON_CLIENT: return "MANUAL_CLIENT";
      case DEAL_REASON_MOBILE: return "MANUAL_MOBILE";
      case DEAL_REASON_WEB:    return "MANUAL_WEB";
      case DEAL_REASON_EXPERT: return "EXPERT";
      default:                 return "OTHER";
   }
}

void ReportEntryInfo(const string side,
                     const double entry_price,
                     const double sl_price,
                     const double tp_price,
                     const double sl_dist,
                     const double tp_dist,
                     const double spread,
                     const double spread_atr,
                     const double body_atr,
                     const double range_atr,
                     const double body_ratio,
                     const bool time_ok,
                     const bool cooldown_ok,
                     const bool spread_ok,
                     const bool strong_move_ok)
{
   Print("\n--- Entry Report at ", TimeToString(TimeCurrent(), TIME_SECONDS), " ---");
   Print("Side: ", side, " | Symbol: ", _Symbol, " | TF: ", GetTimeframeString(_Period), " | Magic: ", InpMagic);
   Print("Lot: ", DoubleToString(InpLot, 2), " | Entry: ", DoubleToString(entry_price, _Digits),
         " | SL: ", DoubleToString(sl_price, _Digits), " | TP: ", DoubleToString(tp_price, _Digits));
   Print("ATR: ", DoubleToString(g_current_atr, _Digits), " | SL_dist: ", DoubleToString(sl_dist, _Digits),
         " | TP_dist: ", DoubleToString(tp_dist, _Digits), " | Spread: ", DoubleToString(spread, _Digits),
         " | Spread/ATR: ", DoubleToString(spread_atr, 4));
   Print("AI: prediction=", g_prediction, " confidence=", DoubleToString(g_confidence * 100.0, 2), "%",
         " | logic=", (InpLogic == LOGIC_MIRROR ? "MIRROR" : "NORMAL"));
   Print("Protections: time_ok=", (time_ok ? "true" : "false"),
         " cooldown_ok=", (cooldown_ok ? "true" : "false"),
         " spread_ok=", (spread_ok ? "true" : "false"),
         " strong_move_ok=", (strong_move_ok ? "true" : "false"));
   Print("Strength: body_atr=", DoubleToString(body_atr, 4),
         " range_atr=", DoubleToString(range_atr, 4),
         " body_ratio=", DoubleToString(body_ratio, 4));
}

void ReportEntryBypassInfo(const bool no_open_pos,
                           const bool time_ok,
                           const bool confidence_ok,
                           const bool cooldown_ok,
                           const bool spread_ok,
                           const bool strong_move_ok,
                           const double spread,
                           const double spread_atr,
                           const double body_atr,
                           const double range_atr,
                           const double body_ratio)
{
   Print("\n--- Entry Bypassed at ", TimeToString(TimeCurrent(), TIME_SECONDS), " ---");
   Print("Symbol: ", _Symbol, " | TF: ", GetTimeframeString(_Period), " | Magic: ", InpMagic);
   Print("AI: prediction=", g_prediction, " confidence=", DoubleToString(g_confidence * 100.0, 2), "% | min_conf=",
         DoubleToString(InpMinConf * 100.0, 2), "%");
   Print("Gate States: no_open_pos=", (no_open_pos ? "true" : "false"),
         " time_ok=", (time_ok ? "true" : "false"),
         " confidence_ok=", (confidence_ok ? "true" : "false"),
         " cooldown_ok=", (cooldown_ok ? "true" : "false"),
         " spread_ok=", (spread_ok ? "true" : "false"),
         " strong_move_ok=", (strong_move_ok ? "true" : "false"));
   if(!no_open_pos)
      Print("Bypass reason: Existing position is open for symbol ", _Symbol);
   if(!time_ok)
   {
      MqlDateTime now_dt;
      TimeCurrent(now_dt);
      Print("Bypass reason: Trading window blocked. Current hour=", now_dt.hour,
            " | allowed=[", InpStartHour, ", ", InpEndHour, ")");
   }
   if(!confidence_ok)
      Print("Bypass reason: Confidence below threshold. current=", DoubleToString(g_confidence, 4),
            " | required>=", DoubleToString(InpMinConf, 4));
   if(!cooldown_ok)
      Print("Bypass reason: Cooldown active. Required bars=", InpCooldownBars,
            " | last_close_time=", TimeToString(g_last_position_close_time, TIME_SECONDS));
   if(!spread_ok)
      Print("Bypass reason: Spread filter blocked. spread=", DoubleToString(spread, _Digits),
            " spread_atr=", DoubleToString(spread_atr, 4),
            " | max_spread_atr=", DoubleToString(InpMaxSpreadATRRatio, 4));
   if(!strong_move_ok)
      Print("Bypass reason: Movement strength below thresholds. body_atr=", DoubleToString(body_atr, 4),
            " range_atr=", DoubleToString(range_atr, 4),
            " body_ratio=", DoubleToString(body_ratio, 4),
            " | mins=[", DoubleToString(InpMinBodyATR, 4), ", ",
            DoubleToString(InpMinRangeATR, 4), ", ", DoubleToString(InpMinBodyRatio, 4), "]");
}

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

int OnInit()
{
   if(InpWindow < 2)
   {
      Print("ERROR: InpWindow must be >= 2. Current value: ", InpWindow);
      return(INIT_PARAMETERS_INCORRECT);
   }

   bool loaded_from_resource = false;

   // Check if we are in tester
   if(!MQLInfoInteger(MQL_TESTER))
   {
      // The ONNX path must be hardcoded in "ExtModel" resource
      // due to MQL5 file access restrictions in live environment.
      // This is the only way to load ONNX for backtesting.
      onnx_handle = OnnxCreateFromBuffer(ExtModel, ONNX_DEFAULT);
      loaded_from_resource = true;
   } else
   {
      // In live, we can load ONNX dynamically from InpModelFile.
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

   long input_shape[] = {1, InpWindow * FEATURES};
   if(!OnnxSetInputShape(onnx_handle, 0, input_shape)) return(INIT_FAILED);

   long out_shape_label[] = {1};
   OnnxSetOutputShape(onnx_handle, 0, out_shape_label);
   long out_shape_probs[] = {1, 2};
   OnnxSetOutputShape(onnx_handle, 1, out_shape_probs);

   m_trade.SetExpertMagicNumber(InpMagic);
   EventSetTimer(InpTimerSeconds);  // Set timer to configured interval
   return(INIT_SUCCEEDED);
}

void OnDeinit(const int reason) 
{ 
   EventKillTimer();  // Stop timer
   if(onnx_handle != INVALID_HANDLE) OnnxRelease(onnx_handle);

   Comment("");  // Clear comment on deinit
}

void OnTick()
{
   // 1. CORRECT TIME FILTER
   MqlDateTime dt;
   TimeCurrent(dt); 
   g_valid_time = (dt.hour >= InpStartHour && dt.hour < InpEndHour);
   UpdatePositionState();

   // 2. CANDLE CONTROL
   static datetime last_bar = 0;
   datetime current_bar = iTime(_Symbol, _Period, 0);
   if(current_bar == last_bar) return;
   last_bar = current_bar;

   // 3. DATA
   GetData();

   // 4. INDICATORS
   GetIndicators();

   // 5. INPUT BUFFER WITH NORMALIZATION BY _Digits
   BuildInputBuffer();

   // 7. EXECUTION WITH TIME FILTER (using global inference results from OnTimer)
   bool no_open_pos = !PositionSelect(_Symbol);
   bool confidence_ok = (g_confidence >= InpMinConf);
   bool cooldown_ok = IsCooldownFinished();
   double spread = 0.0, spread_atr = 0.0;
   bool spread_ok = IsSpreadAcceptable(spread, spread_atr);
   double body_atr = 0.0, range_atr = 0.0, body_ratio = 0.0;
   bool strong_move = HasStrongMovement(body_atr, range_atr, body_ratio);
   g_last_body_atr = body_atr;
   g_last_range_atr = range_atr;
   g_last_body_ratio = body_ratio;
   g_last_strong_move = strong_move;
   if(no_open_pos && g_valid_time && confidence_ok && cooldown_ok && spread_ok && strong_move)
   {
      double sl_dist = g_current_atr * InpMultiplier;
      double tp_dist = sl_dist;

      if((InpLogic == LOGIC_MIRROR && g_prediction == 1) || (InpLogic == LOGIC_NORMAL && g_prediction == 0))
      {
         double price = SymbolInfoDouble(_Symbol, SYMBOL_BID);
         double sl = price + sl_dist;
         double tp = price - tp_dist;
         if(m_trade.Sell(InpLot, _Symbol, price, sl, tp, "AI SELL"))
            ReportEntryInfo("SELL", price, sl, tp, sl_dist, tp_dist, spread, spread_atr, body_atr, range_atr, body_ratio, g_valid_time, cooldown_ok, spread_ok, strong_move);
      }
      else
      {
         double price = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
         double sl = price - sl_dist;
         double tp = price + tp_dist;
         if(m_trade.Buy(InpLot, _Symbol, price, sl, tp, "AI BUY"))
            ReportEntryInfo("BUY", price, sl, tp, sl_dist, tp_dist, spread, spread_atr, body_atr, range_atr, body_ratio, g_valid_time, cooldown_ok, spread_ok, strong_move);
      }
   }
   else
   {
      ReportEntryBypassInfo(no_open_pos, g_valid_time, confidence_ok, cooldown_ok, spread_ok, strong_move,
                            spread, spread_atr, body_atr, range_atr, body_ratio);
   }
}

void OnTradeTransaction(const MqlTradeTransaction &trans,
                        const MqlTradeRequest &request,
                        const MqlTradeResult &result)
{
   if(trans.type == TRADE_TRANSACTION_DEAL_ADD)
      ReportExitInfo(trans.deal);
}

void GetData()
{
   // 3. DATA - Load OHLC prices
   ArraySetAsSeries(g_close, true); ArraySetAsSeries(g_open, true);
   ArraySetAsSeries(g_high, true);  ArraySetAsSeries(g_low, true);

   if(CopyClose(_Symbol, _Period, 0, InpWindow + 15, g_close) < InpWindow + 15 ||
      CopyOpen(_Symbol, _Period, 0, InpWindow, g_open) < InpWindow) return;
   
   CopyHigh(_Symbol, _Period, 0, InpWindow, g_high);
   CopyLow(_Symbol, _Period, 0, InpWindow, g_low);
}

void GetIndicators()
{
   // 4. INDICATORS - Get RSI and ATR values
   int rsi_handle = iRSI(_Symbol, _Period, 14, PRICE_CLOSE);
   ArraySetAsSeries(g_rsi_buffer, true);
   CopyBuffer(rsi_handle, 0, 0, InpWindow, g_rsi_buffer);

   int atr_handle = iATR(_Symbol, _Period, InpATR);
   double atr_buffer[];
   ArraySetAsSeries(atr_buffer, true);
   CopyBuffer(atr_handle, 0, 0, 1, atr_buffer);
   g_current_atr = atr_buffer[0];
}

void BuildInputBuffer()
{
   // 5. INPUT BUFFER WITH NORMALIZATION BY _Digits
   ArrayResize(g_input_buffer, InpWindow * FEATURES);
   
   // _Digits is the correct variable. If 5 or 3 decimals, we adjust to pips (x10).
   double pip_unit = _Point * (_Digits == 5 || _Digits == 3 ? 10 : 1);

   for(int i=0; i < InpWindow; i++)
   {
      int mql_idx = InpWindow - 1 - i;
      g_input_buffer[i * 3 + 0] = (float)((g_close[mql_idx] - g_open[mql_idx]) / pip_unit);
      g_input_buffer[i * 3 + 1] = (float)((g_high[mql_idx] - g_low[mql_idx]) / pip_unit);
      g_input_buffer[i * 3 + 2] = (float)(g_rsi_buffer[mql_idx] / 100.0);
   }
}

string GetTimeframeString(ENUM_TIMEFRAMES tf)
{
   switch(tf)
   {
      case PERIOD_M1: return "M1";
      case PERIOD_M5: return "M5";
      case PERIOD_M15: return "M15";
      case PERIOD_M30: return "M30";
      case PERIOD_H1: return "H1";
      case PERIOD_H4: return "H4";
      case PERIOD_D1: return "D1";
      default: return "Unknown TF";
   }
}

void OnTimer()
{
   Print("\n--- Timer Triggered at ", TimeToString(TimeCurrent(), TIME_SECONDS), " ---");

   PerformInference(); 
   UpdateComment();
}

void UpdateComment()
{
   // This function can be called to update the comment display with latest info
   pred_text = (g_prediction == 1 && InpLogic == LOGIC_MIRROR) || (g_prediction == 0 && InpLogic == LOGIC_NORMAL) ? "SELL" : "BUY";
   Comment("\n\n\nAI " + GetTimeframeString(_Period) + " | Confidence: ", DoubleToString(g_confidence*100, 2), "%",
           "\nTime: ", (g_valid_time ? "ACTIVE" : "RESTRICTED"),
           "\nLogic: ", (InpLogic == LOGIC_MIRROR ? "MIRROR" : "NORMAL"),
           "\nWindow: ", InpWindow,
           "\nMove Strength: ", (g_last_strong_move ? "PASS" : "BLOCK"),
           " | body_atr=", DoubleToString(g_last_body_atr, 3),
           " range_atr=", DoubleToString(g_last_range_atr, 3),
           " body_ratio=", DoubleToString(g_last_body_ratio, 3),
           "\nPrediction: ", pred_text);
}

void PerformInference()
{
   // 6. INFERENCE
   long output_label[]; float output_probs[];
   ArrayResize(output_label, 1); ArrayResize(output_probs, 2);
   if(!OnnxRun(onnx_handle, ONNX_NO_CONVERSION, g_input_buffer, output_label, output_probs)) return;

   g_prediction = output_label[0];
   g_confidence = (g_prediction == 1) ? output_probs[1] : output_probs[0];

   Print("Inference Result: Prediction = ", g_prediction, ", Confidence = ", DoubleToString(g_confidence*100, 2), "%");
   Print("Prediction: ", pred_text);
   Print("Probabilities: [", DoubleToString(output_probs[0]*100, 2), "%, ", DoubleToString(output_probs[1]*100, 2), "%]");
}