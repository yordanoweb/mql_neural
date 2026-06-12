//+------------------------------------------------------------------+
//|                                              SimpleONNX_EA.mq5   |
//+------------------------------------------------------------------+
#property strict

#include <Trade\Trade.mqh>

#resource "\\Files\\US100.cash_M15_20220311_20251230.onnx" as uchar ExtModel[];

//--- ENUMERATIONS
enum ENUM_LOGIC { LOGIC_NORMAL, LOGIC_MIRROR };

//--- INPUTS
input group "AI Configuration"
input string     InpModelFile  = "US100.cash_M15_20220311_20251230.onnx";  // Dynamic model filename
input ENUM_LOGIC InpLogic      = LOGIC_MIRROR; 
input float      InpMinConf    = 0.62;         
input int        InpStartHour  = 9;            
input int        InpEndHour    = 18;           
input group "Risk Management"
input double     InpLot        = 1;          
input int        InpMagic      = 123456;       
input int        InpATR        = 6;           
input double     InpMultiplier = 1.5;          
input group "Entry Protection"
input double     InpMinBodyATR        = 0.35;  // Min candle body / ATR on bar[1]
input double     InpMinRangeATR       = 0.60;  // Min candle range / ATR on bar[1]
input double     InpMinBodyRatio      = 0.55;  // Min body/range ratio on bar[1]
input double     InpMaxSpreadATRRatio = 0.15;  // Max spread / ATR allowed to enter
input int        InpCooldownBars      = 2;     // Bars to wait after position close
input group "Timer Settings"
input int        InpTimerSeconds = 60;  // Timer interval in seconds          

//--- GLOBAL VARIABLES
long     onnx_handle = INVALID_HANDLE;
CTrade   m_trade;
const int WINDOW_SIZE = 20; // For M15 we use window of 20
const int FEATURES    = 3;
long     g_prediction = 0;  // Last inference prediction
float    g_confidence = 0.0;  // Last inference confidence
bool     g_valid_time = false;  // Last time filter result
double   g_rsi_buffer[20];  // RSI values
double   g_current_atr = 0;  // Current ATR value
double   g_close[];  // Close prices
double   g_open[];   // Open prices
double   g_high[];   // High prices
double   g_low[];    // Low prices
float    g_input_buffer[];  // Input buffer for inference
string   pred_text = "";  // Prediction text for display
datetime g_last_position_close_time = 0;
bool     g_prev_position_open = false;

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

bool IsSpreadAcceptable()
{
   if(g_current_atr <= 0.0)
      return false;

   double ask = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
   double bid = SymbolInfoDouble(_Symbol, SYMBOL_BID);
   if(ask <= 0.0 || bid <= 0.0 || ask < bid)
      return false;

   double spread = ask - bid;
   return (spread <= g_current_atr * InpMaxSpreadATRRatio);
}

bool HasStrongMovement()
{
   if(g_current_atr <= 0.0)
      return false;

   double body  = MathAbs(g_close[1] - g_open[1]);
   double range = g_high[1] - g_low[1];
   if(range <= 0.0)
      return false;

   double body_atr   = body / g_current_atr;
   double range_atr  = range / g_current_atr;
   double body_ratio = body / range;

   return (body_atr >= InpMinBodyATR &&
           range_atr >= InpMinRangeATR &&
           body_ratio >= InpMinBodyRatio);
}

int OnInit()
{
   // Check if we are in tester
   if(!MQLInfoInteger(MQL_TESTER))
   {
      // The ONNX path must be hardcoded in "ExtModel" resource
      // due to MQL5 file access restrictions in live environment.
      // This is the only way to load ONNX for backtesting.
      onnx_handle = OnnxCreateFromBuffer(ExtModel, ONNX_DEFAULT);
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

   long input_shape[] = {1, 60}; // 20 candles * 3 features
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
   bool cooldown_ok = IsCooldownFinished();
   bool spread_ok = IsSpreadAcceptable();
   bool strong_move = HasStrongMovement();
   if(no_open_pos && g_valid_time && g_confidence >= InpMinConf && cooldown_ok && spread_ok && strong_move)
   {
      double sl_dist = g_current_atr * InpMultiplier;
      double tp_dist = sl_dist;

      if((InpLogic == LOGIC_MIRROR && g_prediction == 1) || (InpLogic == LOGIC_NORMAL && g_prediction == 0))
      {
         double price = SymbolInfoDouble(_Symbol, SYMBOL_BID);
         m_trade.Sell(InpLot, _Symbol, price, price + sl_dist, price - tp_dist, "AI SELL");
      }
      else
      {
         double price = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
         m_trade.Buy(InpLot, _Symbol, price, price - sl_dist, price + tp_dist, "AI BUY");
      }
   }
}

void GetData()
{
   // 3. DATA - Load OHLC prices
   ArraySetAsSeries(g_close, true); ArraySetAsSeries(g_open, true);
   ArraySetAsSeries(g_high, true);  ArraySetAsSeries(g_low, true);

   if(CopyClose(_Symbol, _Period, 0, WINDOW_SIZE + 15, g_close) < WINDOW_SIZE + 15 ||
      CopyOpen(_Symbol, _Period, 0, WINDOW_SIZE, g_open) < WINDOW_SIZE) return;
   
   CopyHigh(_Symbol, _Period, 0, WINDOW_SIZE, g_high);
   CopyLow(_Symbol, _Period, 0, WINDOW_SIZE, g_low);
}

void GetIndicators()
{
   // 4. INDICATORS - Get RSI and ATR values
   int rsi_handle = iRSI(_Symbol, _Period, 14, PRICE_CLOSE);
   ArraySetAsSeries(g_rsi_buffer, true);
   CopyBuffer(rsi_handle, 0, 0, WINDOW_SIZE, g_rsi_buffer);

   int atr_handle = iATR(_Symbol, _Period, InpATR);
   double atr_buffer[];
   ArraySetAsSeries(atr_buffer, true);
   CopyBuffer(atr_handle, 0, 0, 1, atr_buffer);
   g_current_atr = atr_buffer[0];
}

void BuildInputBuffer()
{
   // 5. INPUT BUFFER WITH NORMALIZATION BY _Digits
   ArrayResize(g_input_buffer, WINDOW_SIZE * FEATURES);
   
   // _Digits is the correct variable. If 5 or 3 decimals, we adjust to pips (x10).
   double pip_unit = _Point * (_Digits == 5 || _Digits == 3 ? 10 : 1);

   for(int i=0; i < WINDOW_SIZE; i++)
   {
      int mql_idx = WINDOW_SIZE - 1 - i;
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