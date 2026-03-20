//+------------------------------------------------------------------+
//|                                           EA_SGRADT60_ONNX.mq5   |
//|                                      SGRADT 6.0 - 10 Features    |
//+------------------------------------------------------------------+
#property copyright "SGRADT 6.0"
#property version   "6.00"
#property strict

#include <Trade\Trade.mqh>

//--- Input Parameters

//=== AI Model Configuration ===
input group "======== AI MODEL ========"
input string InpModelName = "EUR_USD_H1_SGRADT60_combined.onnx";  // Model filename
input string InpMetaFile  = "EUR_USD_H1_SGRADT60_combined.meta.json"; // Metadata file (optional)
input double InpMinConf   = 0.55;      // Minimum confidence (0.0-1.0)
input int    InpWindowSize = 20;       // Window size (must match training)
input int    InpFeaturesPerBar = 10;   // Features per bar (ALWAYS 10 for SGRADT 6.0)

//=== Inference Timing ===
input group "======== INFERENCE ========"
input int  InpInferSeconds = 15;       // Inference frequency (0 = new bar only)
input bool InpOneTradePerBar = true;   // Limit to 1 trade per bar

//=== Trading Session ===
input group "======== SESSION ========"
input int InpStartHour = 0;            // Session start hour (0-23)
input int InpEndHour   = 24;           // Session end hour (0-24)

//=== Indicator Parameters (SGRADT 6.0 Defaults) ===
input group "======== STOCHASTIC ========"
input int    InpStochK          = 7;      // Stochastic %K period
input int    InpStochD          = 3;      // Stochastic %D smoothing
input int    InpStochSlowing    = 3;      // Stochastic slowing
input double InpStochOversold   = 20.0;   // Oversold level
input double InpStochOverbought = 80.0;   // Overbought level

input group "======== ADX ========"
input int    InpADXPeriod = 8;      // ADX period
input double InpADXLimit  = 32.0;   // ADX trend threshold

input group "======== RSI ========"
input int InpRSIPeriod = 14;        // RSI period

input group "======== MACD ========"
input int InpMACDFast   = 12;       // MACD fast period
input int InpMACDSlow   = 26;       // MACD slow period
input int InpMACDSignal = 9;        // MACD signal period

input group "======== ATR (NEW) ========"
input int InpATRPeriod = 14;        // ATR period

//=== Risk Management ===
input group "======== RISK ========"
input double InpLot        = 1.0;   // Lot size
input int    InpMagic      = 6060;  // Magic number
input double InpStopPoints = 50.0;  // Stop Loss in POINTS
input double InpTakePoints = 100.0; // Take Profit in POINTS

//=== Display ===
input group "======== DISPLAY ========"
input bool InpShowPanel = true;     // Show information panel

//--- Global Variables
CTrade trade;
long   g_onnx_handle = INVALID_HANDLE;
int    g_adx_handle  = INVALID_HANDLE;
int    g_stoch_handle = INVALID_HANDLE;
int    g_rsi_handle   = INVALID_HANDLE;
int    g_macd_handle  = INVALID_HANDLE;
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
   Print("\n", StringRepeat("=", 70));
   Print("    SGRADT 6.0 - AI TRADING SYSTEM (10 Features)");
   Print(StringRepeat("=", 70), "\n");
   
   //--- Load ONNX model
   string model_path = InpModelName;
   Print("Loading ONNX model: ", model_path);
   
   g_onnx_handle = OnnxCreate(InpModelName, ONNX_DEFAULT);
   
   if(g_onnx_handle == INVALID_HANDLE) {
      Print("[ERROR] Cannot load ONNX model from resource");
      Print("        Make sure the model is in MQL5/Files/ folder");
      return INIT_FAILED;
   }
   
   Print("[OK] ONNX model loaded successfully");
   
   //--- Set input shape
   int num_inputs = InpFeaturesPerBar * InpWindowSize;
   long input_shape[] = {1, num_inputs};
   
   if(!OnnxSetInputShape(g_onnx_handle, 0, input_shape)) {
      Print("[ERROR] Cannot set input shape");
      return INIT_FAILED;
   }
   
   Print("[OK] Input shape set: [1, ", num_inputs, "] (", InpWindowSize, " bars x ", InpFeaturesPerBar, " features)");
   
   //--- Create indicator handles
   Print("\nInitializing indicators...");
   
   g_adx_handle = iADX(_Symbol, _Period, InpADXPeriod);
   if(g_adx_handle == INVALID_HANDLE) {
      Print("[ERROR] Cannot create ADX indicator");
      return INIT_FAILED;
   }
   
   g_stoch_handle = iStochastic(_Symbol, _Period,
                                InpStochK, InpStochD, InpStochSlowing,
                                MODE_SMA, STO_LOWHIGH);
   if(g_stoch_handle == INVALID_HANDLE) {
      Print("[ERROR] Cannot create Stochastic indicator");
      return INIT_FAILED;
   }
   
   g_rsi_handle = iRSI(_Symbol, _Period, InpRSIPeriod, PRICE_CLOSE);
   if(g_rsi_handle == INVALID_HANDLE) {
      Print("[ERROR] Cannot create RSI indicator");
      return INIT_FAILED;
   }
   
   g_macd_handle = iMACD(_Symbol, _Period, InpMACDFast, InpMACDSlow, InpMACDSignal, PRICE_CLOSE);
   if(g_macd_handle == INVALID_HANDLE) {
      Print("[ERROR] Cannot create MACD indicator");
      return INIT_FAILED;
   }
   
   g_atr_handle = iATR(_Symbol, _Period, InpATRPeriod);
   if(g_atr_handle == INVALID_HANDLE) {
      Print("[ERROR] Cannot create ATR indicator");
      return INIT_FAILED;
   }
   
   Print("[OK] Indicators created:");
   Print("     - ADX(", InpADXPeriod, ")");
   Print("     - Stochastic(", InpStochK, ",", InpStochD, ",", InpStochSlowing, ")");
   Print("     - RSI(", InpRSIPeriod, ") [NEW]");
   Print("     - MACD(", InpMACDFast, ",", InpMACDSlow, ",", InpMACDSignal, ") [NEW]");
   Print("     - ATR(", InpATRPeriod, ") [NEW]");
   
   //--- Initialize trade
   trade.SetExpertMagicNumber(InpMagic);
   trade.SetDeviationInPoints(10);
   trade.SetTypeFilling(ORDER_FILLING_FOK);
   trade.SetAsyncMode(false);
   
   //--- Print configuration
   PrintConfiguration();
   
   ArrayInitialize(g_last_probas, 0.0);
   
   Print("\n", StringRepeat("=", 70));
   Print("    EA INITIALIZED SUCCESSFULLY");
   Print(StringRepeat("=", 70), "\n");
   
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
   
   if(g_rsi_handle != INVALID_HANDLE)
      IndicatorRelease(g_rsi_handle);
   
   if(g_macd_handle != INVALID_HANDLE)
      IndicatorRelease(g_macd_handle);
   
   if(g_atr_handle != INVALID_HANDLE)
      IndicatorRelease(g_atr_handle);
   
   Comment("");
   
   Print("\n", StringRepeat("=", 70));
   Print("    EA DEINITIALIZED");
   Print(StringRepeat("=", 70), "\n");
}

//+------------------------------------------------------------------+
//| Expert tick function                                              |
//+------------------------------------------------------------------+
void OnTick()
{
   //--- Check session
   if(!IsWithinSession()) {
      if(InpShowPanel)
         UpdatePanel();
      return;
   }
   
   //--- Determine if should run inference
   bool should_infer = false;
   datetime current_bar = iTime(_Symbol, _Period, 0);
   
   if(InpInferSeconds == 0) {
      // New bar only mode
      if(current_bar != g_last_bar_time) {
         should_infer = true;
         g_last_bar_time = current_bar;
      }
   }
   else {
      // Time-based mode
      if(TimeCurrent() >= g_last_inference_time + InpInferSeconds) {
         should_infer = true;
         g_last_inference_time = TimeCurrent();
      }
   }
   
   //--- Run inference
   if(should_infer) {
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
   if(InpOneTradePerBar && current_bar == g_last_trade_bar) {
      return;
   }
   
   //--- Prepare input
   float input_buffer[];
   if(!PrepareInput(input_buffer)) {
      Print("⚠ Warning: Cannot prepare input data");
      return;
   }
   
   //--- Run inference
   float output_proba[];
   long  output_label[];
   
   if(!OnnxRun(g_onnx_handle, ONNX_NO_CONVERSION, input_buffer, output_proba, output_label)) {
      Print("❌ ERROR: ONNX inference failed");
      return;
   }
   
   //--- Store results
   for(int i = 0; i < 3; i++)
      g_last_probas[i] = output_proba[i];
   
   g_last_prediction = (int)output_label[0];
   
   //--- Get max probability
   double max_prob = g_last_probas[0];
   int predicted_class = 0;
   
   for(int i = 1; i < 3; i++) {
      if(g_last_probas[i] > max_prob) {
         max_prob = g_last_probas[i];
         predicted_class = i;
      }
   }
   
   //--- Check confidence
   if(max_prob < InpMinConf) {
      return;
   }
   
   //--- Execute trade
   if(predicted_class == 1) {  // BUY
      if(!HasPosition(POSITION_TYPE_BUY)) {
         double sl = SymbolInfoDouble(_Symbol, SYMBOL_BID) - InpStopPoints * _Point;
         double tp = SymbolInfoDouble(_Symbol, SYMBOL_BID) + InpTakePoints * _Point;
         
         if(trade.Buy(InpLot, _Symbol, 0, sl, tp, "SGRADT60_BUY")) {
            Print("[BUY] Order opened | Confidence: ", DoubleToString(max_prob * 100, 2), "%");
            g_last_trade_bar = current_bar;
         }
      }
   }
   else if(predicted_class == 2) {  // SELL
      if(!HasPosition(POSITION_TYPE_SELL)) {
         double sl = SymbolInfoDouble(_Symbol, SYMBOL_ASK) + InpStopPoints * _Point;
         double tp = SymbolInfoDouble(_Symbol, SYMBOL_ASK) - InpTakePoints * _Point;
         
         if(trade.Sell(InpLot, _Symbol, 0, sl, tp, "SGRADT60_SELL")) {
            Print("[SELL] Order opened | Confidence: ", DoubleToString(max_prob * 100, 2), "%");
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
   
   ArrayResize(input_buffer, total_size);
   ArrayInitialize(input_buffer, 0.0);
   
   //--- Get price data
   double open[], high[], low[], close[];
   ArraySetAsSeries(open, true);
   ArraySetAsSeries(high, true);
   ArraySetAsSeries(low, true);
   ArraySetAsSeries(close, true);
   
   if(CopyOpen(_Symbol, _Period, 0, window, open) != window) return false;
   if(CopyHigh(_Symbol, _Period, 0, window, high) != window) return false;
   if(CopyLow(_Symbol, _Period, 0, window, low) != window) return false;
   if(CopyClose(_Symbol, _Period, 0, window, close) != window) return false;
   
   //--- Get indicator data
   double adx_b[], pdi_b[], mdi_b[];
   double stoch_k_b[], stoch_d_b[];
   double rsi_b[];
   double macd_main[], macd_signal[];
   double atr_b[];
   
   ArraySetAsSeries(adx_b, true);
   ArraySetAsSeries(pdi_b, true);
   ArraySetAsSeries(mdi_b, true);
   ArraySetAsSeries(stoch_k_b, true);
   ArraySetAsSeries(stoch_d_b, true);
   ArraySetAsSeries(rsi_b, true);
   ArraySetAsSeries(macd_main, true);
   ArraySetAsSeries(macd_signal, true);
   ArraySetAsSeries(atr_b, true);
   
   if(CopyBuffer(g_adx_handle, 0, 0, window, adx_b) != window) return false;
   if(CopyBuffer(g_adx_handle, 1, 0, window, pdi_b) != window) return false;
   if(CopyBuffer(g_adx_handle, 2, 0, window, mdi_b) != window) return false;
   
   if(CopyBuffer(g_stoch_handle, 0, 0, window, stoch_k_b) != window) return false;
   if(CopyBuffer(g_stoch_handle, 1, 0, window, stoch_d_b) != window) return false;
   
   if(CopyBuffer(g_rsi_handle, 0, 0, window, rsi_b) != window) return false;
   
   if(CopyBuffer(g_macd_handle, 0, 0, window, macd_main) != window) return false;
   if(CopyBuffer(g_macd_handle, 1, 0, window, macd_signal) != window) return false;
   
   if(CopyBuffer(g_atr_handle, 0, 0, window, atr_b) != window) return false;
   
   //--- Fill buffer (CRITICAL: Order must match training script)
   for(int i = 0; i < window; i++) {
      int offset = i * InpFeaturesPerBar;
      
      // Feature order MUST match training script:
      // 0: body, 1: range, 2: stoch_k, 3: stoch_d, 4: rsi
      // 5: adx, 6: pdi, 7: mdi, 8: macd_hist, 9: atr_pct
      
      input_buffer[offset + 0] = (float)(close[i] - open[i]);           // body
      input_buffer[offset + 1] = (float)(high[i] - low[i]);             // range
      input_buffer[offset + 2] = (float)stoch_k_b[i];                   // stoch_main
      input_buffer[offset + 3] = (float)stoch_d_b[i];                   // stoch_signal
      input_buffer[offset + 4] = (float)rsi_b[i];                       // rsi [NEW]
      input_buffer[offset + 5] = (float)adx_b[i];                       // adx
      input_buffer[offset + 6] = (float)pdi_b[i];                       // pdi
      input_buffer[offset + 7] = (float)mdi_b[i];                       // mdi
      input_buffer[offset + 8] = (float)(macd_main[i] - macd_signal[i]); // macd_hist [NEW]
      input_buffer[offset + 9] = (float)((atr_b[i] / close[i]) * 100);  // atr_pct [NEW]
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
   
   if(InpStartHour <= InpEndHour) {
      return (dt.hour >= InpStartHour && dt.hour < InpEndHour);
   }
   else {
      return (dt.hour >= InpStartHour || dt.hour < InpEndHour);
   }
}

//+------------------------------------------------------------------+
//| Check if position exists                                          |
//+------------------------------------------------------------------+
bool HasPosition(ENUM_POSITION_TYPE type)
{
   for(int i = PositionsTotal() - 1; i >= 0; i--) {
      if(PositionGetSymbol(i) == _Symbol && 
         PositionGetInteger(POSITION_MAGIC) == InpMagic &&
         PositionGetInteger(POSITION_TYPE) == type) {
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
   
   //--- Symbol & Session
   panel += "SYMBOL: " + _Symbol + " [" + EnumToString(_Period) + "]\n";
   panel += "SESSION: " + IntegerToString(InpStartHour, 2, '0') + ":00-" + 
            IntegerToString(InpEndHour, 2, '0') + ":00";
   panel += IsWithinSession() ? " [ACTIVE]\n" : " [CLOSED]\n";
   
   string mode = (InpInferSeconds == 0) ? "NEW BAR" : IntegerToString(InpInferSeconds) + "s";
   panel += "MODE: " + mode + " | Inferences: " + IntegerToString(g_inference_count) + "\n";
   
   MqlDateTime dt;
   TimeToStruct(g_last_inference_time, dt);
   panel += StringFormat("Last Run: %02d:%02d:%02d\n", dt.hour, dt.min, dt.sec);
   
   //--- Indicators
   double adx[], pdi[], mdi[], stoch_k[], stoch_d[], rsi[], macd_main[], macd_signal[], atr[];
   
   CopyBuffer(g_adx_handle, 0, 0, 1, adx);
   CopyBuffer(g_adx_handle, 1, 0, 1, pdi);
   CopyBuffer(g_adx_handle, 2, 0, 1, mdi);
   CopyBuffer(g_stoch_handle, 0, 0, 1, stoch_k);
   CopyBuffer(g_stoch_handle, 1, 0, 1, stoch_d);
   CopyBuffer(g_rsi_handle, 0, 0, 1, rsi);
   CopyBuffer(g_macd_handle, 0, 0, 1, macd_main);
   CopyBuffer(g_macd_handle, 1, 0, 1, macd_signal);
   CopyBuffer(g_atr_handle, 0, 0, 1, atr);
   
   panel += StringRepeat("-", 52) + "\n";
   panel += "ADX INDICATOR (Period: " + IntegerToString(InpADXPeriod) + ")\n";
   panel += StringRepeat("-", 52) + "\n";
   panel += "   ADX: " + DoubleToString(adx[0], 2);
   panel += (adx[0] > InpADXLimit) ? " [TRENDING]\n" : " [RANGING]\n";
   panel += "   +DI: " + DoubleToString(pdi[0], 2) + " | " + " -DI: " + DoubleToString(mdi[0], 2) + "\n";
   
   panel += StringRepeat("-", 52) + "\n";
   panel += "STOCHASTIC (" + IntegerToString(InpStochK) + "," + 
            IntegerToString(InpStochD) + "," + IntegerToString(InpStochSlowing) + ")\n";
   panel += StringRepeat("-", 52) + "\n";
   panel += "   %K: " + DoubleToString(stoch_k[0], 2) + " | " + " %D: " + DoubleToString(stoch_d[0], 2) + "\n";
   
   string zone = "";
   if(stoch_k[0] <= InpStochOversold) zone = "OVERSOLD";
   else if(stoch_k[0] >= InpStochOverbought) zone = "OVERBOUGHT";
   else zone = "NEUTRAL";
   panel += "   Zone: " + zone + "\n";
   
   string cross = (stoch_k[0] > stoch_d[0]) ? "%K ABOVE %D" : "%K BELOW %D";
   panel += "   Cross: " + cross + "\n";
   
   // RSI
   panel += StringRepeat("-", 52) + "\n";
   panel += "RSI (Period: " + IntegerToString(InpRSIPeriod) + ") [NEW]\n";
   panel += StringRepeat("-", 52) + "\n";
   panel += "   RSI: " + DoubleToString(rsi[0], 2);
   if(rsi[0] < 30) panel += " [OVERSOLD]\n";
   else if(rsi[0] > 70) panel += " [OVERBOUGHT]\n";
   else panel += " [NEUTRAL]\n";
   
   // MACD
   double macd_hist = macd_main[0] - macd_signal[0];
   panel += StringRepeat("-", 52) + "\n";
   panel += "MACD (" + IntegerToString(InpMACDFast) + "," + 
            IntegerToString(InpMACDSlow) + "," + IntegerToString(InpMACDSignal) + ") [NEW]\n";
   panel += StringRepeat("-", 52) + "\n";
   panel += "   Histogram: " + DoubleToString(macd_hist, 5);
   panel += (macd_hist > 0) ? " [BULLISH]\n" : " [BEARISH]\n";
   
   // ATR
   double close_price = SymbolInfoDouble(_Symbol, SYMBOL_BID);
   double atr_pct = (atr[0] / close_price) * 100;
   panel += StringRepeat("-", 52) + "\n";
   panel += "ATR (Period: " + IntegerToString(InpATRPeriod) + ") [NEW]\n";
   panel += StringRepeat("-", 52) + "\n";
   panel += "   ATR: " + DoubleToString(atr[0], _Digits) + "\n";
   panel += "   ATR%: " + DoubleToString(atr_pct, 2) + "%\n";
   
   //--- AI Prediction
   panel += StringRepeat("=", 22) + "\n";
   panel += "AI PREDICTION\n";
   panel += StringRepeat("=", 22) + "\n";
   
   if(g_last_prediction >= 0) {
      string signal_text = "";
      if(g_last_prediction == 0) signal_text = "HOLD";
      else if(g_last_prediction == 1) signal_text = "BUY";
      else signal_text = "SELL";
      
      panel += "   Signal: " + signal_text + "\n";
      
      panel += "   Confidence Levels:\n";
      panel += "   - HOLD:  " + DoubleToString(g_last_probas[0] * 100, 2) + "%\n";
      panel += "   - BUY:   " + DoubleToString(g_last_probas[1] * 100, 2) + "%\n";
      panel += "   - SELL:  " + DoubleToString(g_last_probas[2] * 100, 2) + "%\n";
      
      panel += "   Minimum Required: " + DoubleToString(InpMinConf * 100, 1) + "%\n";
   }
   else {
      panel += "   Waiting for first inference...\n";
   }
   
   //--- Risk Settings
   panel += StringRepeat("-", 52) + "\n";
   panel += "RISK SETTINGS\n";
   panel += StringRepeat("-", 52) + "\n";
   panel += "   Lot Size: " + DoubleToString(InpLot, 2) + "\n";
   panel += "   Stop Loss:   " + DoubleToString(InpStopPoints, 0) + " pts (" + 
            DoubleToString(InpStopPoints * _Point, _Digits) + ")\n";
   panel += "   Take Profit: " + DoubleToString(InpTakePoints, 0) + " pts (" + 
            DoubleToString(InpTakePoints * _Point, _Digits) + ")\n";
   
   //--- Position Info
   double pnl = 0;
   string pos_info = "";
   
   for(int i = PositionsTotal() - 1; i >= 0; i--) {
      if(PositionGetSymbol(i) == _Symbol && PositionGetInteger(POSITION_MAGIC) == InpMagic) {
         pnl = PositionGetDouble(POSITION_PROFIT);
         ENUM_POSITION_TYPE type = (ENUM_POSITION_TYPE)PositionGetInteger(POSITION_TYPE);
         pos_info = (type == POSITION_TYPE_BUY) ? "BUY" : "SELL";
         break;
      }
   }
   
   if(pos_info != "") {
      panel += StringRepeat("=", 52) + "\n";
      panel += "ACTIVE POSITION: " + pos_info + "\n";
      string pnl_str = (pnl >= 0) ? "+" : "";
      panel += "   P&L: " + pnl_str + DoubleToString(pnl, 2) + " " + AccountInfoString(ACCOUNT_CURRENCY) + "\n";
      panel += StringRepeat("=", 52) + "\n";
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
   Print("SL: ", InpStopPoints, " points = ", DoubleToString(InpStopPoints * _Point, _Digits), " price");
   Print("TP: ", InpTakePoints, " points = ", DoubleToString(InpTakePoints * _Point, _Digits), " price");
   
   Print("\n=== INDICATOR PARAMETERS ===");
   Print("Stochastic: (", InpStochK, ",", InpStochD, ",", InpStochSlowing, ") [", InpStochOversold, "/", InpStochOverbought, "]");
   Print("ADX: Period=", InpADXPeriod, ", Limit=", InpADXLimit);
   Print("RSI: Period=", InpRSIPeriod, " [NEW]");
   Print("MACD: (", InpMACDFast, ",", InpMACDSlow, ",", InpMACDSignal, ") [NEW]");
   Print("ATR: Period=", InpATRPeriod, " [NEW]");
   
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
//| Helper: Center string                                             |
//+------------------------------------------------------------------+
string StringCenter(string text, int width)
{
   int len = StringLen(text);
   if(len >= width) return text;
   
   int padding = (width - len) / 2;
   return StringRepeat(" ", padding) + text + StringRepeat(" ", width - len - padding);
}

//+------------------------------------------------------------------+
