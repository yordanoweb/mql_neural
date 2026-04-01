//+------------------------------------------------------------------+
//|                              EnhancedONNX_StochVol_EA.mq5        |
//|                        Enhanced Stochastic & Volume Features     |
//+------------------------------------------------------------------+
#property strict

#include <Trade\Trade.mqh>

#resource "\\Files\\sp500_m5_enh_w20_f10_atr14_minp0.5.onnx" as uchar ExtModel[];

//--- ENUMERATIONS
enum ENUM_LOGIC { LOGIC_NORMAL, LOGIC_MIRROR };

//--- INPUTS
input group "===== AI Configuration ====="
input ENUM_LOGIC InpLogic      = LOGIC_MIRROR;
input string     InpModelFile  = "sp500_m5_enh_w20_f10_atr14_minp0.5.onnx"; // Informational only
input float      InpMinConf    = 0.55;
input int        InpWindow     = 20;
input int        InpStartHour  = 0;
input int        InpEndHour    = 23;
input bool       InpReverse    = false; // BUY is SELL and SELL is BUY

input group "===== EMA Filter ====="
input int        InpEMAPeriod  = 9;
input bool       InpEmaGate    = true;

input group "===== Risk Management ====="
input int        InpATRPeriod  = 14;
input double     InpLot        = 1.0;
input int        InpMagic      = 8812772188;
input int        InpATRSL      = 6;
input double     InpMultiplier = 1.1;

input group "===== Feature Parameters ====="
input int        InpStochPeriod = 14;
input int        InpStochK      = 3;
input int        InpStochD      = 3;
input int        InpVolWindow   = 20;

//--- GLOBAL VARIABLES
long     onnx_handle = INVALID_HANDLE;
CTrade   m_trade;
const int WINDOW_SIZE = InpWindow;
const int FEATURES    = 11; // Enhanced: 2 basic + 4 stoch + 5 volume
double session_start_balance = AccountInfoDouble(ACCOUNT_BALANCE);
string program_name = MQLInfoString(MQL_PROGRAM_NAME);
int      g_ema_handle  = INVALID_HANDLE;
int      g_stoch_handle = INVALID_HANDLE;
float  g_confidence = 0;
string g_prediction_str = "WAITING...";
bool   g_valid_time = false;

//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit()
  {
   Print("=================================================================");
   Print("  ENHANCED ONNX EA - Advanced Stochastic & Volume Features");
   Print("=================================================================");
   Print("Features: ", FEATURES, " | Window: ", WINDOW_SIZE, " | Total inputs: ", WINDOW_SIZE * FEATURES);
   
   onnx_handle = OnnxCreateFromBuffer(ExtModel, ONNX_DEFAULT);
   if(onnx_handle == INVALID_HANDLE)
     {
      Print("[ERROR] Cannot load ONNX model");
      return(INIT_FAILED);
     }

   long input_shape[] = {1, WINDOW_SIZE * FEATURES};
   if(!OnnxSetInputShape(onnx_handle, 0, input_shape))
     {
      Print("[ERROR] Cannot set input shape");
      return(INIT_FAILED);
     }

   long out_shape_label[] = {1};
   OnnxSetOutputShape(onnx_handle, 0, out_shape_label);
   long out_shape_probs[] = {1, 2};
   OnnxSetOutputShape(onnx_handle, 1, out_shape_probs);

   g_ema_handle = iMA(_Symbol, _Period, InpEMAPeriod, 0, MODE_EMA, PRICE_CLOSE);
   if(g_ema_handle == INVALID_HANDLE)
     {
      Print("[ERROR] Cannot create EMA indicator");
      return INIT_FAILED;
     }
   
   g_stoch_handle = iStochastic(_Symbol, _Period, InpStochPeriod, InpStochK, InpStochD, MODE_SMA, STO_LOWHIGH);
   if(g_stoch_handle == INVALID_HANDLE)
     {
      Print("[ERROR] Cannot create Stochastic indicator");
      return INIT_FAILED;
     }

   m_trade.SetExpertMagicNumber(InpMagic);

   EventSetTimer(60);
   OnTimer();
   
   Print("[SUCCESS] EA initialized successfully");

   return(INIT_SUCCEEDED);
  }

//+------------------------------------------------------------------+
//| Expert deinitialization function                                 |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
  {
   if(onnx_handle != INVALID_HANDLE)
      OnnxRelease(onnx_handle);
   if(g_ema_handle != INVALID_HANDLE)
      IndicatorRelease(g_ema_handle);
   if(g_stoch_handle != INVALID_HANDLE)
      IndicatorRelease(g_stoch_handle);
   Comment("");
   EventKillTimer();
  }

//+------------------------------------------------------------------+
//| Timer function                                                    |
//+------------------------------------------------------------------+
void OnTimer()
  {
   double balance_diff = AccountInfoDouble(ACCOUNT_BALANCE) - session_start_balance;
   Comment("\n\n=== ENHANCED AI TRADING SYSTEM ===",
           "\nModel: ", InpModelFile,
           "\nFeatures: ", FEATURES, " x ", WINDOW_SIZE, " bars = ", FEATURES * WINDOW_SIZE,
           "\nPrediction: ", g_prediction_str,
           "\nConfidence: ", DoubleToString(g_confidence*100, 2), "% / ", DoubleToString(InpMinConf*100, 2), "%",
           "\nSchedule: ", (g_valid_time ? "ACTIVE" : "RESTRICTED"),
           "\nSession P/L: $", DoubleToString(balance_diff, 2));
  }

//+------------------------------------------------------------------+
//| Calculate enhanced stochastic features                           |
//+------------------------------------------------------------------+
bool CalculateStochasticFeatures(const int idx, 
                                  const double &stoch_main[], 
                                  const double &stoch_signal[],
                                  float &momentum, 
                                  float &position, 
                                  float &velocity, 
                                  float &divergence)
  {
   double k = stoch_main[idx];
   double d = stoch_signal[idx];
   
   // Feature 1: Momentum (K - D) normalized to [-1, 1]
   momentum = (float)((k - d) / 100.0);
   
   // Feature 2: Position - where is K in its range (centered around 50)
   position = (float)((k - 50.0) / 50.0);
   
   // Feature 3: Velocity (rate of change of K)
   if(idx < ArraySize(stoch_main) - 1)
     {
      double k_prev = stoch_main[idx + 1];
      velocity = (float)((k - k_prev) / 100.0);
     }
   else
      velocity = 0.0;
   
   // Feature 4: Divergence pressure zones
   float overbought_pressure = (k > 80) ? (float)(-(k - 80) / 20.0) : 0.0;
   float oversold_pressure = (k < 20) ? (float)((20 - k) / 20.0) : 0.0;
   divergence = overbought_pressure + oversold_pressure;
   
   return true;
  }

//+------------------------------------------------------------------+
//| Calculate enhanced volume features                               |
//+------------------------------------------------------------------+
bool CalculateVolumeFeatures(const int idx,
                              const long &volumes[],
                              const double &close[],
                              const int vol_window,
                              float &vol_ratio,
                              float &vol_momentum,
                              float &vol_price_div,
                              float &vol_percentile,
                              float &vol_zscore)
  {
   if(idx + vol_window >= ArraySize(volumes))
      return false;
   
   // Calculate volume MA
   double vol_sum = 0;
   for(int i = 0; i < vol_window; i++)
      vol_sum += (double)volumes[idx + i];
   double vol_ma = vol_sum / vol_window;
   
   // Feature 1: Volume ratio
   vol_ratio = (float)((double)volumes[idx] / (vol_ma > 0 ? vol_ma : 1.0));
   
   // Feature 2: Volume momentum (EMA fast vs slow)
   double vol_ema_fast = 0, vol_ema_slow = 0;
   double alpha_fast = 2.0 / (5.0 + 1.0);
   double alpha_slow = 2.0 / (20.0 + 1.0);
   
   vol_ema_fast = (double)volumes[idx];
   vol_ema_slow = (double)volumes[idx];
   
   for(int i = 1; i < MathMin(20, ArraySize(volumes) - idx); i++)
     {
      vol_ema_fast = alpha_fast * (double)volumes[idx + i] + (1 - alpha_fast) * vol_ema_fast;
      vol_ema_slow = alpha_slow * (double)volumes[idx + i] + (1 - alpha_slow) * vol_ema_slow;
     }
   
   vol_momentum = (float)((vol_ema_fast - vol_ema_slow) / (vol_ema_slow > 0 ? vol_ema_slow : 1.0));
   
   // Feature 3: Volume-Price divergence
   if(idx < ArraySize(close) - 1 && idx < ArraySize(volumes) - 1)
     {
      double price_change = MathAbs((close[idx] - close[idx + 1]) / close[idx + 1]);
      double vol_change = MathAbs((double)(volumes[idx] - volumes[idx + 1]) / (double)volumes[idx + 1]);
      vol_price_div = (float)(vol_change - price_change);
     }
   else
      vol_price_div = 0.0;
   
   // Feature 4: Volume percentile
   int count_below = 0;
   for(int i = 0; i < vol_window; i++)
     {
      if(volumes[idx + i] < volumes[idx])
         count_below++;
     }
   double percentile = (double)count_below / vol_window;
   vol_percentile = (float)((percentile - 0.5) * 2.0);
   
   // Feature 5: Z-score
   double vol_std = 0;
   for(int i = 0; i < vol_window; i++)
     {
      double diff = (double)volumes[idx + i] - vol_ma;
      vol_std += diff * diff;
     }
   vol_std = MathSqrt(vol_std / vol_window);
   
   double zscore = ((double)volumes[idx] - vol_ma) / (vol_std > 0 ? vol_std : 1.0);
   zscore = MathMax(-3.0, MathMin(3.0, zscore)); // Clip to [-3, 3]
   vol_zscore = (float)(zscore / 3.0);
   
   return true;
  }

//+------------------------------------------------------------------+
//| Expert tick function                                             |
//+------------------------------------------------------------------+
void OnTick()
  {
   // --- TIME FILTER ---
   MqlDateTime dt;
   TimeCurrent(dt);
   bool valid_time = (dt.hour >= InpStartHour && dt.hour <= InpEndHour);
   g_valid_time = valid_time;

   // --- BAR CONTROL ---
   static datetime last_bar = 0;
   datetime current_bar = iTime(_Symbol, _Period, 0);
   if(current_bar == last_bar)
      return;
   last_bar = current_bar;

   // --- PREPARE DATA ARRAYS ---
   double close[], open[], high[], low[];
   ArraySetAsSeries(close, true);
   ArraySetAsSeries(open, true);
   ArraySetAsSeries(high, true);
   ArraySetAsSeries(low, true);

   int required_bars = WINDOW_SIZE + InpVolWindow + 10;
   
   if(CopyClose(_Symbol, _Period, 0, required_bars, close) < required_bars ||
      CopyOpen(_Symbol, _Period, 0, required_bars, open) < required_bars ||
      CopyHigh(_Symbol, _Period, 0, required_bars, high) < required_bars ||
      CopyLow(_Symbol, _Period, 0, required_bars, low) < required_bars)
     {
      Print("[ERROR] Cannot copy price data");
      return;
     }

   // --- ATR FOR NORMALIZATION ---
   int atr_handle = iATR(_Symbol, _Period, InpATRPeriod);
   double atr_buffer[];
   ArraySetAsSeries(atr_buffer, true);
   if(CopyBuffer(atr_handle, 0, 0, WINDOW_SIZE, atr_buffer) < WINDOW_SIZE)
     {
      IndicatorRelease(atr_handle);
      return;
     }
   
   // --- ATR FOR SL/TP ---
   int atr_sl_handle = iATR(_Symbol, _Period, InpATRSL);
   double atr_sl_buffer[];
   ArraySetAsSeries(atr_sl_buffer, true);
   CopyBuffer(atr_sl_handle, 0, 0, 1, atr_sl_buffer);
   double current_atr = atr_sl_buffer[0];
   
   // --- STOCHASTIC ---
   double stoch_main[], stoch_signal[];
   ArraySetAsSeries(stoch_main, true);
   ArraySetAsSeries(stoch_signal, true);
   if(CopyBuffer(g_stoch_handle, MAIN_LINE, 0, WINDOW_SIZE + 1, stoch_main) < WINDOW_SIZE + 1 ||
      CopyBuffer(g_stoch_handle, SIGNAL_LINE, 0, WINDOW_SIZE + 1, stoch_signal) < WINDOW_SIZE + 1)
     {
      Print("[ERROR] Cannot copy Stochastic data");
      IndicatorRelease(atr_handle);
      IndicatorRelease(atr_sl_handle);
      return;
     }
   
   // --- VOLUME ---
   long volumes[];
   ArraySetAsSeries(volumes, true);
   if(CopyTickVolume(_Symbol, _Period, 0, required_bars, volumes) < required_bars)
     {
      Print("[ERROR] Cannot copy volume data");
      IndicatorRelease(atr_handle);
      IndicatorRelease(atr_sl_handle);
      return;
     }

   // --- BUILD INPUT BUFFER WITH ENHANCED FEATURES ---
   float input_buffer[];
   ArrayResize(input_buffer, WINDOW_SIZE * FEATURES);
   
   for(int i = 0; i < WINDOW_SIZE; i++)
     {
      int mql_idx = WINDOW_SIZE - 1 - i;
      
      // Prevent division by zero
      double current_bar_atr = (atr_buffer[mql_idx] > 0) ? atr_buffer[mql_idx] : 0.0001;
      
      // BASIC FEATURES (2)
      float feat_body = (float)((close[mql_idx] - open[mql_idx]) / current_bar_atr);
      float feat_range = (float)((high[mql_idx] - low[mql_idx]) / current_bar_atr);
      
      // STOCHASTIC FEATURES (4)
      float stoch_momentum, stoch_position, stoch_velocity, stoch_divergence;
      if(!CalculateStochasticFeatures(mql_idx, stoch_main, stoch_signal,
                                      stoch_momentum, stoch_position, stoch_velocity, stoch_divergence))
        {
         Print("[ERROR] Cannot calculate stochastic features for bar ", i);
         IndicatorRelease(atr_handle);
         IndicatorRelease(atr_sl_handle);
         return;
        }
      
      // VOLUME FEATURES (5)
      float vol_ratio, vol_momentum, vol_price_div, vol_percentile, vol_zscore;
      if(!CalculateVolumeFeatures(mql_idx, volumes, close, InpVolWindow,
                                  vol_ratio, vol_momentum, vol_price_div, vol_percentile, vol_zscore))
        {
         Print("[ERROR] Cannot calculate volume features for bar ", i);
         IndicatorRelease(atr_handle);
         IndicatorRelease(atr_sl_handle);
         return;
        }
      
      // PACK ALL FEATURES (must match Python order!)
      int base_idx = i * FEATURES;
      input_buffer[base_idx + 0] = feat_body;              // 1
      input_buffer[base_idx + 1] = feat_range;             // 2
      input_buffer[base_idx + 2] = stoch_momentum;         // 3
      input_buffer[base_idx + 3] = stoch_position;         // 4
      input_buffer[base_idx + 4] = stoch_velocity;         // 5
      input_buffer[base_idx + 5] = stoch_divergence;       // 6
      input_buffer[base_idx + 6] = vol_ratio;              // 7
      input_buffer[base_idx + 7] = vol_momentum;           // 8
      input_buffer[base_idx + 8] = vol_price_div;          // 9
      input_buffer[base_idx + 9] = vol_percentile;         // 10
      input_buffer[base_idx + 10] = vol_zscore;            // 11
     }

   IndicatorRelease(atr_handle);
   IndicatorRelease(atr_sl_handle);

   // --- ONNX INFERENCE ---
   long output_label[];
   float output_probs[];
   ArrayResize(output_label, 1);
   ArrayResize(output_probs, 2);
   
   if(!OnnxRun(onnx_handle, ONNX_NO_CONVERSION, input_buffer, output_label, output_probs))
     {
      Print("[ERROR] ONNX inference failed");
      return;
     }

   long  prediction = output_label[0];
   
   if(InpReverse)
      prediction = 1 - prediction;
      
   float confidence  = (prediction == 1) ? output_probs[1] : output_probs[0];
   string prediction_str = (prediction == 1) ? "SELL" : "BUY";
   
   g_confidence = confidence;
   g_prediction_str = prediction_str + (InpReverse ? "(R)" : "");
   
   Print("Prediction: ", prediction_str, (InpReverse ? "(R)" : ""), 
         " | Confidence: ", DoubleToString(confidence * 100, 2), "%");

   // --- EXECUTION WITH FILTERS ---
   if(!PositionSelect(_Symbol) && valid_time && confidence >= InpMinConf)
     {
      double sl_dist = current_atr * InpMultiplier;
      double tp_dist = sl_dist * 1.5;

      bool is_sell = (InpLogic == LOGIC_MIRROR && prediction == 1) || 
                     (InpLogic == LOGIC_NORMAL && prediction == 0);

      if(InpEmaGate && !EMAGateAllows(is_sell))
         return;

      if(is_sell)
        {
         double price = SymbolInfoDouble(_Symbol, SYMBOL_BID);
         m_trade.Sell(InpLot, _Symbol, price, price + sl_dist, price - tp_dist, 
                     program_name + " SELL@" + DoubleToString(price, _Digits));
        }
      else
        {
         double price = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
         m_trade.Buy(InpLot, _Symbol, price, price - sl_dist, price + tp_dist, 
                    program_name + " BUY@" + DoubleToString(price, _Digits));
        }
     }
  }

//+------------------------------------------------------------------+
//| EMA Gate filter                                                  |
//+------------------------------------------------------------------+
bool EMAGateAllows(bool is_sell)
  {
   double ema_gate[];
   if(CopyBuffer(g_ema_handle, 0, 0, 1, ema_gate) != 1)
      return false;
   
   double ema_value = ema_gate[0];

   if(!is_sell)
     {
      double ask = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
      if(ask > ema_value)
        {
         Print("[EMA Gate] BUY allowed (Ask=", ask, " > EMA=", ema_value, ")");
         return true;
        }
      Print("[EMA Gate] BUY blocked (Ask=", ask, " <= EMA=", ema_value, ")");
      return false;
     }
   
   double bid = SymbolInfoDouble(_Symbol, SYMBOL_BID);
   if(bid < ema_value)
     {
      Print("[EMA Gate] SELL allowed (Bid=", bid, " < EMA=", ema_value, ")");
      return true;
     }
   Print("[EMA Gate] SELL blocked (Bid=", bid, " >= EMA=", ema_value, ")");
   return false;
  }
//+------------------------------------------------------------------+
