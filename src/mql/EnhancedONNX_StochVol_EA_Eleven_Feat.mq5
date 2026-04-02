//+------------------------------------------------------------------+
//|                              EnhancedONNX_StochVol_EA.mq5        |
//|                        Enhanced Stochastic & Volume Features     |
//|                        WITH TRAILING STOP SYSTEM                 |
//+------------------------------------------------------------------+
#property strict

#include <Trade\Trade.mqh>

#resource "\\Files\\sp500_m5_enh_w20_f10_atr14_minp0.5.onnx" as uchar ExtModel[];

//--- ENUMERATIONS
enum ENUM_LOGIC { LOGIC_NORMAL, LOGIC_MIRROR };
enum ENUM_TRAIL_MODE { TRAIL_FIXED_POINTS, TRAIL_ATR_BASED };

//--- INPUTS
input group "===== AI Configuration ====="
input ENUM_LOGIC InpLogic      = LOGIC_MIRROR;
input string     InpModelFile  = "sp500_m5_enh_w20_f10_atr14_minp0.5.onnx"; // Model File (In Tester, embed and recompile)
input float      InpMinConf    = 0.55;       // Minimum Confidence
input int        InpWindow     = 20;         // Window Size (--window)
input int        InpStartHour  = 0;          // Start Hour
input int        InpEndHour    = 23;         // End Hour
input bool       InpReverse    = false;      // BUY is SELL and SELL is BUY

input group "===== EMA Filter ====="
input int        InpEMAPeriod  = 9;          // EMA Period to filter entry
input bool       InpEmaGate    = true;       // EMA Gate to filter entry

input group "===== Risk Management ====="
input int        InpATRPeriod  = 14;         // ATR Period (--atr_period)
input double     InpLot        = 1.0;        // Order Lot Size
input int        InpMagic      = 8812772188; // Magic Number
input int        InpATRSL      = 6;          // ATR Period for SL
input double     InpMultiplier = 1.1;        // ATR Multiplier for TP

input group "===== TRAILING STOP Configuration ====="
input bool       InpUseTrailing      = true;               // Enable Trailing Stop
input ENUM_TRAIL_MODE InpTrailMode   = TRAIL_ATR_BASED;    // Trailing Mode
input int        InpTrailStart       = 5000;               // Trailing After Points (100pts=$1, so 5000pts=$50)
input int        InpTrailDistance    = 3000;               // Distance in Points (3000pts=$30)
input int        InpTrailStep        = 500;                // Minimum Step to Move SL (500pts=$5)
input bool       InpUseBreakeven     = true;               // Move to Breakeven
input int        InpBreakevenStart   = 3000;               // Breakeven After Profit (3000pts=$30)
input int        InpBreakevenOffset  = 200;                // Breakeven Offset (200pts=$2)

input group "===== Feature Parameters ====="
input int        InpStochPeriod = 10;        // Stoch Period (--stoch_window)
input int        InpStochK      = 3;         // Stoch K Period
input int        InpStochD      = 3;         // Stoch D Period
input int        InpVolWindow   = 10;        // Volume Window (--vol_window)

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

// Trailing stop tracking
struct TrailInfo
  {
   bool     breakeven_applied;
   double   highest_price;  // For buy positions
   double   lowest_price;   // For sell positions
  };
TrailInfo g_trail_info;

//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit()
  {
   Print("=================================================================");
   Print("  ENHANCED ONNX EA - Advanced Stochastic & Volume Features");
   Print("  WITH TRAILING STOP SYSTEM");
   Print("=================================================================");
   Print("Symbol: ", _Symbol, " | Period: ", _Period);
   Print("Features: ", FEATURES, " | Window: ", WINDOW_SIZE, " | Total inputs: ", WINDOW_SIZE * FEATURES);
   
   if(InpUseTrailing)
     {
      Print("Trailing Stop: ENABLED");
      Print("  Mode: ", (InpTrailMode == TRAIL_FIXED_POINTS ? "Fixed Points" : "ATR Based"));
      Print("  Start After: ", InpTrailStart, " points");
      Print("  Trail Distance: ", InpTrailDistance, " points");
      Print("  Trail Step: ", InpTrailStep, " points");
      
      // Calculate approximate dollar values for logging
      double point_value = _Point;
      double tick_value = SymbolInfoDouble(_Symbol, SYMBOL_TRADE_TICK_VALUE);
      double tick_size = SymbolInfoDouble(_Symbol, SYMBOL_TRADE_TICK_SIZE);
      double dollar_per_point = (tick_size > 0) ? (tick_value / tick_size) * point_value : 0;
      
      Print("  Point Value: ", point_value, " | ~$", DoubleToString(dollar_per_point * InpTrailStart, 2), " to start trailing");
      Print("  Trail Distance: ~$", DoubleToString(dollar_per_point * InpTrailDistance, 2));
      
      if(InpUseBreakeven)
         Print("  Breakeven: Start at ", InpBreakevenStart, " pts (~$", 
               DoubleToString(dollar_per_point * InpBreakevenStart, 2), 
               "), Offset ", InpBreakevenOffset, " pts");
     }
   else
      Print("Trailing Stop: DISABLED");

   if(MQLInfoInteger(MQL_TESTER)) {
      Print("Running in Strategy Tester");
      onnx_handle = OnnxCreateFromBuffer(ExtModel, ONNX_DEFAULT);
   } else {
      Print("Running Live/Demo | Model: ", InpModelFile);
      onnx_handle = OnnxCreate(InpModelFile, ONNX_DEFAULT);
   }

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
   
   // Initialize trailing info
   g_trail_info.breakeven_applied = false;
   g_trail_info.highest_price = 0;
   g_trail_info.lowest_price = 0;

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
   
   string trail_status = InpUseTrailing ? "ACTIVE" : "OFF";
   if(InpUseTrailing && PositionSelect(_Symbol))
     {
      double profit_pts = GetPositionProfitPoints();
      trail_status = "Active (" + DoubleToString(profit_pts, 0) + " pts profit)";
     }
   
   Comment("\n\n=== ENHANCED AI TRADING SYSTEM ===",
           "\nModel: ", InpModelFile,
           "\nFeatures: ", FEATURES, " x ", WINDOW_SIZE, " bars = ", FEATURES * WINDOW_SIZE,
           "\nPrediction: ", g_prediction_str,
           "\nConfidence: ", DoubleToString(g_confidence*100, 2), "% / ", DoubleToString(InpMinConf*100, 2), "%",
           "\nSchedule: ", (g_valid_time ? "ACTIVE" : "RESTRICTED"),
           "\nTrailing Stop: ", trail_status,
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
   
   // Feature 2: Volume momentum (current vs previous)
   if(idx < ArraySize(volumes) - 1)
      vol_momentum = (float)(((double)volumes[idx] - (double)volumes[idx + 1]) / 
                             ((double)volumes[idx + 1] > 0 ? (double)volumes[idx + 1] : 1.0));
   else
      vol_momentum = 0.0;
   
   // Feature 3: Volume-Price divergence
   double price_change = 0;
   if(idx < ArraySize(close) - 1)
      price_change = close[idx] - close[idx + 1];
   
   double vol_change = (double)volumes[idx] - (double)volumes[idx + 1];
   vol_price_div = (float)((price_change * vol_change) < 0 ? 1.0 : 0.0);
   
   // Feature 4: Volume percentile rank
   int count_below = 0;
   for(int i = 0; i < vol_window; i++)
     {
      if(volumes[idx + i] < volumes[idx])
         count_below++;
     }
   vol_percentile = (float)count_below / (float)vol_window;
   
   // Feature 5: Volume Z-score
   double vol_stddev = 0;
   for(int i = 0; i < vol_window; i++)
     {
      double diff = (double)volumes[idx + i] - vol_ma;
      vol_stddev += diff * diff;
     }
   vol_stddev = MathSqrt(vol_stddev / vol_window);
   
   if(vol_stddev > 0)
      vol_zscore = (float)(((double)volumes[idx] - vol_ma) / vol_stddev);
   else
      vol_zscore = 0.0;
   
   return true;
  }

//+------------------------------------------------------------------+
//| OnTick function                                                  |
//+------------------------------------------------------------------+
void OnTick()
  {
   // Check and manage trailing stop
   if(InpUseTrailing)
      ManageTrailingStop();
   
   // Time filter
   MqlDateTime dt;
   TimeCurrent(dt);
   bool valid_time = (dt.hour >= InpStartHour && dt.hour <= InpEndHour);
   g_valid_time = valid_time;

   // Check new bar
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
   
   Print(_Symbol, " | Prediction: ", prediction_str, (InpReverse ? "(R)" : ""), 
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
         if(m_trade.Sell(InpLot, _Symbol, price, price + sl_dist, price - tp_dist, 
                     program_name + " SELL@" + DoubleToString(price, _Digits)))
           {
            // Reset trailing info for new position
            g_trail_info.breakeven_applied = false;
            g_trail_info.highest_price = 0;
            g_trail_info.lowest_price = price;
           }
        }
      else
        {
         double price = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
         if(m_trade.Buy(InpLot, _Symbol, price, price - sl_dist, price + tp_dist, 
                    program_name + " BUY@" + DoubleToString(price, _Digits)))
           {
            // Reset trailing info for new position
            g_trail_info.breakeven_applied = false;
            g_trail_info.highest_price = price;
            g_trail_info.lowest_price = 0;
           }
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
//| Get position profit in points                                    |
//+------------------------------------------------------------------+
double GetPositionProfitPoints()
  {
   if(!PositionSelect(_Symbol))
      return 0;
   
   double open_price = PositionGetDouble(POSITION_PRICE_OPEN);
   double current_price;
   ENUM_POSITION_TYPE pos_type = (ENUM_POSITION_TYPE)PositionGetInteger(POSITION_TYPE);
   
   if(pos_type == POSITION_TYPE_BUY)
      current_price = SymbolInfoDouble(_Symbol, SYMBOL_BID);
   else
      current_price = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
   
   double profit_price = (pos_type == POSITION_TYPE_BUY) ? 
                         (current_price - open_price) : 
                         (open_price - current_price);
   
   return profit_price / _Point;
  }

//+------------------------------------------------------------------+
//| Manage Trailing Stop                                             |
//+------------------------------------------------------------------+
void ManageTrailingStop()
  {
   if(!PositionSelect(_Symbol))
      return;
   
   ulong ticket = PositionGetInteger(POSITION_TICKET);
   ENUM_POSITION_TYPE pos_type = (ENUM_POSITION_TYPE)PositionGetInteger(POSITION_TYPE);
   double open_price = PositionGetDouble(POSITION_PRICE_OPEN);
   double current_sl = PositionGetDouble(POSITION_SL);
   double current_tp = PositionGetDouble(POSITION_TP);
   
   double current_price;
   if(pos_type == POSITION_TYPE_BUY)
      current_price = SymbolInfoDouble(_Symbol, SYMBOL_BID);
   else
      current_price = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
   
   // Calculate profit in points
   double profit_points = GetPositionProfitPoints();
   
   // Calculate trailing distance based on mode
   double trail_distance_price;
   if(InpTrailMode == TRAIL_ATR_BASED)
     {
      int atr_handle = iATR(_Symbol, _Period, InpATRSL);
      double atr_buffer[];
      if(CopyBuffer(atr_handle, 0, 0, 1, atr_buffer) == 1)
        {
         // InpTrailDistance is multiplier x 10, so divide by 10
         trail_distance_price = atr_buffer[0] * (InpTrailDistance / 10.0);
        }
      else
        {
         trail_distance_price = InpTrailDistance * _Point;
        }
      IndicatorRelease(atr_handle);
     }
   else
     {
      trail_distance_price = InpTrailDistance * _Point;
     }
   
   double trail_start_price = InpTrailStart * _Point;
   double trail_step_price = InpTrailStep * _Point;
   
   // BREAKEVEN LOGIC
   if(InpUseBreakeven && !g_trail_info.breakeven_applied && profit_points >= InpBreakevenStart)
     {
      double new_sl;
      if(pos_type == POSITION_TYPE_BUY)
         new_sl = open_price + (InpBreakevenOffset * _Point);
      else
         new_sl = open_price - (InpBreakevenOffset * _Point);
      
      // Check if new SL is better than current
      bool should_modify = false;
      if(pos_type == POSITION_TYPE_BUY && (current_sl == 0 || new_sl > current_sl))
         should_modify = true;
      else if(pos_type == POSITION_TYPE_SELL && (current_sl == 0 || new_sl < current_sl))
         should_modify = true;
      
      if(should_modify)
        {
         new_sl = NormalizeDouble(new_sl, _Digits);
         if(m_trade.PositionModify(ticket, new_sl, current_tp))
           {
            g_trail_info.breakeven_applied = true;
            Print("[Breakeven] SL moved to breakeven + ", InpBreakevenOffset, " pts at ", new_sl);
           }
         return; // Exit after breakeven modification
        }
     }
   
   // TRAILING STOP LOGIC
   if(profit_points >= InpTrailStart)
     {
      double new_sl = 0;
      
      if(pos_type == POSITION_TYPE_BUY)
        {
         // Update highest price seen
         if(current_price > g_trail_info.highest_price || g_trail_info.highest_price == 0)
            g_trail_info.highest_price = current_price;
         
         // Calculate new SL based on highest price
         new_sl = g_trail_info.highest_price - trail_distance_price;
         
         // Check if new SL is better than current (higher)
         if(current_sl == 0 || (new_sl > current_sl + trail_step_price))
           {
            new_sl = NormalizeDouble(new_sl, _Digits);
            if(m_trade.PositionModify(ticket, new_sl, current_tp))
              {
               Print("[Trailing BUY] SL moved from ", current_sl, " to ", new_sl, 
                     " | High: ", g_trail_info.highest_price, " | Profit: ", profit_points, " pts");
              }
           }
        }
      else // SELL position
        {
         // Update lowest price seen
         if(current_price < g_trail_info.lowest_price || g_trail_info.lowest_price == 0)
            g_trail_info.lowest_price = current_price;
         
         // Calculate new SL based on lowest price
         new_sl = g_trail_info.lowest_price + trail_distance_price;
         
         // Check if new SL is better than current (lower)
         if(current_sl == 0 || (new_sl < current_sl - trail_step_price))
           {
            new_sl = NormalizeDouble(new_sl, _Digits);
            if(m_trade.PositionModify(ticket, new_sl, current_tp))
              {
               Print("[Trailing SELL] SL moved from ", current_sl, " to ", new_sl,
                     " | Low: ", g_trail_info.lowest_price, " | Profit: ", profit_points, " pts");
              }
           }
        }
     }
  }
//+------------------------------------------------------------------+
