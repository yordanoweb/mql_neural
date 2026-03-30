//+------------------------------------------------------------------+
//|                                              SimpleONNX_EA.mq5   |
//+------------------------------------------------------------------+
#property strict

#include <Trade\Trade.mqh>

#resource "\\Files\\ndx100_rates_m5_w20_f5_atr6_minp0.5.onnx" as uchar ExtModel[];

//--- ENUMERATIONS
enum ENUM_LOGIC { LOGIC_NORMAL, LOGIC_MIRROR };

//--- INPUTS
input group "AI Config"
input ENUM_LOGIC InpLogic      = LOGIC_MIRROR;
input string     InpModelFile  = "ndx100_rates_m5_w20_f5_atr6_minp0.5.onnx"; // Informational only
input float      InpMinConf    = 0.55;
input int        InpStartHour  = 0;
input int        InpEndHour    = 23;
input bool       InpReverse    = false; // BUY is SELL and SELL is BUY
input group "EMA"
input int        InpEMAPeriod  = 9;
input bool       InpEmaGate    = true;
input group "Risk"
input int        InpATRPeriod  = 14;
input double     InpLot        = 1;
input int        InpMagic      = 8812345688;
input int        InpATRSL      = 6;
input double     InpMultiplier = 1.1;

//--- GLOBAL VARIABLES
long     onnx_handle = INVALID_HANDLE;
CTrade   m_trade;
const int WINDOW_SIZE = 20;
const int FEATURES    = 4;
double session_start_balance = AccountInfoDouble(ACCOUNT_BALANCE);
string program_name = MQLInfoString(MQL_PROGRAM_NAME);
int      g_ema_handle  = INVALID_HANDLE;

//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
int OnInit()
  {
   onnx_handle = OnnxCreateFromBuffer(ExtModel, ONNX_DEFAULT);
   if(onnx_handle == INVALID_HANDLE)
      return(INIT_FAILED);

   long input_shape[] = {1, WINDOW_SIZE * FEATURES}; // 20 candles * 5 attributes
   if(!OnnxSetInputShape(onnx_handle, 0, input_shape))
      return(INIT_FAILED);

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

   m_trade.SetExpertMagicNumber(InpMagic);

   return(INIT_SUCCEEDED);
  }

//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
  {
   if(onnx_handle != INVALID_HANDLE)
      OnnxRelease(onnx_handle);
   Comment("");
  }

//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void OnTick()
  {
// --- 1. STATIC VARIABLES FOR PERSISTENCE ---
// Save the data of last prediction to use them between ticks
   static float s_confidence = 0;
   static string s_prediction_str = "WAITING...";

// --- 2. REAL TIME LOGIC (Se ejecuta en cada Tick) ---
// 1. THE RIGHT TIME FILTER
   MqlDateTime dt;
   TimeCurrent(dt);
   bool valid_time = (dt.hour >= InpStartHour && dt.hour <= InpEndHour);

// Calculate the balance difference in real time
   double balance_diff = AccountInfoDouble(ACCOUNT_BALANCE) - session_start_balance;

// UPDATE THE COMMENT (Now it updates in each tick)
   Comment("\n\nAI | Conf: ", DoubleToString(s_confidence*100, 2), "% / ",
           DoubleToString(InpMinConf*100, 2), "%", 
           "\nModel: ", InpModelFile,
           "\nPrediction: ", s_prediction_str,
           "\nSchedule: ", (valid_time ? "ACTIVE" : "RESTRICTED"),
           "\nSession P/ L: $", DoubleToString(balance_diff, 2));

// 2. BAR CONTROL
   static datetime last_bar = 0;
   datetime current_bar = iTime(_Symbol, _Period, 0);
   if(current_bar == last_bar)
      return;
   last_bar = current_bar;

// 3. DATA
   double close[], open[], high[], low[];
   ArraySetAsSeries(close, true);
   ArraySetAsSeries(open, true);
   ArraySetAsSeries(high, true);
   ArraySetAsSeries(low, true);

   if(CopyClose(_Symbol, _Period, 0, WINDOW_SIZE + 15, close) < WINDOW_SIZE + 15 ||
      CopyOpen(_Symbol, _Period, 0, WINDOW_SIZE, open) < WINDOW_SIZE ||
      CopyHigh(_Symbol, _Period, 0, WINDOW_SIZE, high) < WINDOW_SIZE ||
      CopyLow(_Symbol, _Period, 0, WINDOW_SIZE, low) < WINDOW_SIZE)
      return;

   int atr_handle = iATR(_Symbol, _Period, InpATRPeriod);
   double atr_buffer[];
   ArraySetAsSeries(atr_buffer, true);
   CopyBuffer(atr_handle, 0, 0, WINDOW_SIZE, atr_buffer);
   
   int atr_sl_handle = iATR(_Symbol, _Period, InpATRSL);
   double atr_sl_buffer[];
   ArraySetAsSeries(atr_sl_buffer, true);
   CopyBuffer(atr_sl_handle, 0, 0, 1, atr_sl_buffer);
   double current_atr = atr_sl_buffer[0];
   
   int stoch_handle = iStochastic(_Symbol, _Period, 14, 3, 3, MODE_SMA, STO_LOWHIGH);
   double stoch_main[], stoch_signal[];
   ArraySetAsSeries(stoch_main, true);
   ArraySetAsSeries(stoch_signal, true);
   CopyBuffer(stoch_handle, MAIN_LINE, 0, WINDOW_SIZE, stoch_main);
   CopyBuffer(stoch_handle, SIGNAL_LINE, 0, WINDOW_SIZE, stoch_signal);
   
   long volumes[];
   CopyTickVolume(_Symbol, _Period, 0, WINDOW_SIZE + 20, volumes);
   ArraySetAsSeries(volumes, true);

// 5. INPUT BUFFER WITH ATR NORMALIZATION
   float input_buffer[];
   ArrayResize(input_buffer, WINDOW_SIZE * FEATURES);

   for(int i=0; i < WINDOW_SIZE; i++)
     {
      int mql_idx = WINDOW_SIZE - 1 - i;
      
      // Prevent division by zero
      double current_bar_atr = (atr_buffer[mql_idx] > 0) ? atr_buffer[mql_idx] : 0.0001;
      
      // Features normalized by ATR
      input_buffer[i * 4 + 0] = (float)((close[mql_idx] - open[mql_idx]) / current_bar_atr);
      input_buffer[i * 4 + 1] = (float)((high[mql_idx] - low[mql_idx]) / current_bar_atr);
      
      // Feature 3: Stochastic Strategy (Main - Signal)
      input_buffer[i * 4 + 2] = (float)((stoch_main[mql_idx] - stoch_signal[mql_idx]) / 100.0);
      
      // Feature 4: Volume vs MA(20)
      double vol_sum = 0;
      for(int j=0; j<20; j++) vol_sum += (double)volumes[mql_idx + j];
      double vol_avg = vol_sum / 20.0;
      input_buffer[i * 4 + 3] = (float)((double)volumes[mql_idx] / (vol_avg > 0 ? vol_avg : 1.0));
     }

// 6. INFERENCE
   long output_label[];
   float output_probs[];
   ArrayResize(output_label, 1);
   ArrayResize(output_probs, 2);
   if(!OnnxRun(onnx_handle, ONNX_NO_CONVERSION, input_buffer, output_label, output_probs))
      return;

   long  prediction = output_label[0];
   Print("Raw prediction: ", prediction);

   if(InpReverse)
     {
      prediction = 1 - prediction;
     }
   float confidence  = (prediction == 1) ? output_probs[1] : output_probs[0];

   string prediction_str = (prediction == 1) ? "SELL" : "BUY";
   Print("Prediction: ", prediction_str, (InpReverse ? "(R)" : ""), " | Confidence: ", confidence);

   // Save the data in the static variables to use them in the next ticks
   s_confidence = confidence;
   s_prediction_str = prediction_str + (InpReverse ? "(R)" : "");

// 7. EXECUTION WITH TIME FILTER
   if(!PositionSelect(_Symbol) && valid_time && confidence >= InpMinConf)
     {
      double sl_dist = current_atr * InpMultiplier;
      double tp_dist = sl_dist * 1.5;

      if(InpEmaGate && !EMAGateAllows(prediction))
         return;

      if((InpLogic == LOGIC_MIRROR && prediction == 1) || (InpLogic == LOGIC_NORMAL && prediction == 0))
        {
         double price = SymbolInfoDouble(_Symbol, SYMBOL_BID);
         m_trade.Sell(InpLot, _Symbol, price, price + sl_dist, price - tp_dist, program_name + " SELL@" + DoubleToString(price, _Digits));
        }
      else
        {
         double price = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
         m_trade.Buy(InpLot, _Symbol, price, price - sl_dist, price + tp_dist, program_name + " BUY@" + DoubleToString(price, _Digits));
        }
     }
  }

//+------------------------------------------------------------------+
//| Check if price position agrees with predicted direction           |
//+------------------------------------------------------------------+
bool EMAGateAllows(int predicted_class)
  {
   double ema_gate[];
   if(CopyBuffer(g_ema_handle, 0, 0, 1, ema_gate) != 1)
      return false;

   if(predicted_class == 1)
      return SymbolInfoDouble(_Symbol, SYMBOL_ASK) > ema_gate[0];  // BUY: ask must be above EMA
   if(predicted_class == 2)
      return SymbolInfoDouble(_Symbol, SYMBOL_BID) < ema_gate[0];  // SELL: bid must be below EMA

   Print("EMA Gate does not allow the trade");
   Print("Trade direction: ", predicted_class == 1 ? "SELL" : "BUY");
   Print("EMA Gate: ", ema_gate[0]);
   Print("Price: ", SymbolInfoDouble(_Symbol, SYMBOL_ASK));
   return false;
  }

//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
string GetPeriodString()
  {
   ENUM_TIMEFRAMES period = _Period;
   switch(period)
     {
      case PERIOD_M1:
         return "M1";
      case PERIOD_M2:
         return "M2";
      case PERIOD_M3:
         return "M3";
      case PERIOD_M5:
         return "M5";
      case PERIOD_M10:
         return "M10";
      case PERIOD_M15:
         return "M15";
      case PERIOD_M20:
         return "M20";
      case PERIOD_M30:
         return "M30";
      case PERIOD_H1:
         return "H1";
      case PERIOD_H2:
         return "H2";
      case PERIOD_H3:
         return "H3";
      case PERIOD_H4:
         return "H4";
      case PERIOD_D1:
         return "D1";
      default:
         return "Unknown";
     }
  }
//+------------------------------------------------------------------+
