//+------------------------------------------------------------------+
//|                                              SimpleONNX_EA.mq5   |
//+------------------------------------------------------------------+
#property strict

#include <Trade\Trade.mqh>

#resource "\\Files\\sp500_rates_m5_w20_f4_atr8_rsi8_minp1.5.onnx" as uchar ExtModel[];

//--- ENUMERATIONS
enum ENUM_LOGIC { LOGIC_NORMAL, LOGIC_MIRROR };

//--- INPUTS
input group "AI Config"
input ENUM_LOGIC InpLogic      = LOGIC_MIRROR;
input float      InpMinConf    = 0.55;
input int        InpStartHour  = 0;
input int        InpEndHour    = 23;
input string     InpModelName  = ""; // Model name (informational only)
input group "Risk"
input int        InpRSI        = 14;
input int        InpATRPeriod  = 14;
input double     InpLot        = 1;
input int        InpMagic      = 8812345688;
input int        InpATRSL      = 6;
input double     InpMultiplier = 1.1;

//--- GLOBAL VARIABLES
long     onnx_handle = INVALID_HANDLE;
CTrade   m_trade;
const int WINDOW_SIZE = 20;
const int FEATURES    = 3;
double session_start_balance = AccountInfoDouble(ACCOUNT_BALANCE);
string program_name = MQLInfoString(MQL_PROGRAM_NAME);

//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
int OnInit()
  {
   onnx_handle = OnnxCreateFromBuffer(ExtModel, ONNX_DEFAULT);
   if(onnx_handle == INVALID_HANDLE)
      return(INIT_FAILED);

   long input_shape[] = {1, 60}; // 20 candles * 3 attributes
   if(!OnnxSetInputShape(onnx_handle, 0, input_shape))
      return(INIT_FAILED);

   long out_shape_label[] = {1};
   OnnxSetOutputShape(onnx_handle, 0, out_shape_label);
   long out_shape_probs[] = {1, 2};
   OnnxSetOutputShape(onnx_handle, 1, out_shape_probs);

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
// 1. CORRECT TIME FILTER
   MqlDateTime dt;
   TimeCurrent(dt);
   bool valid_time = (dt.hour >= InpStartHour && dt.hour <= InpEndHour);

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

// 4. INDICATORS
   int rsi_handle = iRSI(_Symbol, _Period, InpRSI, PRICE_CLOSE);
   double rsi_buffer[];
   ArraySetAsSeries(rsi_buffer, true);
   CopyBuffer(rsi_handle, 0, 0, WINDOW_SIZE, rsi_buffer);

   int atr_handle = iATR(_Symbol, _Period, InpATRPeriod);
   double atr_buffer[];
   ArraySetAsSeries(atr_buffer, true);
   CopyBuffer(atr_handle, 0, 0, WINDOW_SIZE, atr_buffer);
   
   int atr_sl_handle = iATR(_Symbol, _Period, InpATRSL);
   double atr_sl_buffer[];
   ArraySetAsSeries(atr_sl_buffer, true);
   CopyBuffer(atr_sl_handle, 0, 0, 1, atr_sl_buffer);
   double current_atr = atr_sl_buffer[0];

// 5. INPUT BUFFER WITH ATR NORMALIZATION
   float input_buffer[];
   ArrayResize(input_buffer, WINDOW_SIZE * FEATURES);

   for(int i=0; i < WINDOW_SIZE; i++)
     {
      int mql_idx = WINDOW_SIZE - 1 - i;
      
      // Prevent division by zero
      double current_bar_atr = (atr_buffer[mql_idx] > 0) ? atr_buffer[mql_idx] : 0.0001;
      
      // Features normalized by ATR
      input_buffer[i * 3 + 0] = (float)((close[mql_idx] - open[mql_idx]) / current_bar_atr);
      input_buffer[i * 3 + 1] = (float)((high[mql_idx] - low[mql_idx]) / current_bar_atr);
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

   string prediction_str = (prediction == 1) ? "SELL" : "BUY";
   Print("Prediction: ", prediction_str, " | Confidence: ", confidence);

// 7. EXECUTION WITH TIME FILTER
   if(!PositionSelect(_Symbol) && valid_time && confidence >= InpMinConf)
     {
      double sl_dist = current_atr * InpMultiplier;
      double tp_dist = sl_dist * 1.5;

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

   double balance_diff = AccountInfoDouble(ACCOUNT_BALANCE) - session_start_balance;

   Comment("\n\nAI | Conf: ", DoubleToString(confidence*100, 2), "% / ",
           DoubleToString(InpMinConf*100, 2), "%",
           "\nPrediction: ", prediction_str,
           "\nSchedule: ", (valid_time ? "ACTIVE" : "RESTRICTED"),
           "\nSession P/ L: $", DoubleToString(balance_diff, 2));
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
