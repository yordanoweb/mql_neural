//+------------------------------------------------------------------+
//|                                         SimpleONNX_3_Feat_Test.mq5 |
//+------------------------------------------------------------------+
#property strict

#include <Trade\Trade.mqh>

#resource "\\Files\\ndx100_rates_m5_w20_f4_vsa12_atr8_min0.5.onnx" as uchar InpOnnxFile[];

//--- ENUMERATIONS
enum ENUM_LOGIC { LOGIC_NORMAL, LOGIC_MIRROR };

//--- INPUTS
input group "AI Config"
input ENUM_LOGIC InpLogic       = LOGIC_MIRROR;
input float      InpMinConf     = 0.55;
input int        InpStartHour   = 0;
input int        InpEndHour     = 23;
input group "Model"
// Bars in the feature window — must match --window used during training (default: 20).
// Controls the ONNX input shape {1, InpWindow * 3}. Wrong value = garbage inference.
input int        InpWindow      = 20;
input group "Risk"
input double     InpLot         = 1;
input int        InpMagic       = 8812345688;
input int        InpATRPeriod   = 6;   // ATR Period (match training)
input double     InpMultiplier  = 1.1;  // SL = ATR * multiplier; TP = SL * 1.5
input int        InpVSAMAPeriod = 8;   // VSA Period (match training)

//--- GLOBAL VARIABLES
long      onnx_handle         = INVALID_HANDLE;
int       g_atr_handle        = INVALID_HANDLE;
CTrade    m_trade;
const int FEATURES            = 3;  // fixed: body, range, vsa
double    session_start_balance;
string    program_name;

// Counters for diagnostics printed in OnDeinit
int g_bars_processed  = 0;
int g_onnx_calls      = 0;
int g_trades_sent     = 0;

//+------------------------------------------------------------------+
int OnInit()
  {
   session_start_balance = AccountInfoDouble(ACCOUNT_BALANCE);
   program_name          = MQLInfoString(MQL_PROGRAM_NAME);
   g_bars_processed = 0;
   g_onnx_calls     = 0;
   g_trades_sent    = 0;

   //--- ONNX: runtime load — works in optimizer because all agents
   //    share the same MQL5/Files/ path.
   onnx_handle = OnnxCreateFromBuffer(InpOnnxFile, ONNX_DEFAULT);
   if(onnx_handle == INVALID_HANDLE)
     {
      Print("ERROR: OnnxCreate() failed. Error=", GetLastError(),
            " | Place the .onnx file in the terminal MQL5/Files/ folder.");
      return(INIT_FAILED);
     }

   long input_shape[] = {1, (long)InpWindow * FEATURES};
   if(!OnnxSetInputShape(onnx_handle, 0, input_shape))
     {
      Print("ERROR: OnnxSetInputShape failed. Error=", GetLastError());
      return(INIT_FAILED);
     }

   long out_shape_label[] = {1};
   long out_shape_probs[]  = {1, 2};
   if(!OnnxSetOutputShape(onnx_handle, 0, out_shape_label) ||
      !OnnxSetOutputShape(onnx_handle, 1, out_shape_probs))
     {
      Print("ERROR: OnnxSetOutputShape failed. Error=", GetLastError());
      return(INIT_FAILED);
     }

   g_atr_handle = iATR(_Symbol, _Period, InpATRPeriod);
   if(g_atr_handle == INVALID_HANDLE)
     {
      Print("ERROR: iATR failed. Error=", GetLastError());
      return(INIT_FAILED);
     }

   m_trade.SetExpertMagicNumber(InpMagic);
   Print("OnInit OK ",
         " | ATR=", InpATRPeriod, " | VSA_MA=", InpVSAMAPeriod,
         " | window=", InpWindow);
   return(INIT_SUCCEEDED);
  }

//+------------------------------------------------------------------+
void OnDeinit(const int reason)
  {
   // Summary printed at the end of every backtest pass — visible in Journal
   Print("=== SESSION SUMMARY | bars=", g_bars_processed,
         " | onnx_calls=", g_onnx_calls,
         " | trades_sent=", g_trades_sent, " ===");

   if(onnx_handle  != INVALID_HANDLE) { OnnxRelease(onnx_handle);       onnx_handle  = INVALID_HANDLE; }
   if(g_atr_handle != INVALID_HANDLE) { IndicatorRelease(g_atr_handle); g_atr_handle = INVALID_HANDLE; }
   Comment("");
  }

//+------------------------------------------------------------------+
void OnTick()
  {
//--- 1. TIME FILTER
   MqlDateTime dt;
   TimeCurrent(dt);
   bool valid_time = (dt.hour >= InpStartHour && dt.hour <= InpEndHour);

//--- 2. BAR CONTROL
   static datetime last_bar = 0;
   datetime current_bar = iTime(_Symbol, _Period, 0);
   if(current_bar == last_bar)
      return;
   last_bar = current_bar;
   g_bars_processed++;

//--- 3. BARS NEEDED
   int bars_needed = InpWindow + InpVSAMAPeriod + 10;

//--- 4. PRICE DATA (start=1: skip the still-forming bar 0)
   double close_arr[], open_arr[], high_arr[], low_arr[];
   ArraySetAsSeries(close_arr, true);
   ArraySetAsSeries(open_arr,  true);
   ArraySetAsSeries(high_arr,  true);
   ArraySetAsSeries(low_arr,   true);

   int got_close = CopyClose(_Symbol, _Period, 1, bars_needed, close_arr);
   int got_open  = CopyOpen (_Symbol, _Period, 1, bars_needed, open_arr);
   int got_high  = CopyHigh (_Symbol, _Period, 1, bars_needed, high_arr);
   int got_low   = CopyLow  (_Symbol, _Period, 1, bars_needed, low_arr);

   if(got_close < bars_needed || got_open < bars_needed ||
      got_high  < bars_needed || got_low  < bars_needed)
     {
      // Only print on the very first few bars to avoid log spam
      if(g_bars_processed <= 3)
         Print("SKIP bar ", g_bars_processed, ": price data not ready yet (",
               got_close, "/", bars_needed, " bars)");
      return;
     }

//--- 5. TICK VOLUME
//    In the Strategy Tester, CopyTickVolume with start_pos=1 can fail even when
//    price data is available. Use start_pos=0 and manually skip bar 0 by
//    copying one extra bar, then offsetting the index by 1 when reading.
//    This is the most reliable pattern across all MT5 backtest modes.
   long vol_arr[];
   ArraySetAsSeries(vol_arr, true);
   int got_vol = CopyTickVolume(_Symbol, _Period, 0, bars_needed + 1, vol_arr);
   if(got_vol < bars_needed + 1)
     {
      if(g_bars_processed <= 3)
         Print("SKIP bar ", g_bars_processed, ": volume data not ready (",
               got_vol, "/", bars_needed + 1, ")");
      return;
     }
   // vol_arr[0] = forming bar 0 (skip), vol_arr[1] = last closed bar, etc.
   // To align with close_arr/open_arr (which start at closed bar 1),
   // we read vol_arr with offset +1.

//--- 6. ATR BUFFER
   double atr_buf[];
   ArraySetAsSeries(atr_buf, true);
   int got_atr = CopyBuffer(g_atr_handle, 0, 0, bars_needed + 1, atr_buf);
   if(got_atr < bars_needed + 1)
     {
      if(g_bars_processed <= 3)
         Print("SKIP bar ", g_bars_processed, ": ATR buffer not ready (",
               got_atr, "/", bars_needed + 1, ")");
      return;
     }
   // Same offset logic: atr_buf[0]=forming bar, atr_buf[1]=last closed bar.

//--- 7. RAW VSA
//    Both vol_arr and high/low arrays need the same bar alignment.
//    price arrays: index k → closed bar (k+1) ago from now
//    vol_arr:      index k → bar k ago from now (0=forming)
//    So for price index k, the matching volume is vol_arr[k+1].
   double raw_vsa[];
   ArrayResize(raw_vsa, bars_needed);
   for(int k = 0; k < bars_needed; k++)
     {
      double spd = high_arr[k] - low_arr[k];
      // vol_arr[k+1] aligns with price arrays starting at closed bar 1
      raw_vsa[k] = (spd > _Point) ? (double)vol_arr[k + 1] / spd : 0.0;
     }

//--- 8. BUILD INPUT BUFFER
   float input_buffer[];
   ArrayResize(input_buffer, InpWindow * FEATURES);

   for(int i = 0; i < InpWindow; i++)
     {
      int mql_idx = InpWindow - 1 - i;  // i=0 → oldest bar in window

      // ATR: atr_buf[k+1] aligns with closed bar k in price arrays
      double atr_val = atr_buf[mql_idx + 1];
      if(atr_val <= 0.0) atr_val = 1e-10;

      double feat_body  = (close_arr[mql_idx] - open_arr[mql_idx]) / atr_val;
      double feat_range = (high_arr[mql_idx]  - low_arr[mql_idx])  / atr_val;

      // VSA rolling mean
      double vsa_sum = 0.0;
      int    vsa_cnt = 0;
      int    vsa_end = mql_idx + InpVSAMAPeriod;
      if(vsa_end > bars_needed) vsa_end = bars_needed;
      for(int m = mql_idx; m < vsa_end; m++)
         { vsa_sum += raw_vsa[m]; vsa_cnt++; }
      double vsa_mean = (vsa_cnt == InpVSAMAPeriod && vsa_sum > 0.0)
                        ? vsa_sum / vsa_cnt : 1.0;
      double feat_vsa = raw_vsa[mql_idx] / vsa_mean;

      // Clip [-20, 20]
      feat_body  = MathMax(-20.0, MathMin(20.0, feat_body));
      feat_range = MathMax(-20.0, MathMin(20.0, feat_range));
      feat_vsa   = MathMax(-20.0, MathMin(20.0, feat_vsa));

      input_buffer[i * FEATURES + 0] = (float)feat_body;
      input_buffer[i * FEATURES + 1] = (float)feat_range;
      input_buffer[i * FEATURES + 2] = (float)feat_vsa;
     }

//--- 9. INFERENCE
//    Use ONNX_DEFAULT so MT5 handles INT64→int coercion automatically.
//    With ONNX_DEFAULT the output label array must be long[].
//    With ONNX_NO_CONVERSION it would need to match the raw tensor type exactly,
//    which varies by skl2onnx version and is not worth the fragility.
   long  output_label[];
   float output_probs[];
   ArrayResize(output_label, 1);
   ArrayResize(output_probs, 2);

   g_onnx_calls++;
   if(!OnnxRun(onnx_handle, ONNX_DEFAULT, input_buffer, output_label, output_probs))
     {
      Print("ERROR: OnnxRun failed at bar ", g_bars_processed,
            ". Error=", GetLastError());
      return;
     }

   long  prediction      = output_label[0];
   float prob0           = output_probs[0];
   float prob1           = output_probs[1];
   float confidence      = (prediction == 1) ? prob1 : prob0;
   string prediction_str = (prediction == 1) ? "SELL" : "BUY";

   // Print every bar so you can see inference is running in the Journal
   Print("BAR ", g_bars_processed,
         " | pred=", prediction, " (", prediction_str, ")",
         " | p0=", DoubleToString(prob0, 4),
         " | p1=", DoubleToString(prob1, 4),
         " | conf=", DoubleToString(confidence, 4),
         " | threshold=", DoubleToString(InpMinConf, 4),
         " | valid_time=", valid_time,
         " | has_pos=", PositionSelect(_Symbol));

//--- 10. TRADE EXECUTION
   if(!PositionSelect(_Symbol) && valid_time && confidence >= InpMinConf)
     {
      double current_atr = atr_buf[1]; // closed bar ATR for sizing
      double sl_dist     = current_atr * InpMultiplier;
      double tp_dist     = sl_dist * 1.5;

      g_trades_sent++;
      if((InpLogic == LOGIC_MIRROR && prediction == 1) ||
         (InpLogic == LOGIC_NORMAL && prediction == 0))
        {
         double price = SymbolInfoDouble(_Symbol, SYMBOL_BID);
         m_trade.Sell(InpLot, _Symbol, price,
                      price + sl_dist, price - tp_dist,
                      program_name + " SELL@" + DoubleToString(price, _Digits));
        }
      else
        {
         double price = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
         m_trade.Buy(InpLot, _Symbol, price,
                     price - sl_dist, price + tp_dist,
                     program_name + " BUY@" + DoubleToString(price, _Digits));
        }
     }

   double balance_diff = AccountInfoDouble(ACCOUNT_BALANCE) - session_start_balance;
   Comment("\n\nAI | Conf: ",  DoubleToString(confidence * 100, 2), "% / ",
                               DoubleToString(InpMinConf * 100, 2), "%",
           "\nPrediction: ",   prediction_str,
           "\nBars: ",         g_bars_processed, " | ONNX calls: ", g_onnx_calls,
           "\nATR: ",          DoubleToString(atr_buf[1], _Digits + 1),
           "\nSchedule: ",     (valid_time ? "ACTIVE" : "RESTRICTED"),
           "\nSession P/L: $", DoubleToString(balance_diff, 2));
  }

//+------------------------------------------------------------------+
string GetPeriodString()
  {
   switch(_Period)
     {
      case PERIOD_M1:  return "M1";
      case PERIOD_M2:  return "M2";
      case PERIOD_M3:  return "M3";
      case PERIOD_M5:  return "M5";
      case PERIOD_M10: return "M10";
      case PERIOD_M15: return "M15";
      case PERIOD_M20: return "M20";
      case PERIOD_M30: return "M30";
      case PERIOD_H1:  return "H1";
      case PERIOD_H2:  return "H2";
      case PERIOD_H3:  return "H3";
      case PERIOD_H4:  return "H4";
      case PERIOD_D1:  return "D1";
      default:         return "Unknown";
     }
  }
//+------------------------------------------------------------------+
