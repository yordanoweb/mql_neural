//+------------------------------------------------------------------+
//|                                         Relaxed_M15_Trend_EA.mq5 |
//+------------------------------------------------------------------+
#property strict

#include <Trade\Trade.mqh>

//--- ENUMERATIONS
enum STOCH_SIGNAL { SIGNAL_NONE, SIGNAL_BUY, SIGNAL_SELL };

//--- INPUTS
input group "Stochastic (M15 Swing Settings)"
input int        InpKPeriod      = 14;    // Increased for macro swing tracking
input int        InpDPeriod      = 3;
input int        InpSlowing      = 3;
input double     InpStochBuyZone = 45.0;  // Relaxed: Allow buy crosses in lower half
input double     InpStochSellZone= 55.0;  // Relaxed: Allow sell crosses in upper half

input group "ADX (M15 Trend Settings)"
input int        InpADXPeriod    = 14;    // Standard smoothing for M15+
input double     InpADXLevel     = 20.0;  // Relaxed: Catch trends earlier

input group "Risk & Volatility"
input double     InpLot          = 1.0;
input ulong      InpMagic        = 8812345688;
input int        InpATR          = 14;    // Smooth volatility tracking
input double     InpMultiplier   = 1.5;

//--- GLOBAL VARIABLES
string program_name = MQLInfoString(MQL_PROGRAM_NAME);

//--- INDICATOR HANDLES
int g_atr_handle   = INVALID_HANDLE;
int g_adx_handle   = INVALID_HANDLE;
int g_stoch_handle = INVALID_HANDLE;

//--- TRADE OBJECT
CTrade g_trade;

//+------------------------------------------------------------------+
int OnInit()
  {
   g_trade.SetExpertMagicNumber(InpMagic);
   g_trade.SetDeviationInPoints(10);

   g_atr_handle = iATR(_Symbol, _Period, InpATR);
   if(g_atr_handle == INVALID_HANDLE)
     { Print("[ERROR] Cannot create ATR indicator"); return(INIT_FAILED); }

   g_adx_handle = iADX(_Symbol, _Period, InpADXPeriod);
   if(g_adx_handle == INVALID_HANDLE)
     { Print("[ERROR] Cannot create ADX indicator"); return(INIT_FAILED); }

   g_stoch_handle = iStochastic(_Symbol, _Period, InpKPeriod, InpDPeriod, InpSlowing, MODE_SMA, STO_LOWHIGH);
   if(g_stoch_handle == INVALID_HANDLE)
     { Print("[ERROR] Cannot create Stochastic indicator"); return(INIT_FAILED); }

   return(INIT_SUCCEEDED);
  }

//+------------------------------------------------------------------+
void OnDeinit(const int reason)
  {
   Comment("");
   if(g_atr_handle   != INVALID_HANDLE) IndicatorRelease(g_atr_handle);
   if(g_adx_handle   != INVALID_HANDLE) IndicatorRelease(g_adx_handle);
   if(g_stoch_handle != INVALID_HANDLE) IndicatorRelease(g_stoch_handle);
  }

//+------------------------------------------------------------------+
void OnTick()
  {
   if(!IsNewCandle()) return;

   // Dynamic arrays
   double stoch_k[], stoch_d[], adx_val[], di_plus[], di_minus[], atr_val[];

   // Set TimeSeries indexing: [0] is Bar 1 (closed), [1] is Bar 2
   ArraySetAsSeries(stoch_k, true);
   ArraySetAsSeries(stoch_d, true);
   ArraySetAsSeries(adx_val, true);
   ArraySetAsSeries(di_plus, true);
   ArraySetAsSeries(di_minus, true);
   ArraySetAsSeries(atr_val, true);

   // Fetch indicator data
   if(CopyBuffer(g_stoch_handle, 0, 1, 3, stoch_k)  != 3) { Print("[WARN] Stoch K not ready");  return; }
   if(CopyBuffer(g_stoch_handle, 1, 1, 3, stoch_d)  != 3) { Print("[WARN] Stoch D not ready");  return; }
   if(CopyBuffer(g_adx_handle,   0, 1, 3, adx_val)  != 3) { Print("[WARN] ADX Main not ready"); return; }
   if(CopyBuffer(g_adx_handle,   1, 1, 3, di_plus)  != 3) { Print("[WARN] DI+ not ready");       return; }
   if(CopyBuffer(g_adx_handle,   2, 1, 3, di_minus) != 3) { Print("[WARN] DI- not ready");       return; }
   if(CopyBuffer(g_atr_handle,   0, 1, 1, atr_val)  != 1) { Print("[WARN] ATR not ready");       return; }

   // Map variables to human-readable format
   double k1 = stoch_k[0];  // Bar 1
   double d1 = stoch_d[0];
   double k2 = stoch_k[1];  // Bar 2
   double d2 = stoch_d[1];

   double adx1 = adx_val[0]; // Bar 1
   double adx2 = adx_val[1]; // Bar 2
   
   double dip1 = di_plus[0];
   double dim1 = di_minus[0];
   double atr  = atr_val[0];

   // 1. Stochastic Crossover with Relaxed Pullback Filtering
   STOCH_SIGNAL signal = SIGNAL_NONE;
   if(k2 <= d2 && k1 > d1 && k1 <= InpStochBuyZone)   signal = SIGNAL_BUY;
   if(k2 >= d2 && k1 < d1 && k1 >= InpStochSellZone)  signal = SIGNAL_SELL;

   // 2. Optimized ADX/DMI Environment Rules
   bool adx_active = adx1 > InpADXLevel;
   bool adx_rising = adx1 > adx2;
   bool trend_up   = dip1 > dim1;
   bool trend_down = dim1 > dip1;

   // 3. Trade Routing Matrix
   if(signal == SIGNAL_BUY && adx_active && adx_rising && trend_up)
     {
      PrintFormat(">>> M15 Trend BUY Triggered. ADX: %.2f, Stoch K: %.2f", adx1, k1);
      PlaceOrder(ORDER_TYPE_BUY, atr);
     }
   else if(signal == SIGNAL_SELL && adx_active && adx_rising && trend_down)
     {
      PrintFormat(">>> M15 Trend SELL Triggered. ADX: %.2f, Stoch K: %.2f", adx1, k1);
      PlaceOrder(ORDER_TYPE_SELL, atr);
     }
  }

//+------------------------------------------------------------------+
bool HasOpenPosition()
  {
   for(int i = PositionsTotal() - 1; i >= 0; i--)
     {
      if(PositionGetTicket(i) == 0) continue;
      if(PositionGetString(POSITION_SYMBOL)   == _Symbol &&
         PositionGetInteger(POSITION_MAGIC)   == (long)InpMagic)
         return true;
     }
   return false;
  }

//+------------------------------------------------------------------+
void PlaceOrder(ENUM_ORDER_TYPE order_type, double atr)
  {
   if(HasOpenPosition()) return;
   if(atr <= 0) return;

   double spread       = SymbolInfoInteger(_Symbol, SYMBOL_SPREAD) * _Point;
   double atr_distance = atr * InpMultiplier;
   double price, sl, tp;

   if(order_type == ORDER_TYPE_BUY)
     {
      price = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
      sl = price - atr_distance - spread;
      tp = price + atr_distance;
      g_trade.Buy(InpLot, _Symbol, price, sl, tp, "M15_BUY");
     }
     
   else if(order_type == ORDER_TYPE_SELL)
     {
      price = SymbolInfoDouble(_Symbol, SYMBOL_BID);
      sl = price + atr_distance + spread;
      tp = price - atr_distance;
      g_trade.Sell(InpLot, _Symbol, price, sl, tp, "M15_SELL");
     }
  }

//+------------------------------------------------------------------+
bool IsNewCandle()
  {
   static datetime last_time = 0;
   datetime current_time = iTime(_Symbol, _Period, 0);
   if(current_time != last_time)
     {
      last_time = current_time;
      return true;
     }
   return false;
  }