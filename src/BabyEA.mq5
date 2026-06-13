//+------------------------------------------------------------------+
//|                                  NASDAQ_Pure_PriceAction_EA.mq5 |
//+------------------------------------------------------------------+
#property strict

#include <Trade\Trade.mqh>

//--- INPUTS
input group "Price Action Breakout Trigger"
input int        InpPALookback   = 3;     // Number of bars to check for the breakout box (e.g., 3-5)

input group "ADX Filter Settings"
input int        InpADXPeriod    = 14;    // Standard macro smoothing
input double     InpADXLevel     = 20.0;  // Minimum trend intensity baseline

input group "Risk & Volatility"
input double     InpLot          = 1.0;
input ulong      InpMagic        = 8812345688;
input int        InpATR          = 14;    // Volatility tracking for SL/TP
input double     InpMultiplier   = 1.5;

input group "Time Filter (Broker Time)"
input bool       InpUseTimeFilter= true;  
input int        InpStartHour    = 15;    // US Open push
input int        InpEndHour      = 21;    // Avoid end-of-day chop

input group "Trailing Stop Engine"
input bool       InpUseTrailing  = true;  
input int        InpTrailingStop = 500;   // Distance from current price in Points
input int        InpTrailingStep = 100;   // Minimum step to modify SL in Points

//--- GLOBAL VARIABLES
string program_name = MQLInfoString(MQL_PROGRAM_NAME);

//--- INDICATOR HANDLES
int g_atr_handle   = INVALID_HANDLE;
int g_adx_handle   = INVALID_HANDLE;

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

   return(INIT_SUCCEEDED);
  }

//+------------------------------------------------------------------+
void OnDeinit(const int reason)
  {
   Comment("");
   if(g_atr_handle   != INVALID_HANDLE) IndicatorRelease(g_atr_handle);
   if(g_adx_handle   != INVALID_HANDLE) IndicatorRelease(g_adx_handle);
  }

//+------------------------------------------------------------------+
void OnTick()
  {
   // Trailing stop processes on every single tick
   if(InpUseTrailing)
     {
      ManageTrailingStop();
     }

   // Strategy Entry checks execute ONLY on Candle Close
   if(!IsNewCandle()) return;
   
   // Session protection filter
   if(!IsTimeAllowed()) return;

   // Dynamic arrays for trend tracking
   double adx_val[], di_plus[], di_minus[], atr_val[];

   // Set TimeSeries indexing
   ArraySetAsSeries(adx_val, true);
   ArraySetAsSeries(di_plus, true);
   ArraySetAsSeries(di_minus, true);
   ArraySetAsSeries(atr_val, true);

   // Fetch data arrays (starting at bar 1)
   if(CopyBuffer(g_adx_handle,   0, 1, 3, adx_val)  != 3) return;
   if(CopyBuffer(g_adx_handle,   1, 1, 3, di_plus)  != 3) return;
   if(CopyBuffer(g_adx_handle,   2, 1, 3, di_minus) != 3) return;
   if(CopyBuffer(g_atr_handle,   0, 1, 1, atr_val)  != 1) return;

   double adx1 = adx_val[0]; double adx2 = adx_val[1];
   double dip1 = di_plus[0]; double dim1 = di_minus[0];
   double atr  = atr_val[0];

   // 1. PURE PRICE ACTION BREAKOUT CALCULATIONS
   // Find highest high and lowest low of the lookback period STARTING FROM BAR 2
   int highest_idx = iHighest(_Symbol, _Period, MODE_HIGH, InpPALookback, 2);
   int lowest_idx  = iLowest(_Symbol, _Period, MODE_LOW, InpPALookback, 2);
   
   if(highest_idx < 0 || lowest_idx < 0) return; // Guard clause for history sync
   
   double target_high = iHigh(_Symbol, _Period, highest_idx);
   double target_low  = iLow(_Symbol, _Period, lowest_idx);
   
   // Fetch Close price of Bar 1 (the bar that just completely closed)
   double close1 = iClose(_Symbol, _Period, 1);

   bool pa_buy_breakout  = (close1 > target_high);
   bool pa_sell_breakout = (close1 < target_low);

   // 2. TREND FILTERS (Ensuring we breakout WITH the macro movement)
   bool adx_active = adx1 > InpADXLevel;
   bool adx_rising = adx1 > adx2;
   bool trend_up   = dip1 > dim1;
   bool trend_down = dim1 > dip1;

   // 3. EXECUTION ROUTING
   if(pa_buy_breakout && adx_active && adx_rising && trend_up)
     {
      PlaceOrder(ORDER_TYPE_BUY, atr);
     }
   else if(pa_sell_breakout && adx_active && adx_rising && trend_down)
     {
      PlaceOrder(ORDER_TYPE_SELL, atr);
     }
  }

//+------------------------------------------------------------------+
bool IsTimeAllowed()
  {
   if(!InpUseTimeFilter) return true;
   MqlDateTime current_time;
   TimeCurrent(current_time);
   return (current_time.hour >= InpStartHour && current_time.hour < InpEndHour);
  }

//+------------------------------------------------------------------+
void ManageTrailingStop()
  {
   for(int i = PositionsTotal() - 1; i >= 0; i--)
     {
      if(!PositionGetTicket(i)) continue;
      if(PositionGetString(POSITION_SYMBOL) != _Symbol || PositionGetInteger(POSITION_MAGIC) != (long)InpMagic) continue;
      
      ENUM_POSITION_TYPE type = (ENUM_POSITION_TYPE)PositionGetInteger(POSITION_TYPE);
      double current_sl       = PositionGetDouble(POSITION_SL);
      double open_price       = PositionGetDouble(POSITION_PRICE_OPEN);
      
      if(type == POSITION_TYPE_BUY)
        {
         double bid = SymbolInfoDouble(_Symbol, SYMBOL_BID);
         if(bid - open_price > InpTrailingStop * _Point)
           {
            double target_sl = bid - InpTrailingStop * _Point;
            if(target_sl > current_sl + (InpTrailingStep * _Point) || current_sl == 0)
              {
               g_trade.PositionModify(PositionGetInteger(POSITION_TICKET), target_sl, PositionGetDouble(POSITION_TP));
              }
           }
        }
      else if(type == POSITION_TYPE_SELL)
        {
         double ask = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
         if(open_price - ask > InpTrailingStop * _Point)
           {
            double target_sl = ask + InpTrailingStop * _Point;
            if(target_sl < current_sl - (InpTrailingStep * _Point) || current_sl == 0)
              {
               g_trade.PositionModify(PositionGetInteger(POSITION_TICKET), target_sl, PositionGetDouble(POSITION_TP));
              }
           }
        }
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
      g_trade.Buy(InpLot, _Symbol, price, sl, tp, "NSDQ_PA_BREAKOUT");
     }
   else if(order_type == ORDER_TYPE_SELL)
     {
      price = SymbolInfoDouble(_Symbol, SYMBOL_BID);
      sl = price + atr_distance + spread;
      tp = price - atr_distance;
      g_trade.Sell(InpLot, _Symbol, price, sl, tp, "NSDQ_PA_BREAKOUT");
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