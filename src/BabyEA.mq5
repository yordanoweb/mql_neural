//+------------------------------------------------------------------+
//|                                              SimpleONNX_EA.mq5   |
//+------------------------------------------------------------------+
#property strict

#include <Trade\Trade.mqh>

//--- ENUMERATIONS
enum ENUM_LOGIC { LOGIC_NORMAL, LOGIC_MIRROR };

enum STOCH_SIGNAL { SIGNAL_NONE, SIGNAL_BUY, SIGNAL_SELL };

enum ADX_SIGNAL_STRENGTH { ADX_SIGNAL_NONE, ADX_SIGNAL_STRONG, ADX_SIGNAL_WEAK };
enum ADX_SIGNAL_SIDE { ADX_SIDE_NONE, ADX_SIDE_BUY, ADX_SIDE_SELL };

//--- INPUTS
input group "Risk"
input double     InpLot        = 1;
input int        InpMagic      = 8812345688;
input int        InpATR        = 6;
input double     InpMultiplier = 1.1;

//--- GLOBAL VARIABLES
double session_start_balance = AccountInfoDouble(ACCOUNT_BALANCE);
string program_name = MQLInfoString(MQL_PROGRAM_NAME);

//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
int OnInit()
  {
   return(INIT_SUCCEEDED);
  }

//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
  {
   Comment("");
  }

//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void OnTick()
  {
   if(!IsNewCandle())
     {
      return;
     }

   STOCH_SIGNAL stoch_signal = GetStochasticSignal();
   ADX_SIGNAL_STRENGTH adx_strength = GetADXSignalStrength();
   ADX_SIGNAL_SIDE adx_side = GetADXSignalSide();

   if(stoch_signal == SIGNAL_BUY && adx_strength == ADX_SIGNAL_STRONG && adx_side == ADX_SIDE_BUY)
     {
      //--- Place buy order
      Print("Placing BUY order");
     }
   else if(stoch_signal == SIGNAL_SELL && adx_strength == ADX_SIGNAL_STRONG && adx_side == ADX_SIDE_SELL)
     {
      //--- Place sell order
      Print("Placing SELL order");
     }
  }

//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
STOCH_SIGNAL GetStochasticSignal()
  {
   STOCH_SIGNAL stoch_signal = SIGNAL_NONE;

   float stoch_k_1 = GetStochK(5, 3, 3, 1);
   float stoch_d_1 = GetStochD(5, 3, 3, 1);
   float stoch_k_2 = GetStochK(5, 3, 3, 2);
   float stoch_d_2 = GetStochD(5, 3, 3, 2);

   if(stoch_k_2 < stoch_d_2 && stoch_k_1 > stoch_d_1)
     {
      //--- Buy signal
      stoch_signal = SIGNAL_BUY;
      Print("Buy signal detected");
     }

   if(stoch_k_2 > stoch_d_2 && stoch_k_1 < stoch_d_1)
     {
      //--- Sell signal
      stoch_signal = SIGNAL_SELL;
      Print("Sell signal detected");
     }

   return stoch_signal;
  }

ADX_SIGNAL_STRENGTH GetADXSignalStrength()
  {
   float adx_value = GetADX(8, 1);
   if(adx_value > 25)
     {
      return ADX_SIGNAL_STRONG;
     }
   else if(adx_value < 25)
     {
      return ADX_SIGNAL_WEAK;
     }
   else
     {
      return ADX_SIGNAL_NONE;
     }
  }

ADX_SIGNAL_SIDE GetADXSignalSide()
  {
   float di_plus = GetDIPlus(8, 1);
   float di_minus = GetDIMinus(8, 1);

   if(di_plus > di_minus)
     {
      return ADX_SIDE_BUY;
     }
   else if(di_plus < di_minus)
     {
      return ADX_SIDE_SELL;
     }
   else
     {
      return ADX_SIDE_NONE;
     }
  }

//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
float GetADX(int period, int shift = 0)
  {
   int adx_handle = 0, copied = 0;
   double adx_b[];

   adx_handle = iADX(_Symbol, _Period, period);
   if(adx_handle == INVALID_HANDLE)
     {
      Print("[ERROR] Cannot create ADX indicator");
      return 0;
     }

   copied = CopyBuffer(adx_handle, 0, 0, 1, adx_b);
   if(copied != 1)
     {
      Print("[ERROR] CopyBuffer ADX failed");
      return 0;
     }

   return adx_b[0];
  }

//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
float GetDIPlus(int period, int shift = 0)
  {
   int adx_handle = 0, copied = 0;
   double di_b[];

   adx_handle = iADX(_Symbol, _Period, period);
   if(adx_handle == INVALID_HANDLE)
     {
      Print("[ERROR] Cannot create ADX indicator");
      return 0;
     }

   copied = CopyBuffer(adx_handle, 1, 0, 1, di_b);
   if(copied != 1)
     {
      Print("[ERROR] CopyBuffer DI+ failed");
      return 0;
     }

   return di_b[0];
  }

//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
float GetDIMinus(int period, int shift = 0)
  {
   int adx_handle = 0, copied = 0;
   double di_b[];

   adx_handle = iADX(_Symbol, _Period, period);
   if(adx_handle == INVALID_HANDLE)
     {
      Print("[ERROR] Cannot create ADX indicator");
      return 0;
     }

   copied = CopyBuffer(adx_handle, 2, 0, 1, di_b);
   if(copied != 1)
     {
      Print("[ERROR] CopyBuffer DI- failed");
      return 0;
     }

   return di_b[0];
  }

//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
float GetStochK(int k_period, int d_period = 3, int slowing = 3, int shift = 0)
  {
   int stoch_handle = 0, copied = 0;
   double stoch_b[];

   stoch_handle = iStochastic(_Symbol, _Period, k_period, d_period, slowing, MODE_SMA, STO_LOWHIGH);
   if(stoch_handle == INVALID_HANDLE)
     {
      Print("[ERROR] Cannot create Stochastic indicator");
      return 0;
     }

   copied = CopyBuffer(stoch_handle, 0, 0, 1, stoch_b);
   if(copied != 1)
     {
      Print("[ERROR] CopyBuffer Stochastic K failed");
      return 0;
     }

   return stoch_b[0];
  }

//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
float GetStochD(int k_period, int d_period = 3, int slowing = 3, int shift = 0)
  {
   int stoch_handle = 0, copied = 0;
   double stoch_d[];

   stoch_handle = iStochastic(_Symbol, _Period, k_period, d_period, slowing, MODE_SMA, STO_LOWHIGH);
   if(stoch_handle == INVALID_HANDLE)
     {
      Print("[ERROR] Cannot create Stochastic indicator");
      return 0;
     }

   copied = CopyBuffer(stoch_handle, 1, 0, 1, stoch_d);
   if(copied != 1)
     {
      Print("[ERROR] CopyBuffer Stochastic D failed");
      return 0;
     }

   return stoch_d[0];
  }

//+------------------------------------------------------------------+
//|                                                                  |
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
