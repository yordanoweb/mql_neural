//+------------------------------------------------------------------+
//|                                              SimpleONNX_EA.mq5   |
//+------------------------------------------------------------------+
#property strict

#include <Trade\Trade.mqh>

//--- ENUMERATIONS
enum ENUM_LOGIC { LOGIC_NORMAL, LOGIC_MIRROR };

//--- INPUTS
input group "AI Config"
input ENUM_LOGIC InpLogic      = LOGIC_MIRROR;
input string     InpModelFile  = "ndx100_rates_m5_3_feat.onnx";
input float      InpMinConf    = 0.55;
input int        InpStartHour  = 0;
input int        InpEndHour    = 23;
input group "Risk"
input int        InpRSI        = 14;
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
  }

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
