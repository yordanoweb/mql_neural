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
    if(IsNewCandle())
      {
       
      }
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
