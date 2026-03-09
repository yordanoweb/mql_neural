//+------------------------------------------------------------------+
//|                                               SupabaseMonitor.mq5 |
//|                                      Polls Supabase for new trades |
//+------------------------------------------------------------------+
#property copyright "Your Name"
#property link      "https://yourwebsite.com"
#property version   "1.00"

#include <Trade/Trade.mqh>

#include "Credentials.mqh"
#include "Supabase.mqh"   // ensure this file is in Include folder

CSupabaseTrades *supa = NULL;

//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit()
{
   supa = new CSupabaseTrades(CredentialSupabaseURL(), CredentialSupabaseAnonKey(), CredentialSupabaseServiceKey());
   supa.UseServiceRole(true);
   robotName = RobotName;

   if(EnableLogging)
      Print("SupabaseTradeLogger initialized. Robot: ", robotName);
   
   return INIT_SUCCEEDED;
}
