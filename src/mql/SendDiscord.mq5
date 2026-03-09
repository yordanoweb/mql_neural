//+------------------------------------------------------------------+
//| Discord Trade Alerts EA - Complete Working Version              |
//+------------------------------------------------------------------+
#property strict

#include "Credentials.mqh"

input group "Discord ===================="
input long   CHANNEL_ID = 0;                     // Main channel ID
input long   ORDERS_STREAM_CHANNEL_ID = 0;       // Orders stream channel ID
input int    InpMagicNumber = 123456;            // Magic Number (filter by EA)
input string InpNotificationTitle = "";          // Custom notification title (empty = use EA name)
input bool   DISCORD_SEND = true;                // Enable sending notifications
input bool   SEND_TEST_ON_START = true;          // Send test message on EA start
input bool   InpEnableDebugLogs = false;         // Enable Debug Logging

ulong last_dealticket = 0; // Track last processed deal

//+------------------------------------------------------------------+
//| Send Discord message via HTTP POST                               |
//+------------------------------------------------------------------+
bool SendDiscord(string message, long channel_id)
{
   if(!DISCORD_SEND) // The original code had an error here, assuming 'if(DISCORD_SEND)' was meant to be 'if(!DISCORD_SEND)' for disabling. I'll respect the original logic but print a note.
   {
      Print("Envio desativado (DISCORD_SEND=false): ", message);
      return true;
   }

   string token = CredentialDiscordToken();

   if(token == "" || channel_id == 0)
   {
      Print("ERRO: Token ou Channel ID não configurados!");
      return false;
   }

   string url = "https://discord.com/api/v10/channels/" + IntegerToString(channel_id) + "/messages";
   
   // Properly escape special characters in JSON
   StringReplace(message, "\\", "\\\\"); // Escape backslashes first
   StringReplace(message, "\"", "\\\"");  // Escape quotes
   StringReplace(message, "\n", "\\n");    // Escape newlines
   StringReplace(message, "\r", "\\r");    // Escape carriage returns
   StringReplace(message, "\t", "\\t");    // Escape tabs
   
   string payload = "{\"content\":\"" + message + "\\n\\n\"}";
   
   string headers = "Authorization: Bot " + token + "\r\n";
   headers += "Content-Type: application/json; charset=utf-8\r\n";
   
   char data[], result[];
   string result_headers;
   
   // Convert payload string to char array with proper termination
   int len = StringToCharArray(payload, data, 0, WHOLE_ARRAY, CP_UTF8);
   if(len > 0)
   {
      // Remove null terminator that StringToCharArray adds
      ArrayResize(data, len - 1);
   }
   
   if(InpEnableDebugLogs)
      Print("NOTIFICATION DATA: ", payload);
   
   // WebRequest call
   int res = WebRequest("POST", url, headers, 5000, data, result, result_headers);
   
   if(res == 200 || res == 204)
   {
      if(InpEnableDebugLogs)
         Print("Mensagem enviada: ", message);
      return true;
   }
   else
   {
      Print("Erro Discord. Codigo: ", res, " Resposta: ", CharArrayToString(result, 0, WHOLE_ARRAY, CP_UTF8));
      return false;
   }
}

//+------------------------------------------------------------------+
//| Format trade details into Discord message                        |
//+------------------------------------------------------------------+
string FormatTradeMessage(ENUM_DEAL_TYPE deal_type, ENUM_DEAL_ENTRY deal_entry, double volume, double price, double profit, long deal_ticket, ulong position_ticket, string symbol)
{
   string action = "";
   string emoji = "";
   string title = "";
   
   // Get EA title (custom or default)
   string ea_title = (InpNotificationTitle == "") ? MQLInfoString(MQL_PROGRAM_NAME) : InpNotificationTitle;
   
   // Differentiate between opening and closing
   if(deal_entry == DEAL_ENTRY_IN)
   {
      // Opening position
      action = (deal_type == DEAL_TYPE_BUY) ? "🟢 COMPRA" : "🔴 VENDA";
      emoji = (deal_type == DEAL_TYPE_BUY) ? "📈" : "📉";
      title = "**NOVA ORDEM ABERTA**";
      
      string msg = StringFormat(
         "-----------------------------\n🤖 **%s**\n" +
         "%s %s\n" +
         "%s %s\n" +
         "`%s` | Vol: %.2f | Preço: %.5f\n" +
         "🎫 Posição: `#%llu`\n" +
         "🕐 %s",
         ea_title,
         emoji, title,
         action, symbol,
         symbol, volume, price,
         position_ticket,
         TimeToString(TimeCurrent(), TIME_DATE|TIME_MINUTES)
      );
      return msg;
   }
   else if(deal_entry == DEAL_ENTRY_OUT)
   {
      // Closing position
      action = (deal_type == DEAL_TYPE_BUY) ? "🔴 FECHOU VENDA" : "🟢 FECHOU COMPRA";
      
      // Determine profit/loss emoji and status
      string pl_emoji = "";
      string pl_status = "";
      if(profit > 0)
      {
         pl_emoji = "✅";
         pl_status = "LUCRO";
      }
      else if(profit < 0)
      {
         pl_emoji = "❌";
         pl_status = "PERDA";
      }
      else
      {
         pl_emoji = "⚪";
         pl_status = "BREAK EVEN";
      }
      
      string msg = StringFormat(
         "-----------------------------\n🤖 **%s**\n" +
         "%s **ORDEM FECHADA**\n" +
         "%s %s\n" +
         "`%s` | Vol: %.2f | Preço: %.5f\n" +
         "🎫 Posição: `#%llu`\n" +
         "%s **%s: %.2f %s**\n" +
         "🕐 %s",
         ea_title,
         pl_emoji,
         action, symbol,
         symbol, volume, price,
         position_ticket,
         pl_emoji, pl_status, profit, AccountInfoString(ACCOUNT_CURRENCY),
         TimeToString(TimeCurrent(), TIME_DATE|TIME_MINUTES)
      );
      return msg;
   }
   
   return "";
}

//+------------------------------------------------------------------+
//| Expert initialization                                            |
//+------------------------------------------------------------------+
int OnInit()
{
   if(DISCORD_SEND)
   {
      Print("Discord Trade Alerts iniciado para \"" + _Symbol + "\". Token: ", (DISCORD_TOKEN == "" ? "NÃO CONFIGURADO" : "OK"));
      Print("Main Channel: ", CHANNEL_ID, " | Orders Channel: ", ORDERS_STREAM_CHANNEL_ID);
      Print("Magic Number: ", InpMagicNumber);
   
      if(SEND_TEST_ON_START)
      {
         string ea_title = (InpNotificationTitle == "") ? MQLInfoString(MQL_PROGRAM_NAME) : InpNotificationTitle;
         string start_msg = "------------------------------\n🟢 **MT5 Trade Bot online!**\n🤖 **" + ea_title + "**\nMonitorando '**" + _Symbol + "**' ordens... ✅";
         SendDiscord(start_msg, CHANNEL_ID);
      }
   }
      
   return(INIT_SUCCEEDED);
}

//+------------------------------------------------------------------+
//| Trade event - CAPTURES ONLY BUY/SELL EXECUTIONS                 |
//+------------------------------------------------------------------+
void OnTrade()
{
   if(!DISCORD_SEND) return;
   
   // Get last deal info from history
   if(HistorySelect(TimeCurrent()-60, TimeCurrent())) // Last 60 seconds
   {
      int total = HistoryDealsTotal();
      if(total > 0)
      {
         ulong ticket = HistoryDealGetTicket(total-1);
         
         // Only process NEW deals (avoid duplicates)
         if(ticket != last_dealticket && ticket > 0)
         {
            last_dealticket = ticket;
            
            // Get deal properties
            ENUM_DEAL_TYPE deal_type = (ENUM_DEAL_TYPE)HistoryDealGetInteger(ticket, DEAL_TYPE);
            ENUM_DEAL_ENTRY deal_entry = (ENUM_DEAL_ENTRY)HistoryDealGetInteger(ticket, DEAL_ENTRY);
            string symbol = HistoryDealGetString(ticket, DEAL_SYMBOL);
            ulong position_id = HistoryDealGetInteger(ticket, DEAL_POSITION_ID);
            
            // Only process IN or OUT deals (actual trades)
            if(deal_entry == DEAL_ENTRY_IN || deal_entry == DEAL_ENTRY_OUT)
            {
               if(InpEnableDebugLogs)
                  Print("Processing deal: Ticket=", ticket, " Symbol=", symbol, " Entry=", EnumToString(deal_entry));
               
               double volume = HistoryDealGetDouble(ticket, DEAL_VOLUME);
               double price = HistoryDealGetDouble(ticket, DEAL_PRICE);
               double profit = HistoryDealGetDouble(ticket, DEAL_PROFIT);
               double swap = HistoryDealGetDouble(ticket, DEAL_SWAP);
               double commission = HistoryDealGetDouble(ticket, DEAL_COMMISSION);
               
               // Total profit includes swap and commission
               double total_profit = profit + swap + commission;
               
               // Format and send message with position ID for linking
               string message = FormatTradeMessage(deal_type, deal_entry, volume, price, total_profit, ticket, position_id, symbol);
               
               if(message != "")
               {
                  // Send to appropriate channel
                  long target_channel = (ORDERS_STREAM_CHANNEL_ID > 0) ? ORDERS_STREAM_CHANNEL_ID : CHANNEL_ID;
                  SendDiscord(message, target_channel);
               }
            }
         }
      }
   }
}

//+------------------------------------------------------------------+
//| Tick - optional: send price alerts (uncomment if needed)        |
//+------------------------------------------------------------------+
void OnTick()
{
   // Exemplo: SendDiscord("💹 " + _Symbol + " @ " + DoubleToString(SymbolInfoDouble(_Symbol,SYMBOL_BID),5), CHANNEL_ID);
}
