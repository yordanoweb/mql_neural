//+------------------------------------------------------------------+
//|                                                 SupabaseTrades.mqh |
//|                                    Generated from REST guidelines |
//|                                              https://supabase.com |
//+------------------------------------------------------------------+
#property copyright "Your Name"
#property link      "https://yourwebsite.com"
#property version   "1.00"

//+------------------------------------------------------------------+
//| Includes                                                         |
//+------------------------------------------------------------------+
#include <Arrays/ArrayString.mqh>

input group "Supabase ==================== ";
input string SupabaseURL   = "***REMOVED***";
input string AnonKey       = "***REMOVED***";
input string ServiceKey    = "***REMOVED***";
input string RobotName = "TestSupa";
input bool EnableLogging = true;

ulong            lastDealTicket = 0;   // last processed deal ticket (avoid duplicates)
string           robotName;            // EA identifier
string           tradeDirection = "";

//+------------------------------------------------------------------+
//| CSupabaseTrades class                                            |
//| Provides HTTP GET/POST methods to interact with Supabase trades  |
//| table via PostgREST API.                                         |
//+------------------------------------------------------------------+
class CSupabaseTrades
{
private:
   string            m_base_url;      // Full URL to trades endpoint (e.g., https://xyz.supabase.co/rest/v1/trades)
   string            m_anon_key;      // anon/public key (RLS applied)
   string            m_service_key;   // service_role key (bypasses RLS)
   bool              m_use_service_role; // flag to use service key by default

   //--- Internal request sender (uses uchar arrays for WebRequest)
   int               SendRequest(string method, string url, string headers, string data, string &response, string &response_headers);

public:
   //--- Constructor / Destructor
                     CSupabaseTrades(string base_url, string anon_key, string service_key = "");
                    ~CSupabaseTrades();

   //--- Key management
   bool              SetKeys(string anon_key, string service_key = "");
   void              UseServiceRole(bool use) { m_use_service_role = use; }

   //--- Core REST methods
   bool              Get(string query, string &response, int &status_code, string &response_headers);
   bool              Post(string body, string &response, int &status_code, string &response_headers, bool use_service_role = false);
   bool              GetUseServiceRole();

   //--- Helper: Insert a single trade (constructs JSON body)
   bool              InsertTrade(string ticket, long login, string symbol, int cmd, double volume,
                                 double price_open, datetime time_setup, string &response,
                                 int &status_code, string &response_headers, bool use_service_role = false);

   //--- Utility functions
   static string     TimeToISO(datetime time);
   static string     URLEncode(string str);
};

//+------------------------------------------------------------------+
//| Constructor                                                      |
//+------------------------------------------------------------------+
CSupabaseTrades::CSupabaseTrades(string base_url, string anon_key, string service_key = "")
{
   m_base_url = base_url;
   m_anon_key = anon_key;
   m_service_key = service_key;
   m_use_service_role = false;
}

//+------------------------------------------------------------------+
//| Destructor                                                       |
//+------------------------------------------------------------------+
CSupabaseTrades::~CSupabaseTrades()
{
}

//+------------------------------------------------------------------+
//| Set or update API keys                                           |
//+------------------------------------------------------------------+
bool CSupabaseTrades::SetKeys(string anon_key, string service_key = "")
{
   m_anon_key = anon_key;
   m_service_key = service_key;
   return true;
}

bool CSupabaseTrades::GetUseServiceRole(void)
{
   return m_use_service_role;
}

//+------------------------------------------------------------------+
//| Low-level HTTP request (fixed: uses uchar arrays)               |
//+------------------------------------------------------------------+
int CSupabaseTrades::SendRequest(string method, string url, string headers, string data, string &response, string &response_headers)
{
   response = "";
   response_headers = "";
   int timeout = 5000; // milliseconds

   // Convert data string to uchar array (for POST)
   uchar post_data[];
   int data_len = StringLen(data);
   if(data_len > 0)
   {
      ArrayResize(post_data, data_len);
      for(int i = 0; i < data_len; i++)
         post_data[i] = (uchar)StringGetCharacter(data, i);
   }

   // Buffer for response data
   uchar result_data[];
   string result_headers_str;

   // Call WebRequest with uchar arrays (7‑parameter overload)
   int res = WebRequest(method, url, headers, timeout, post_data, result_data, result_headers_str);
   response_headers = result_headers_str;

   if(res == -1)
   {
      int err = GetLastError();
      Print("WebRequest error: ", err);
      return -1;
   }

   // Convert result uchar array back to string
   int result_len = ArraySize(result_data);
   if(result_len > 0)
   {
      for(int i = 0; i < result_len; i++)
         response += CharToString(result_data[i]);
   }

   // Extract HTTP status code from response_headers (first line)
   int status = 0;
   if(StringFind(result_headers_str, "HTTP/") == 0)
   {
      string first_line = StringSubstr(result_headers_str, 0, StringFind(result_headers_str, "\r\n"));
      string parts[];
      if(StringSplit(first_line, ' ', parts) >= 2)
         status = (int)StringToInteger(parts[1]);
   }
   return status;
}

//+------------------------------------------------------------------+
//| GET request                                                      |
//+------------------------------------------------------------------+
bool CSupabaseTrades::Get(string query, string &response, int &status_code, string &response_headers)
{
   string url = m_base_url + query;
   string headers = "";
   string key = m_use_service_role ? m_service_key : m_anon_key;

   headers += "apikey: " + key + "\r\n";
   headers += "Authorization: Bearer " + key + "\r\n";

   status_code = SendRequest("GET", url, headers, "", response, response_headers);
   return (status_code >= 200 && status_code < 300);
}

//+------------------------------------------------------------------+
//| POST request                                                     |
//+------------------------------------------------------------------+
bool CSupabaseTrades::Post(string body, string &response, int &status_code, string &response_headers, bool use_service_role = false)
{
   string url = m_base_url;
   string headers = "";
   string key = use_service_role ? m_service_key : m_anon_key;

   headers += "apikey: " + key + "\r\n";
   headers += "Authorization: Bearer " + key + "\r\n";
   headers += "Content-Type: application/json\r\n";

   status_code = SendRequest("POST", url, headers, body, response, response_headers);
   return (status_code >= 200 && status_code < 300);
}

//+------------------------------------------------------------------+
//| Helper: Insert a single trade                                    |
//+------------------------------------------------------------------+
bool CSupabaseTrades::InsertTrade(string ticket, long login, string symbol, int cmd, double volume,
                                   double price_open, datetime time_setup, string &response,
                                   int &status_code, string &response_headers, bool use_service_role = false)
{
   // Build JSON object
   string body = "{";
   body += "\"ticket\":\"" + ticket + "\",";
   body += "\"login\":" + IntegerToString(login) + ",";
   body += "\"symbol\":\"" + symbol + "\",";
   body += "\"cmd\":" + IntegerToString(cmd) + ",";
   body += "\"volume\":\"" + DoubleToString(volume, 2) + "\",";
   body += "\"price_open\":\"" + DoubleToString(price_open, 5) + "\",";
   body += "\"time_setup\":\"" + TimeToISO(time_setup) + "\"";
   body += "}";

   return Post(body, response, status_code, response_headers, use_service_role);
}

//+------------------------------------------------------------------+
//| Convert MQL5 datetime to ISO 8601 string (UTC)                  |
//+------------------------------------------------------------------+
string CSupabaseTrades::TimeToISO(datetime time)
{
   MqlDateTime dt;
   TimeToStruct(time, dt);
   return StringFormat("%04d-%02d-%02dT%02d:%02d:%02dZ",
                       dt.year, dt.mon, dt.day, dt.hour, dt.min, dt.sec);
}

//+------------------------------------------------------------------+
//| Simple URL encoding                                              |
//+------------------------------------------------------------------+
string CSupabaseTrades::URLEncode(string str)
{
   string res = "";
   for(int i = 0; i < StringLen(str); i++)
   {
      ushort c = StringGetCharacter(str, i);
      if((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') ||
         c == '-' || c == '_' || c == '.' || c == '~')
         res += ShortToString(c);
      else
         res += "%" + StringFormat("%02X", c);
   }
   return res;
}

//+------------------------------------------------------------------+
//| Trade event handler                                              |
//+------------------------------------------------------------------+
void OnTrade()
{
   ProcessTradeEvent();
}

//+------------------------------------------------------------------+
//| Process the most recent trade deal                               |
//+------------------------------------------------------------------+
void ProcessTradeEvent()
{
   // Look at deals in the last 60 seconds
   if(!HistorySelect(TimeCurrent() - 60, TimeCurrent()))
      return;
   
   int total = HistoryDealsTotal();
   if(total == 0)
      return;
   
   // Get the most recent deal
   ulong ticket = HistoryDealGetTicket(total - 1);
   if(ticket == lastDealTicket)
      return;   // already processed
   
   ENUM_DEAL_ENTRY entry = (ENUM_DEAL_ENTRY)HistoryDealGetInteger(ticket, DEAL_ENTRY);
   // Only log actual trade executions (open or close)
   if(entry == DEAL_ENTRY_IN || entry == DEAL_ENTRY_OUT)
   {
      lastDealTicket = ticket;
      BuildAndSendTradeData(ticket, entry);
   }
}

//+------------------------------------------------------------------+
//| Build JSON payload and send to Supabase                          |
//+------------------------------------------------------------------+
bool BuildAndSendTradeData(ulong dealTicket, ENUM_DEAL_ENTRY entry)
{
   // --- Common data from the current deal ---
   string symbol       = HistoryDealGetString(dealTicket, DEAL_SYMBOL);
   long   login        = AccountInfoInteger(ACCOUNT_LOGIN);
   double volume       = HistoryDealGetDouble(dealTicket, DEAL_VOLUME);
   double commission   = HistoryDealGetDouble(dealTicket, DEAL_COMMISSION);
   double swap         = HistoryDealGetDouble(dealTicket, DEAL_SWAP);
   double profit       = HistoryDealGetDouble(dealTicket, DEAL_PROFIT);
   long   position_id  = HistoryDealGetInteger(dealTicket, DEAL_POSITION_ID);
   int    reason       = (int)HistoryDealGetInteger(dealTicket, DEAL_REASON);
   datetime dealTime   = (datetime)HistoryDealGetInteger(dealTicket, DEAL_TIME);
   int    digits       = (int)SymbolInfoInteger(symbol, SYMBOL_DIGITS);
   
   // --- Variables to be filled according to entry type ---
   int      cmd          = -1;
   double   price_open   = 0.0;
   double   price_current = 0.0;
   double   price_sl     = 0.0;
   double   price_tp     = 0.0;
   int      magic        = 0;
   string   comment      = "";
   datetime time_setup   = 0;
   datetime time_done    = 0;
   int      state        = -1;
   
   if(entry == DEAL_ENTRY_IN)   // --- OPEN DEAL ---
   {
      cmd          = (int)HistoryDealGetInteger(dealTicket, DEAL_TYPE);   // 0=BUY,1=SELL
      price_open   = HistoryDealGetDouble(dealTicket, DEAL_PRICE);
      price_current = price_open;            // initially same as open
      time_setup   = dealTime;
      time_done    = 0;
      state        = 0;                      // 0 = open
      tradeDirection = "ENTRY";
      
      // Get order details for magic, comment, SL/TP
      ulong orderTicket = HistoryDealGetInteger(dealTicket, DEAL_ORDER);
      if(orderTicket > 0 && HistoryOrderSelect(orderTicket))
      {
         magic    = (int)HistoryOrderGetInteger(orderTicket, ORDER_MAGIC);
         string _comment = HistoryOrderGetString(orderTicket, ORDER_COMMENT);
         comment  = StringFormat("%s %s", tradeDirection, StringLen(_comment) > 0 ? "- " + _comment : "");
         price_sl = HistoryOrderGetDouble(orderTicket, ORDER_SL);
         price_tp = HistoryOrderGetDouble(orderTicket, ORDER_TP);
      }
   }
   else if(entry == DEAL_ENTRY_OUT)   // --- CLOSE DEAL ---
   {
      tradeDirection = "EXIT";
      // Find the corresponding open deal for this position
      ulong openTicket = FindOpenDealForPosition(position_id);
      if(openTicket == 0)
      {
         if(EnableLogging)
            Print("Error: Cannot find open deal for position ", position_id);
         return false;
      }
      
      // Data from the open deal
      cmd          = (int)HistoryDealGetInteger(openTicket, DEAL_TYPE);
      price_open   = HistoryDealGetDouble(openTicket, DEAL_PRICE);
      time_setup   = (datetime)HistoryDealGetInteger(openTicket, DEAL_TIME);
      
      // Get order details from the open deal (magic, comment, original SL/TP)
      ulong orderTicket = HistoryDealGetInteger(openTicket, DEAL_ORDER);
      if(orderTicket > 0 && HistoryOrderSelect(orderTicket))
      {
         magic   = (int)HistoryOrderGetInteger(orderTicket, ORDER_MAGIC);
         string _comment = HistoryOrderGetString(orderTicket, ORDER_COMMENT);
         comment  = StringFormat("%s %s", tradeDirection, StringLen(_comment) > 0 ? "- " + _comment : "");
         // We do NOT set SL/TP from the order because they may have been modified.
         // Instead, we leave them as 0 (null in JSON) or you could fetch current position SL/TP.
      }
      
      // Data from the closing deal
      price_current = HistoryDealGetDouble(dealTicket, DEAL_PRICE);   // close price
      time_done     = dealTime;
      state         = 1;                      // 1 = closed
   }
   
   // --- Build JSON payload (matching your table columns) ---
   string json = "{";
   json += "\"ticket\":\"" + (string)dealTicket + "\",";                // as string to preserve bigint
   json += "\"login\":" + (string)login + ",";
   json += "\"symbol\":\"" + StringEscapeJSON(symbol) + "\",";
   json += "\"cmd\":" + (string)cmd + ",";
   json += "\"volume\":" + DoubleToString(volume, 6) + ",";
   json += "\"price_open\":" + DoubleToString(price_open, digits) + ",";
   
   if(price_current != 0.0)
      json += "\"price_current\":" + DoubleToString(price_current, digits) + ",";
   if(price_sl != 0.0)
      json += "\"price_stoploss\":" + DoubleToString(price_sl, digits) + ",";
   if(price_tp != 0.0)
      json += "\"price_takeprofit\":" + DoubleToString(price_tp, digits) + ",";
   
   json += "\"commission\":" + DoubleToString(commission, 2) + ",";
   json += "\"swap\":" + DoubleToString(swap, 2) + ",";
   json += "\"profit\":" + DoubleToString(profit, 2) + ",";
   json += "\"magic\":" + (string)magic + ",";
   
   if(comment != "")
      json += "\"comment\":\"" + StringEscapeJSON(comment) + "\",";
   
   // expiration not used – omit
   
   if(time_setup != 0)
      json += "\"time_setup\":\"" + CSupabaseTrades::TimeToISO(time_setup) + "\",";
   if(time_done != 0)
      json += "\"time_done\":\"" + CSupabaseTrades::TimeToISO(time_done) + "\",";
   
   json += "\"state\":" + (string)state + ",";
   json += "\"digits\":" + (string)digits + ",";
   json += "\"reason\":\"" + (string)reason + "\",";
   json += "\"position_id\":" + (string)position_id + ",";
   json += "\"robot\":\"" + StringEscapeJSON(robotName) + "\"";
   
   json += "}";
   
   // --- Send POST request ---
   string response;
   int    status_code;
   string response_headers;
   bool   success = supa.Post(json, response, status_code, response_headers, supa.GetUseServiceRole());
   
   if(EnableLogging)
   {
      if(success)
         Print("Trade sent successfully. Ticket: ", dealTicket, " Status: ", status_code);
      else
        {
         Print("Failed to send trade. Ticket: ", dealTicket, " Status: ", status_code, " Response: ", response);
         Print("LAST ERROR: ", GetLastError());
        }
   }
   
   return success;
}

//+------------------------------------------------------------------+
//| Find the open deal ticket for a given position ID                |
//+------------------------------------------------------------------+
ulong FindOpenDealForPosition(ulong position_id)
{
   // Search all history (from the beginning until now)
   HistorySelect(0, TimeCurrent());
   int total = HistoryDealsTotal();
   for(int i = 0; i < total; i++)
   {
      ulong ticket = HistoryDealGetTicket(i);
      if(HistoryDealGetInteger(ticket, DEAL_POSITION_ID) == position_id)
      {
         ENUM_DEAL_ENTRY entry = (ENUM_DEAL_ENTRY)HistoryDealGetInteger(ticket, DEAL_ENTRY);
         if(entry == DEAL_ENTRY_IN)
            return ticket;
      }
   }
   return 0;
}

//+------------------------------------------------------------------+
//| Escape special characters for JSON string                        |
//+------------------------------------------------------------------+
string StringEscapeJSON(string s)
{
   string res = "";
   for(int i = 0; i < StringLen(s); i++)
   {
      ushort c = StringGetCharacter(s, i);
      switch(c)
      {
         case '"':  res += "\\\""; break;
         case '\\': res += "\\\\"; break;
         case '/':  res += "\\/";  break;
         case '\b': res += "\\b";  break;
         case '\f': res += "\\f";  break;
         case '\n': res += "\\n";  break;
         case '\r': res += "\\r";  break;
         case '\t': res += "\\t";  break;
         default:   res += ShortToString(c);
      }
   }
   return res;
}

//+------------------------------------------------------------------+
//+------------------------------------------------------------------+
//| EXAMPLE USAGE (to be placed in your EA)                         |
//|                                                                  |
//| 1. Define these input parameters in your EA:                    |
//|                                                                  |
//|    input string SupabaseURL   = "https://xyz.supabase.co/rest/v1/trades"; |
//|    input string AnonKey       = "eyJhbGci...";                  |
//|    input string ServiceKey    = "";   // optional                |
//|                                                                  |
//| 2. Include this file and use the class:                         |
//|                                                                  |
//|    #include <SupabaseTrades.mqh>                                |
//|    CSupabaseTrades *supa;                                       |
//|                                                                  |
//|    int OnInit()                                                 |
//|    {                                                            |
//|       supa = new CSupabaseTrades(SupabaseURL, AnonKey, ServiceKey); |
//|       // supa.UseServiceRole(true); // if you want to use service key |
//|       return INIT_SUCCEEDED;                                    |
//|    }                                                            |
//|                                                                  |
//|    void OnDeinit(const int reason)                              |
//|    {                                                            |
//|       delete supa;                                              |
//|    }                                                            |
//|                                                                  |
//|    // Now use supa.Get(), supa.Post(), etc.                     |
//+------------------------------------------------------------------+
