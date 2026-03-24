//+------------------------------------------------------------------+
//| Multi-Strategy EA (MT5)                                          |
//| 5 strategies + ATR exits + OnTimer status                        |
//+------------------------------------------------------------------+
#property strict
#include <Trade/Trade.mqh>
CTrade trade;

// --- Inputs ---
input double RiskPerTrade = 1.0;       // % risk per trade
input double ATRMultiplierSL = 1.5;    // Stop-loss = 1.5 * ATR
input double ATRMultiplierTP = 3.0;    // Take-profit = 3.0 * ATR
input int ATRPeriod = 14;              // ATR calculation period
input int TimerSeconds = 2;            // Update chart every X seconds
input int MagicNumber = 9999111; // Unique ID for this EA

datetime lastCandleTime = 0;

// --- Function prototypes ---
bool CheckStrategy1();
bool CheckStrategy2();
bool CheckStrategy3();
bool CheckStrategy4();
bool CheckStrategy5();

//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void ExecuteTrade(string side);
void ManageExits();
bool IsPositionOpen();
bool IsNewCandle();

//+------------------------------------------------------------------+
//| Initialization                                                   |
//+------------------------------------------------------------------+
int OnInit()
  {
   EventSetTimer(TimerSeconds);
   Print("Multi-Strategy EA initialized.");
   trade.SetExpertMagicNumber(MagicNumber);
   return(INIT_SUCCEEDED);
  }

//+------------------------------------------------------------------+
void OnDeinit()
  {
   EventKillTimer();
   Print("EA stopped.");
  }

//+------------------------------------------------------------------+
//| Tick function (trading logic only)                               |
//+------------------------------------------------------------------+
void OnTick()
  {
   // Only proceed on a new candle
   if(!IsNewCandle())
      return;
   if(!IsPositionOpen())
     {
      if(CheckStrategy1())
         ExecuteTrade("BUY");
      else if(CheckStrategy2())
         ExecuteTrade("BUY");
      else if(CheckStrategy3())
         ExecuteTrade("BUY");
      else if(CheckStrategy4())
         ExecuteTrade("SELL");
      else if(CheckStrategy5())
         ExecuteTrade("BUY");
     }
   ManageExits();
  }

//+------------------------------------------------------------------+
//| Timer event (status reporting only)                              |
//+------------------------------------------------------------------+
void OnTimer()
  {
   string status = "\n\n=== Strategy Dashboard ===\n";

   if(CheckStrategy1())
      status += "Strategy 1: BUY signal detected\n";
   if(CheckStrategy2())
      status += "Strategy 2: BUY signal detected\n";
   if(CheckStrategy3())
      status += "Strategy 3: BUY signal detected\n";
   if(CheckStrategy4())
      status += "Strategy 4: SELL signal detected\n";
   if(CheckStrategy5())
      status += "Strategy 5: BUY signal detected\n";

   double atr = iATR(_Symbol, PERIOD_M15, ATRPeriod);
   status += "ATR(" + IntegerToString(ATRPeriod) + "): " + DoubleToString(atr, 5) + "\n";

// Show open positions info
   int total = PositionsTotal();
   status += "Open positions: " + IntegerToString(total) + "\n";

   Comment(status);
  }

bool IsNewCandle()
{
   datetime currentCandle = iTime(_Symbol, PERIOD_CURRENT, 0);

   if(currentCandle != lastCandleTime)
   {
      lastCandleTime = currentCandle;
      return true;
   }

   return false;
}

// Function to check if a position is already open for this symbol and magic number
bool IsPositionOpen()
  {
   for(int i = PositionsTotal() - 1; i >= 0; i--)
     {
      ulong ticket = PositionGetTicket(i);
      if(PositionSelectByTicket(ticket))
        {
         if(PositionGetString(POSITION_SYMBOL) == _Symbol &&
            PositionGetInteger(POSITION_MAGIC) == MagicNumber)
            return true; // A position is already open
        }
     }
   return false;
  }

//+------------------------------------------------------------------+
//| Strategy implementations                                         |
//+------------------------------------------------------------------+
bool CheckStrategy1()
  {
   double rsi = iRSI(_Symbol, PERIOD_M15, 14, PRICE_CLOSE);
   double stochK = iStochastic(_Symbol, PERIOD_M15, 5, 3, 3, MODE_SMA, STO_LOWHIGH);
   double ema50 = iMA(_Symbol, PERIOD_M15, 50, 0, MODE_EMA, PRICE_CLOSE);
   double ema200 = iMA(_Symbol, PERIOD_M15, 200, 0, MODE_EMA, PRICE_CLOSE);
   return (rsi < 20 && stochK < 10 && ema50 < ema200);
  }

//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CheckStrategy2()
  {
   double rsi = iRSI(_Symbol, PERIOD_M15, 14, PRICE_CLOSE);
   double ema50 = iMA(_Symbol, PERIOD_M15, 50, 0, MODE_EMA, PRICE_CLOSE);
   double ema200 = iMA(_Symbol, PERIOD_M15, 200, 0, MODE_EMA, PRICE_CLOSE);
   double atr = iATR(_Symbol, PERIOD_M15, ATRPeriod);
   return (rsi < 20 && ema50 < ema200 && atr > SymbolInfoDouble(_Symbol, SYMBOL_POINT) * 100);
  }

//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CheckStrategy3()
  {
   double rsi = iRSI(_Symbol, PERIOD_M15, 14, PRICE_CLOSE);
   double ema50 = iMA(_Symbol, PERIOD_M15, 50, 0, MODE_EMA, PRICE_CLOSE);
   double ema200 = iMA(_Symbol, PERIOD_M15, 200, 0, MODE_EMA, PRICE_CLOSE);
   double atr = iATR(_Symbol, PERIOD_M15, ATRPeriod);
   return (rsi < 25 && ema50 < ema200 && atr > SymbolInfoDouble(_Symbol, SYMBOL_POINT) * 100);
  }

//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CheckStrategy4()
  {
   double stochK = iStochastic(_Symbol, PERIOD_M15, 5, 3, 3, MODE_SMA, STO_LOWHIGH);
   double ema10 = iMA(_Symbol, PERIOD_M15, 10, 0, MODE_EMA, PRICE_CLOSE);
   double ema20 = iMA(_Symbol, PERIOD_M15, 20, 0, MODE_EMA, PRICE_CLOSE);
   double macdMain = iMACD(_Symbol, PERIOD_M15, 12, 26, 9, PRICE_CLOSE);
   return (stochK > 80 && ema10 < ema20 && macdMain < 0);
  }

//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CheckStrategy5()
  {
   double rsi = iRSI(_Symbol, PERIOD_M15, 14, PRICE_CLOSE);
   double stochK = iStochastic(_Symbol, PERIOD_M15, 5, 3, 3, MODE_SMA, STO_LOWHIGH);
   double ema50 = iMA(_Symbol, PERIOD_M15, 50, 0, MODE_EMA, PRICE_CLOSE);
   double ema200 = iMA(_Symbol, PERIOD_M15, 200, 0, MODE_EMA, PRICE_CLOSE);
   return (rsi < 25 && stochK < 10 && ema50 < ema200);
  }

//+------------------------------------------------------------------+
//| Trade execution                                                  |
//+------------------------------------------------------------------+
void ExecuteTrade(string side)
  {
   double atr = iATR(_Symbol, PERIOD_M15, ATRPeriod);
   double slDistance = atr * ATRMultiplierSL;
   double tpDistance = atr * ATRMultiplierTP;

   double lotSize = 0.1; // TODO: dynamic lot sizing
   double price = (side == "BUY") ? SymbolInfoDouble(_Symbol, SYMBOL_ASK)
                  : SymbolInfoDouble(_Symbol, SYMBOL_BID);

   double sl = (side == "BUY") ? price - slDistance : price + slDistance;
   double tp = (side == "BUY") ? price + tpDistance : price - tpDistance;

   Print("Executing ", side, " trade | SL=", sl, " TP=", tp);

   if(side == "BUY")
      trade.Buy(lotSize, NULL, price, sl, tp,
                MQLInfoString(MQL_PROGRAM_NAME) + " BUY@" + DoubleToString(price, _Digits));
   else
      trade.Sell(lotSize, NULL, price, sl, tp,
                 MQLInfoString(MQL_PROGRAM_NAME) + " SELL@" + DoubleToString(price, _Digits));
  }

//+------------------------------------------------------------------+
//| Exit management (Trailing + Breakeven)                          |
//+------------------------------------------------------------------+
void ManageExits()
  {
   for(int i = PositionsTotal() - 1; i >= 0; i--)
     {
      ulong ticket = PositionGetTicket(i);
      if(ticket > 0 && PositionSelectByTicket(ticket))
        {
         string symbol = PositionGetString(POSITION_SYMBOL);
         double entryPrice = PositionGetDouble(POSITION_PRICE_OPEN);
         double sl = PositionGetDouble(POSITION_SL);
         double tp = PositionGetDouble(POSITION_TP);
         long type = PositionGetInteger(POSITION_TYPE); // 0=BUY, 1=SELL

         double atr = iATR(symbol, PERIOD_M15, ATRPeriod); // NOTE: This is MQL4 style – must be fixed for MQL5
         double trailDistance = atr * ATRMultiplierSL;

         if(type == POSITION_TYPE_BUY)
           {
            double currentBid = SymbolInfoDouble(symbol, SYMBOL_BID);
            double newSL = currentBid - trailDistance;
            if(newSL > sl && currentBid > entryPrice)
               trade.PositionModify(ticket, newSL, tp);
            if(PositionGetDouble(POSITION_PROFIT) > atr && sl < entryPrice)
               trade.PositionModify(ticket, entryPrice, tp);
           }
         else
            if(type == POSITION_TYPE_SELL)
              {
               double currentAsk = SymbolInfoDouble(symbol, SYMBOL_ASK);
               double newSL = currentAsk + trailDistance;
               if(newSL < sl && currentAsk < entryPrice)
                  trade.PositionModify(ticket, newSL, tp);
               if(PositionGetDouble(POSITION_PROFIT) > atr && sl > entryPrice)
                  trade.PositionModify(ticket, entryPrice, tp);
              }
        }
     }
  }
//+------------------------------------------------------------------+
