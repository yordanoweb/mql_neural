//+------------------------------------------------------------------+
//|                                            PrintSymbolPipInfo.mq5|
//+------------------------------------------------------------------+
#property strict
#property script_show_inputs

void OnStart()
{
   string symbol = _Symbol;
   int digits = (int)SymbolInfoInteger(symbol, SYMBOL_DIGITS);
   double point = SymbolInfoDouble(symbol, SYMBOL_POINT);
   double tick_size = SymbolInfoDouble(symbol, SYMBOL_TRADE_TICK_SIZE);
   double tick_value = SymbolInfoDouble(symbol, SYMBOL_TRADE_TICK_VALUE);
   double pip_unit = point * ((digits == 3 || digits == 5) ? 10.0 : 1.0);

   Print("\n--- Symbol Pip Info ---");
   Print("Symbol: ", symbol);
   Print("Digits: ", digits);
   Print("Point: ", DoubleToString(point, digits));
   Print("Pip Unit: ", DoubleToString(pip_unit, digits));
   Print("Tick Size: ", DoubleToString(tick_size, digits));
   Print("Tick Value: ", DoubleToString(tick_value, 6));
   Print("Rule: pip = point * 10 when digits are 3 or 5; otherwise pip = point");
}
