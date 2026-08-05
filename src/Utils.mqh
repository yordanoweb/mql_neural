double CalculateVolumeByPercent(double            percent,
                                ENUM_ORDER_TYPE   orderType = ORDER_TYPE_BUY)
  {
//--- 1. Validate input
   if(percent <= 0.0 || percent > 100.0)
     {
      PrintFormat("%s: 'percent' must be in the range (0, 100]. Received: %.4f",
                  __FUNCTION__, percent);
      return 0.0;
     }
 
   string symbol = Symbol();
 
//--- 2. Get available free margin
   double freeMargin = AccountInfoDouble(ACCOUNT_MARGIN_FREE);
   if(freeMargin <= 0.0)
     {
      PrintFormat("%s: No free margin available (%.2f).", __FUNCTION__, freeMargin);
      return 0.0;
     }
 
//--- 3. Choose price according to order direction
   bool isBuy = (orderType == ORDER_TYPE_BUY       ||
                 orderType == ORDER_TYPE_BUY_LIMIT  ||
                 orderType == ORDER_TYPE_BUY_STOP   ||
                 orderType == ORDER_TYPE_BUY_STOP_LIMIT);
 
   double price = isBuy ? SymbolInfoDouble(symbol, SYMBOL_ASK)
                        : SymbolInfoDouble(symbol, SYMBOL_BID);
 
   if(price <= 0.0)
     {
      PrintFormat("%s: Could not get a valid price for %s.", __FUNCTION__, symbol);
      return 0.0;
     }
 
//--- 4. Margin required to open exactly 1.0 lot
   double marginPer1Lot = 0.0;
   if(!OrderCalcMargin(orderType, symbol, 1.0, price, marginPer1Lot) ||
      marginPer1Lot <= 0.0)
     {
      PrintFormat("%s: OrderCalcMargin() failed for %s. Error: %d",
                  __FUNCTION__, symbol, GetLastError());
      return 0.0;
     }
 
//--- 5. Calculate raw lot from the margin budget
   double marginBudget = freeMargin * (percent / 100.0);
   double rawLot       = marginBudget / marginPer1Lot;
 
//--- 6. Fetch symbol volume constraints
   double lotStep = SymbolInfoDouble(symbol, SYMBOL_VOLUME_STEP);
   double lotMin  = SymbolInfoDouble(symbol, SYMBOL_VOLUME_MIN);
   double lotMax  = SymbolInfoDouble(symbol, SYMBOL_VOLUME_MAX);
 
   if(lotStep <= 0.0)
     {
      PrintFormat("%s: Invalid VOLUME_STEP (%.8f) for %s.", __FUNCTION__, lotStep, symbol);
      return 0.0;
     }
 
//--- 7. Floor to the nearest lot step (never overshoot the budget)
   double lot = MathFloor(rawLot / lotStep) * lotStep;
 
//--- 8. Normalise decimal places to match the step
   int digits = (int)MathRound(MathLog10(1.0 / lotStep));   // e.g. step=0.01 → 2 dp
   lot        = NormalizeDouble(lot, digits);
 
//--- 9. Enforce symbol min / max limits
   if(lot < lotMin)
     {
      PrintFormat("%s: Calculated lot %.5f is below VOLUME_MIN %.5f for %.1f%% of margin "
                  "(free: %.2f, budget: %.2f, margin/lot: %.2f). Returning 0.",
                  __FUNCTION__, lot, lotMin, percent, freeMargin, marginBudget, marginPer1Lot);
      return 0.0;
     }
 
   if(lot > lotMax)
     {
      PrintFormat("%s: Clamping lot from %.5f to VOLUME_MAX %.5f.", __FUNCTION__, lot, lotMax);
      lot = lotMax;
     }
 
   PrintFormat("%s: percent=%.2f%% | freeMargin=%.2f | budget=%.2f | "
               "marginPerLot=%.2f | lots=%.5f",
               __FUNCTION__, percent, freeMargin, marginBudget, marginPer1Lot, lot);
 
   return lot;
  }
