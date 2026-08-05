//+------------------------------------------------------------------+
//| Calcula el volumen (lotes) correspondiente a un porcentaje       |
//| específico del margen libre para una orden de mercado.           |
//+------------------------------------------------------------------+
double CalculateVolumeByPercent(double percent)
  {
   // Validar que el porcentaje ingresado sea mayor a 0
   if(percent <= 0.0)
     {
      Print("El porcentaje debe ser mayor a 0.");
      return 0.0;
     }

   // 1. Obtener el margen libre y calcular el objetivo
   double free_margin = AccountInfoDouble(ACCOUNT_MARGIN_FREE);
   double target_margin = free_margin * (percent / 100.0);
   
   // 2. Obtener el precio actual (Ask se usa para simular el margen de compra)
   double current_price = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
   
   // 3. Calcular el margen exacto requerido para 1 lote estándar
   double margin_per_lot = 0.0;
   
   if(!OrderCalcMargin(ORDER_TYPE_BUY, _Symbol, 1.0, current_price, margin_per_lot))
     {
      Print("Error calculando el margen para 1 lote. Código de error: ", GetLastError());
      return 0.0;
     }
     
   if(margin_per_lot <= 0.0)
     {
      Print("El margen requerido calculado es 0 o menor.");
      return 0.0;
     }
      
   // 4. Calcular el volumen bruto
   double raw_volume = target_margin / margin_per_lot;
   
   // 5. Obtener las restricciones de volumen del símbolo
   double min_volume  = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MIN);
   double max_volume  = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MAX);
   double volume_step = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_STEP);
   
   // 6. Normalizar el volumen según el step del símbolo
   double steps = MathFloor(raw_volume / volume_step);
   double final_volume = steps * volume_step;
   
   // Prevenir errores de imprecisión en coma flotante
   int digits = (int)MathRound(MathAbs(MathLog10(volume_step)));
   final_volume = NormalizeDouble(final_volume, digits);
   
   // 7. Aplicar los límites de lote máximo y mínimo
   if(final_volume > max_volume)
     {
      final_volume = max_volume;
     }
     
   if(final_volume < min_volume)
     {
      Print("El volumen resultante (", final_volume, ") es menor al mínimo permitido (", min_volume, ").");
      return 0.0; 
     }
     
   return final_volume;
  }
