//+------------------------------------------------------------------+
//| Calcula el volumen (lotes) correspondiente a un porcentaje       |
//| específico del margen libre para una orden de mercado.           |
//+------------------------------------------------------------------+
double CalculateVolumeByPercent(double percent)
  {
   if(percent <= 0.0)
     {
      Print("Error: El porcentaje debe ser mayor a 0.");
      return 0.0;
     }

   // 1. Obtener el margen libre
   double free_margin = AccountInfoDouble(ACCOUNT_MARGIN_FREE);
   double target_margin = free_margin * (percent / 100.0);

   // 2. Obtener el precio Ask actual como referencia
   double current_price = SymbolInfoDouble(_Symbol, SYMBOL_ASK);

   // 3. Calcular el margen exacto requerido para 1 lote estándar
   double margin_per_lot = 0.0;
   if(!OrderCalcMargin(ORDER_TYPE_BUY, _Symbol, 1.0, current_price, margin_per_lot) || margin_per_lot <= 0.0)
     {
      Print("Error: No se pudo calcular el margen requerido por lote. Código: ", GetLastError());
      return 0.0;
     }

   // 4. Calcular el volumen bruto
   double raw_volume = target_margin / margin_per_lot;

   // 5. Obtener restricciones del broker para el símbolo
   double min_volume   = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MIN);
   double max_volume   = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MAX);
   double volume_step  = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_STEP);
   double limit_volume = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_LIMIT);

   if(min_volume <= 0.0 || volume_step <= 0.0)
     {
      Print("Error: Parámetros de volumen del símbolo inválidos.");
      return 0.0;
     }

   // 6. Si el volumen calculado no alcanza el mínimo, abortar
   if(raw_volume < min_volume)
     {
      PrintFormat("Volumen insuficiente: Calculado (%.4f) < Mínimo permitido (%.4f)", raw_volume, min_volume);
      return 0.0;
     }

   // 7. Normalizar el volumen según el paso (step) relativo a min_volume
   double steps = MathFloor((raw_volume - min_volume) / volume_step);
   double final_volume = min_volume + (steps * volume_step);

   // 8. Redondear con precisión matemática exacta para evitar imprecisión flotante
   int digits = 0;
   if(volume_step < 1.0)
     {
      digits = (int)MathCeil(-MathLog10(volume_step));
     }
   final_volume = NormalizeDouble(final_volume, digits);

   // 9. Validar límite máximo por orden
   if(final_volume > max_volume)
     {
      final_volume = max_volume;
     }

   // 10. Validar límite máximo acumulado por símbolo (si el broker lo impone)
   if(limit_volume > 0.0 && final_volume > limit_volume)
     {
      final_volume = limit_volume;
     }

   // 11. Verificación final de seguridad
   if(final_volume < min_volume)
     {
      PrintFormat("Volumen final (%.4f) menor al mínimo permitido (%.4f).", final_volume, min_volume);
      return 0.0;
     }

   return final_volume;
  }
