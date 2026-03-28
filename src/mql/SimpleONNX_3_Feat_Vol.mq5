// PATCHED: SimpleONNX_3_Feat.mq5 (RSI -> Volume Effort/Result)

// --- ADD INPUT ---
input int InpVolWindow = 20;

// --- ADD FUNCTIONS ---
double GetRelativeVolume(int i)
{
   double sum = 0.0;
   for(int j = i - InpVolWindow; j < i; j++)
      sum += (double)Volume[j];

   double mean = sum / InpVolWindow;
   if(mean <= 0.0) mean = 1.0;

   double v = (double)Volume[i] / mean;

   if(v > 5.0) v = 5.0;
   if(v < 0.0) v = 0.0;

   return v;
}

double GetEffortResult(int i)
{
   double range = High[i] - Low[i];
   if(range <= 0.0) return 0.0;

   double body = MathAbs(Close[i] - Open[i]);
   double efficiency = body / range;

   double rel_vol = GetRelativeVolume(i);

   double result = rel_vol * efficiency;

   if(result > 5.0) result = 5.0;

   return result;
}

// --- REPLACE RSI USAGE ---
// OLD:
// feat_rsi = rsi_buffer[i] / 100.0;

// NEW:
feat_vol = GetEffortResult(i);

// Ensure feature order remains:
// body, range, volume
