//+------------------------------------------------------------------+
//|                                            EnsembleBuyEA.mq5     |
//|  EA de Ensemble Buy-Only con 5 modelos ONNX                      |
//|  CORREGIDO: shapes ONNX, vectorf, 2 outputs, referencias         |
//+------------------------------------------------------------------+
#property copyright "Ensemble EA"
#property version   "1.10"
#property strict

#include <Trade\Trade.mqh>

//+------------------------------------------------------------------+
//| INPUTS DEL USUARIO                                               |
//+------------------------------------------------------------------+
input group "=== ARCHIVOS ONNX ==="
input string InpModelA_Path = "model_A_impulse.onnx";     // Modelo A (Impulse)
input string InpModelB_Path = "model_B_swing.onnx";       // Modelo B (Swing)
input string InpModelC_Path = "model_C_trend.onnx";       // Modelo C (Trend)
input string InpModelD_Path = "model_D_structure.onnx";   // Modelo D (Structure)
input string InpModelE_Path = "model_E_volatility.onnx";  // Modelo E (Volatility)

input group "=== AGREGACION ==="
enum ENUM_ENSEMBLE_MODE
{
   ENSEMBLE_MEAN,      // Media simple
   ENSEMBLE_WEIGHTED,  // Media ponderada
   ENSEMBLE_MEDIAN,    // Mediana
   ENSEMBLE_MAJORITY,  // Voto mayoritario
   ENSEMBLE_TRIMMEAN   // Media truncada
};
input ENUM_ENSEMBLE_MODE InpEnsembleMode = ENSEMBLE_WEIGHTED; // Modo de agregacion
input double InpWeightA = 0.10;  // Peso Modelo A
input double InpWeightB = 0.25;  // Peso Modelo B
input double InpWeightC = 0.25;  // Peso Modelo C
input double InpWeightD = 0.20;  // Peso Modelo D
input double InpWeightE = 0.20;  // Peso Modelo E
input double InpConfidenceThreshold = 0.60;  // Umbral confianza BUY (0-1)
input double InpMinConfidenceDiff = 0.05;    // Diferencia minima vs SELL prob

input group "=== GESTION DE RIESGO ==="
input double InpLotSize = 0.01;        // Tamaño de lote
input double InpSL_Pips = 50.0;        // Stop Loss (puntos)
input double InpTP_Pips = 100.0;       // Take Profit (puntos)
input int    InpMaxPositions = 1;      // Max posiciones abiertas simultaneas
input bool   InpUseTrailingStop = true;// Usar trailing stop
input double InpTrailDistance = 30.0;  // Distancia trailing (puntos)

input group "=== PARAMETROS DE FEATURES ==="
input int    InpRSI_Period = 14;       // Periodo RSI
input int    InpATR_Period = 20;       // Periodo ATR (para modelo D)

input group "=== LOGGING ==="
input bool   InpVerbose = true;        // Log detallado
input int    InpLogEveryNBars = 100;   // Log cada N barras

//+------------------------------------------------------------------+
//| ESTRUCTURA DE MODELO                                             |
//+------------------------------------------------------------------+
struct ModelConfig
{
   string   path;           // Ruta archivo .onnx
   string   id;             // ID (A, B, C, D, E)
   string   alias;          // Nombre legible
   int      window;         // Ventana de barras
   int      featureCount;   // Features por barra
   int      inputSize;      // window * featureCount
   long     onnxHandle;     // Handle ONNX
   double   weight;         // Peso en agregacion
   bool     loaded;         // Cargado correctamente?
   string   perspective;    // Perspectiva
   int      outputCount;    // Numero de outputs del modelo ONNX
};

//+------------------------------------------------------------------+
//| VARIABLES GLOBALES                                               |
//+------------------------------------------------------------------+
ModelConfig g_models[5];
CTrade      g_trade;
datetime    g_lastBarTime = 0;
int         g_barsProcessed = 0;
int         g_logCounter = 0;

// Handles de indicadores
int g_handleRSI = INVALID_HANDLE;
int g_handleATR = INVALID_HANDLE;

//+------------------------------------------------------------------+
//| EXPERT INITIALIZATION                                            |
//+------------------------------------------------------------------+
int OnInit()
{
   Print("=== EnsembleBuyEA v1.10 Iniciando ===");

   // Inicializar trade
   g_trade.SetDeviationInPoints(10);
   g_trade.SetTypeFilling(ORDER_FILLING_IOC);
   g_trade.SetAsyncMode(false);

   // Crear handles de indicadores
   g_handleRSI = iRSI(_Symbol, PERIOD_CURRENT, InpRSI_Period, PRICE_CLOSE);
   g_handleATR = iATR(_Symbol, PERIOD_CURRENT, InpATR_Period);

   if(g_handleRSI == INVALID_HANDLE || g_handleATR == INVALID_HANDLE)
   {
      Print("ERROR: No se pudieron crear handles de indicadores");
      return INIT_FAILED;
   }

   // Configurar modelos
   InitModel(0, InpModelA_Path, "A", "impulse", 5, 3, InpWeightA, "microstructure");
   InitModel(1, InpModelB_Path, "B", "swing", 15, 3, InpWeightB, "short_term");
   InitModel(2, InpModelC_Path, "C", "trend", 30, 3, InpWeightC, "medium_term");
   InitModel(3, InpModelD_Path, "D", "structure", 20, 3, InpWeightD, "scale_invariant");
   InitModel(4, InpModelE_Path, "E", "volatility", 20, 3, InpWeightE, "volatility_regime");

   // Cargar modelos ONNX
   int loadedCount = 0;
   for(int i = 0; i < 5; i++)
   {
      if(LoadONNXModel(i))
         loadedCount++;
   }

   if(loadedCount == 0)
   {
      Print("ERROR CRITICO: Ningun modelo ONNX pudo cargarse. Verifique rutas.");
      return INIT_FAILED;
   }

   Print("Modelos cargados: ", loadedCount, "/5");

   if(Bars(_Symbol, PERIOD_CURRENT) < 50)
   {
      Print("Esperando mas barras de historico...");
      return INIT_SUCCEEDED;
   }

   g_lastBarTime = iTime(_Symbol, PERIOD_CURRENT, 0);

   Print("=== EnsembleBuyEA Iniciado correctamente ===");
   return INIT_SUCCEEDED;
}

//+------------------------------------------------------------------+
//| EXPERT DEINITIALIZATION                                          |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
{
   for(int i = 0; i < 5; i++)
   {
      if(g_models[i].onnxHandle != INVALID_HANDLE)
      {
         OnnxRelease(g_models[i].onnxHandle);
         g_models[i].onnxHandle = INVALID_HANDLE;
      }
   }

   if(g_handleRSI != INVALID_HANDLE) IndicatorRelease(g_handleRSI);
   if(g_handleATR != INVALID_HANDLE) IndicatorRelease(g_handleATR);

   Print("=== EnsembleBuyEA Finalizado. Razon: ", reason, " ===");
}

//+------------------------------------------------------------------+
//| EXPERT TICK                                                      |
//+------------------------------------------------------------------+
void OnTick()
{
   datetime currentBarTime = iTime(_Symbol, PERIOD_CURRENT, 0);

   // Solo procesar al cerrar una nueva barra
   if(currentBarTime == g_lastBarTime)
      return;

   g_lastBarTime = currentBarTime;
   g_barsProcessed++;
   g_logCounter++;

   int totalBars = Bars(_Symbol, PERIOD_CURRENT);
   if(totalBars < 50)
   {
      if(g_barsProcessed % 10 == 0)
         Print("Esperando barras... Actual: ", totalBars, "/50");
      return;
   }

   // Ejecutar inferencia en cada modelo
   double probs[5];
   bool allOk = true;

   for(int i = 0; i < 5; i++)
   {
      if(!g_models[i].loaded)
      {
         probs[i] = 0.0;
         continue;
      }

      vectorf modelInput;
      if(!PrepareFeatures(i, modelInput))
      {
         if(InpVerbose)
            Print("WARN: No se pudieron preparar features para modelo ", g_models[i].id);
         probs[i] = 0.0;
         allOk = false;
         continue;
      }

      if(!RunInference(i, modelInput, probs[i]))
      {
         if(InpVerbose)
            Print("WARN: Inferencia fallida para modelo ", g_models[i].id);
         probs[i] = 0.0;
         allOk = false;
         continue;
      }
   }

   // Agregar probabilidades
   double ensembleProb = AggregateProbabilities(probs);
   double sellProb = 1.0 - ensembleProb;

   // Logging periodico
   if(InpVerbose && (g_logCounter >= InpLogEveryNBars || allOk))
   {
      g_logCounter = 0;
      string logMsg = StringFormat(
         "BAR[%d] | A:%.3f B:%.3f C:%.3f D:%.3f E:%.3f | ENSEMBLE:%.3f | SELL:%.3f",
         g_barsProcessed, probs[0], probs[1], probs[2], probs[3], probs[4],
         ensembleProb, sellProb
      );
      Print(logMsg);
   }

   // Gestion de posiciones existentes
   if(InpUseTrailingStop)
      ManageTrailingStops();

   // Decision de trading
   int openPositions = CountOpenPositions(POSITION_TYPE_BUY);

   bool shouldBuy = false;
   if(ensembleProb >= InpConfidenceThreshold &&
      (ensembleProb - sellProb) >= InpMinConfidenceDiff &&
      openPositions < InpMaxPositions &&
      allOk)
   {
      shouldBuy = true;
   }

   if(shouldBuy)
   {
      OpenBuyPosition(ensembleProb, probs);
   }
}

//+------------------------------------------------------------------+
//| INICIALIZAR CONFIGURACION DE MODELO                              |
//+------------------------------------------------------------------+
void InitModel(int idx, string path, string id, string alias, int window,
               int featCount, double weight, string perspective)
{
   g_models[idx].path = path;
   g_models[idx].id = id;
   g_models[idx].alias = alias;
   g_models[idx].window = window;
   g_models[idx].featureCount = featCount;
   g_models[idx].inputSize = window * featCount;
   g_models[idx].onnxHandle = INVALID_HANDLE;
   g_models[idx].weight = weight;
   g_models[idx].loaded = false;
   g_models[idx].perspective = perspective;
   g_models[idx].outputCount = 0;
}

//+------------------------------------------------------------------+
//| CARGAR MODELO ONNX + DEFINIR SHAPES                              |
//+------------------------------------------------------------------+
bool LoadONNXModel(int idx)
{
   string fullPath = g_models[idx].path;

   // Extraer nombre de archivo
   string filename = fullPath;
   int sepPos = StringFind(fullPath, "\\", 0);
   if(sepPos == -1) sepPos = StringFind(fullPath, "/", 0);
   if(sepPos != -1)
      filename = StringSubstr(fullPath, sepPos + 1);

   long handle = OnnxCreate(filename, ONNX_DEFAULT);

   if(handle == INVALID_HANDLE)
   {
      int err = GetLastError();
      Print("ERROR cargando modelo ", g_models[idx].id, " (", filename, "): ", err);

      handle = OnnxCreate(fullPath, ONNX_DEFAULT);
      if(handle == INVALID_HANDLE)
      {
         Print("ERROR: Tampoco funciona con ruta completa: ", fullPath);
         return false;
      }
   }

   g_models[idx].onnxHandle = handle;

   // Verificar inputs/outputs del modelo
   long inputCount = OnnxGetInputCount(handle);
   long outputCount = OnnxGetOutputCount(handle);
   g_models[idx].outputCount = (int)outputCount;

   Print("Modelo ", g_models[idx].id, " ONNX info: inputs=", inputCount, ", outputs=", outputCount);

   // Mostrar nombres de inputs/outputs para debug
   for(int i = 0; i < (int)inputCount; i++)
   {
      string inName = OnnxGetInputName(handle, i);
      Print("  Input[", i, "]: ", inName);
   }
   for(int i = 0; i < (int)outputCount; i++)
   {
      string outName = OnnxGetOutputName(handle, i);
      Print("  Output[", i, "]: ", outName);
   }

   // Definir shape del input [1, inputSize]
   long inputShape[] = {1, g_models[idx].inputSize};
   if(!OnnxSetInputShape(handle, 0, inputShape))
   {
      Print("ERROR OnnxSetInputShape modelo ", g_models[idx].id, ": ", GetLastError());
      OnnxRelease(handle);
      return false;
   }

   // Definir shapes de outputs
   // Output 0: label (int64) -> shape [1]
   long outputShape0[] = {1};
   if(!OnnxSetOutputShape(handle, 0, outputShape0))
   {
      Print("ERROR OnnxSetOutputShape[0] modelo ", g_models[idx].id, ": ", GetLastError());
      OnnxRelease(handle);
      return false;
   }

   // Output 1: probabilities (float32) -> shape [1, 2]
   if(outputCount >= 2)
   {
      long outputShape1[] = {1, 2};
      if(!OnnxSetOutputShape(handle, 1, outputShape1))
      {
         Print("ERROR OnnxSetOutputShape[1] modelo ", g_models[idx].id, ": ", GetLastError());
         OnnxRelease(handle);
         return false;
      }
   }

   g_models[idx].loaded = true;
   Print("Modelo ", g_models[idx].id, " cargado OK. Input size: ", g_models[idx].inputSize,
         ", Outputs: ", outputCount);
   return true;
}

//+------------------------------------------------------------------+
//| PREPARAR FEATURES PARA UN MODELO (devuelve vectorf)              |
//+------------------------------------------------------------------+
bool PrepareFeatures(int modelIdx, vectorf &outVec)
{
   ModelConfig m = g_models[modelIdx];  // <-- REFERENCIA, no copia
   int w = m.window;

   int requiredBars = w + InpATR_Period + 5;
   if(Bars(_Symbol, PERIOD_CURRENT) < requiredBars)
      return false;

   outVec.Resize(m.inputSize);

   // Obtener datos de precios
   double opens[], highs[], lows[], closes[];
   ArraySetAsSeries(opens, true);
   ArraySetAsSeries(highs, true);
   ArraySetAsSeries(lows, true);
   ArraySetAsSeries(closes, true);

   if(CopyOpen(_Symbol, PERIOD_CURRENT, 0, requiredBars, opens) < requiredBars) return false;
   if(CopyHigh(_Symbol, PERIOD_CURRENT, 0, requiredBars, highs) < requiredBars) return false;
   if(CopyLow(_Symbol, PERIOD_CURRENT, 0, requiredBars, lows) < requiredBars) return false;
   if(CopyClose(_Symbol, PERIOD_CURRENT, 0, requiredBars, closes) < requiredBars) return false;

   // Obtener RSI
   double rsiValues[];
   ArraySetAsSeries(rsiValues, true);
   if(CopyBuffer(g_handleRSI, 0, 0, requiredBars, rsiValues) < requiredBars) return false;

   // Obtener ATR
   double atrValues[];
   ArraySetAsSeries(atrValues, true);
   if(CopyBuffer(g_handleATR, 0, 0, requiredBars, atrValues) < requiredBars) return false;

   int featIdx = 0;

   for(int i = 0; i < w; i++)
   {
      int barIdx = i;

      double o = opens[barIdx];
      double h = highs[barIdx];
      double l = lows[barIdx];
      double c = closes[barIdx];
      double body = c - o;
      double range = h - l;
      double rsi = rsiValues[barIdx] / 100.0;

      if(m.id == "A" || m.id == "B" || m.id == "C")
      {
         outVec[featIdx++] = (float)body;
         outVec[featIdx++] = (float)range;
         outVec[featIdx++] = (float)rsi;
      }
      else if(m.id == "D")
      {
         double bodyRatio = (range > 0.0000001) ? (body / range) : 0.0;
         double atr = atrValues[barIdx];
         double rangeNorm = (atr > 0.0000001) ? (range / atr) : 1.0;

         outVec[featIdx++] = (float)bodyRatio;
         outVec[featIdx++] = (float)rangeNorm;
         outVec[featIdx++] = (float)rsi;
      }
      else if(m.id == "E")
      {
         double prevRange = (barIdx + 1 < requiredBars) ? (highs[barIdx + 1] - lows[barIdx + 1]) : range;
         double rangeExpansion = (prevRange > 0.0000001) ? (range / prevRange) : 1.0;

         outVec[featIdx++] = (float)range;
         outVec[featIdx++] = (float)rangeExpansion;
         outVec[featIdx++] = (float)rsi;
      }
   }

   return (featIdx == m.inputSize);
}

//+------------------------------------------------------------------+
//| EJECUTAR INFERENCIA ONNX                                         |
//| El modelo skl2onnx tiene 2 outputs: label + probabilities        |
//+------------------------------------------------------------------+
bool RunInference(int modelIdx, const vectorf &inputVec, double &buyProbability)
{
   ModelConfig m = g_models[modelIdx];  // <-- REFERENCIA, no copia

   if(!m.loaded || m.onnxHandle == INVALID_HANDLE)
      return false;

   // FIX: Usar un array dinámico estándar para tipos enteros (long)
   // Output 0: label predicha (int64) - no la usamos pero hay que pasarla
   long outputLabel[];
   ArrayResize(outputLabel, 1);

   // Output 1: probabilidades [P(class0), P(class1)]
   vectorf outputProbs;
   outputProbs.Resize(2);

   bool success = false;

   if(m.outputCount >= 2)
   {
      // Modelo con 2 outputs: pasar ambos (array 'long' y vector 'float')
      success = OnnxRun(m.onnxHandle, ONNX_NO_CONVERSION, inputVec, outputLabel, outputProbs);
   }
   else
   {
      // Fallback: solo 1 output
      success = OnnxRun(m.onnxHandle, ONNX_NO_CONVERSION, inputVec, outputProbs);
   }

   if(!success)
   {
      int err = GetLastError();
      Print("ERROR ONNX Run modelo ", m.id, ": ", err);
      return false;
   }

   // outputProbs[0] = P(Sell/NoBuy), outputProbs[1] = P(Buy)
   buyProbability = (double)outputProbs[1];

   return true;
}

//+------------------------------------------------------------------+
//| AGREGAR PROBABILIDADES                                           |
//+------------------------------------------------------------------+
double AggregateProbabilities(const double &probs[])
{
   double result = 0.0;
   double totalWeight = 0.0;
   int count = 0;

   switch(InpEnsembleMode)
   {
      case ENSEMBLE_MEAN:
      {
         for(int i = 0; i < 5; i++)
         {
            if(g_models[i].loaded)
            {
               result += probs[i];
               count++;
            }
         }
         return (count > 0) ? result / count : 0.0;
      }

      case ENSEMBLE_WEIGHTED:
      {
         for(int i = 0; i < 5; i++)
         {
            if(g_models[i].loaded)
            {
               result += probs[i] * g_models[i].weight;
               totalWeight += g_models[i].weight;
            }
         }
         return (totalWeight > 0) ? result / totalWeight : 0.0;
      }

      case ENSEMBLE_MEDIAN:
      {
         double validProbs[];
         ArrayResize(validProbs, 0);
         for(int i = 0; i < 5; i++)
         {
            if(g_models[i].loaded)
            {
               int sz = ArraySize(validProbs);
               ArrayResize(validProbs, sz + 1);
               validProbs[sz] = probs[i];
            }
         }
         int n = ArraySize(validProbs);
         if(n == 0) return 0.0;
         ArraySort(validProbs);
         if(n % 2 == 1)
            return validProbs[n / 2];
         else
            return (validProbs[n / 2 - 1] + validProbs[n / 2]) / 2.0;
      }

      case ENSEMBLE_MAJORITY:
      {
         int buyVotes = 0;
         int totalVotes = 0;
         for(int i = 0; i < 5; i++)
         {
            if(g_models[i].loaded)
            {
               if(probs[i] > 0.5) buyVotes++;
               totalVotes++;
            }
         }
         return (totalVotes > 0) ? (double)buyVotes / totalVotes : 0.0;
      }

      case ENSEMBLE_TRIMMEAN:
      {
         double validProbs[];
         ArrayResize(validProbs, 0);
         for(int i = 0; i < 5; i++)
         {
            if(g_models[i].loaded)
            {
               int sz = ArraySize(validProbs);
               ArrayResize(validProbs, sz + 1);
               validProbs[sz] = probs[i];
            }
         }
         int n = ArraySize(validProbs);
         if(n <= 2) return (n > 0) ? validProbs[0] : 0.0;
         ArraySort(validProbs);
         double sum = 0;
         for(int i = 1; i < n - 1; i++)
            sum += validProbs[i];
         return sum / (n - 2);
      }
   }

   return 0.0;
}

//+------------------------------------------------------------------+
//| ABRIR POSICION BUY                                               |
//+------------------------------------------------------------------+
void OpenBuyPosition(double ensembleProb, const double &modelProbs[])
{
   if(!TerminalInfoInteger(TERMINAL_TRADE_ALLOWED))
   {
      Print("Trading no permitido en este terminal");
      return;
   }

   if(!MQLInfoInteger(MQL_TRADE_ALLOWED))
   {
      Print("Trading no permitido para este EA");
      return;
   }

   double ask = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
   double bid = SymbolInfoDouble(_Symbol, SYMBOL_BID);
   double point = SymbolInfoDouble(_Symbol, SYMBOL_POINT);
   int digits = (int)SymbolInfoInteger(_Symbol, SYMBOL_DIGITS);

   if(ask == 0 || bid == 0)
   {
      Print("ERROR: Precios no disponibles");
      return;
   }

   double sl = (InpSL_Pips > 0) ? NormalizeDouble(ask - InpSL_Pips * point, digits) : 0.0;
   double tp = (InpTP_Pips > 0) ? NormalizeDouble(ask + InpTP_Pips * point, digits) : 0.0;

   double minStopLevel = SymbolInfoInteger(_Symbol, SYMBOL_TRADE_STOPS_LEVEL) * point;
   if(sl > 0 && (ask - sl) < minStopLevel)
      sl = NormalizeDouble(ask - minStopLevel - point, digits);
   if(tp > 0 && (tp - ask) < minStopLevel)
      tp = NormalizeDouble(ask + minStopLevel + point, digits);

   string comment = StringFormat("Ens:%.2f|A:%.2f B:%.2f C:%.2f D:%.2f E:%.2f",
      ensembleProb, modelProbs[0], modelProbs[1], modelProbs[2], modelProbs[3], modelProbs[4]);

   if(!g_trade.Buy(InpLotSize, _Symbol, ask, sl, tp, comment))
   {
      Print("ERROR abriendo BUY: ", GetLastError());
   }
   else
   {
      Print("BUY ABIERTO | Ask: ", ask, " | SL: ", sl, " | TP: ", tp, " | ", comment);
   }
}

//+------------------------------------------------------------------+
//| CONTAR POSICIONES ABIERTAS                                       |
//+------------------------------------------------------------------+
int CountOpenPositions(ENUM_POSITION_TYPE posType)
{
   int count = 0;
   for(int i = PositionsTotal() - 1; i >= 0; i--)
   {
      ulong ticket = PositionGetTicket(i);
      if(ticket <= 0) continue;

      if(PositionGetString(POSITION_SYMBOL) == _Symbol &&
         PositionGetInteger(POSITION_MAGIC) == g_trade.RequestMagic() &&
         PositionGetInteger(POSITION_TYPE) == posType)
      {
         count++;
      }
   }
   return count;
}

//+------------------------------------------------------------------+
//| GESTIONAR TRAILING STOP                                          |
//+------------------------------------------------------------------+
void ManageTrailingStops()
{
   double point = SymbolInfoDouble(_Symbol, SYMBOL_POINT);
   int digits = (int)SymbolInfoInteger(_Symbol, SYMBOL_DIGITS);
   double trailDistance = InpTrailDistance * point;

   for(int i = PositionsTotal() - 1; i >= 0; i--)
   {
      ulong ticket = PositionGetTicket(i);
      if(ticket <= 0) continue;

      if(PositionGetString(POSITION_SYMBOL) != _Symbol) continue;
      if(PositionGetInteger(POSITION_MAGIC) != g_trade.RequestMagic()) continue;
      if(PositionGetInteger(POSITION_TYPE) != POSITION_TYPE_BUY) continue;

      double openPrice = PositionGetDouble(POSITION_PRICE_OPEN);
      double currentSL = PositionGetDouble(POSITION_SL);
      double currentTP = PositionGetDouble(POSITION_TP);
      double bid = SymbolInfoDouble(_Symbol, SYMBOL_BID);

      double newSL = NormalizeDouble(bid - trailDistance, digits);

      if(newSL > openPrice && (currentSL == 0 || newSL > currentSL))
      {
         double minStopLevel = SymbolInfoInteger(_Symbol, SYMBOL_TRADE_STOPS_LEVEL) * point;
         if((bid - newSL) >= minStopLevel)
         {
            g_trade.PositionModify(ticket, newSL, currentTP);
         }
      }
   }
}
//+------------------------------------------------------------------+
