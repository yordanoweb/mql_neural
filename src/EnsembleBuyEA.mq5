//+------------------------------------------------------------------+
//|                                            EnsembleBuyEA.mq5     |
//|  EA de Ensemble Buy-Only con 5 modelos ONNX                      |
//|  Carga modelos A-E, calcula features por modelo, agrega probs    |
//+------------------------------------------------------------------+
#property copyright "Ensemble EA"
#property version   "1.00"
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
//+------------------------------------------------------------------+
//| ENUMS                                                            |
//+------------------------------------------------------------------+
enum ENUM_ENSEMBLE_MODE
{
   ENSEMBLE_MEAN,      // Media simple
   ENSEMBLE_WEIGHTED,  // Media ponderada
   ENSEMBLE_MEDIAN,    // Mediana
   ENSEMBLE_MAJORITY,  // Voto mayoritario
   ENSEMBLE_TRIMMEAN   // Media truncada (quita max y min)
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
input int    InpMaxBarsWait = 100;     // Max barras esperando datos

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
   long     onnxHandle;     // Handle ONNX (-1 si no cargado)
   double   weight;         // Peso en agregacion
   bool     loaded;         // Cargado correctamente?
   string   perspective;    // Perspectiva (metadata)
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
   Print("=== EnsembleBuyEA Iniciando ===");

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

   // Esperar a que haya suficientes barras
   if(Bars(_Symbol, PERIOD_CURRENT) < 50)
   {
      Print("Esperando mas barras de historico...");
      return INIT_SUCCEEDED; // Seguira intentando en OnTick
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
   // Liberar modelos ONNX
   for(int i = 0; i < 5; i++)
   {
      if(g_models[i].onnxHandle != INVALID_HANDLE)
      {
         OnnxRelease(g_models[i].onnxHandle);
         g_models[i].onnxHandle = INVALID_HANDLE;
      }
   }

   // Liberar indicadores
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

   // Solo procesar al cerrar una nueva barra (modo barra cerrada)
   if(currentBarTime == g_lastBarTime)
      return;

   g_lastBarTime = currentBarTime;
   g_barsProcessed++;
   g_logCounter++;

   // Verificar que tenemos suficientes barras
   int totalBars = Bars(_Symbol, PERIOD_CURRENT);
   if(totalBars < 50)
   {
      if(g_barsProcessed % 10 == 0)
         Print("Esperando barras... Actual: ", totalBars, "/50");
      return;
   }

   // Actualizar datos de indicadores
   if(!CopyIndicatorBuffers())
   {
      Print("ERROR: No se pudieron copiar buffers de indicadores");
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

      double modelInput[];
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

   // Gestion de posiciones existentes (trailing stop)
   if(InpUseTrailingStop)
      ManageTrailingStops();

   // Decision de trading
   int openPositions = CountOpenPositions(POSITION_TYPE_BUY);

   bool shouldBuy = false;

   // Condiciones de entrada
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
}

//+------------------------------------------------------------------+
//| CARGAR MODELO ONNX                                               |
//+------------------------------------------------------------------+
bool LoadONNXModel(int idx)
{
   string fullPath = g_models[idx].path;

   // Verificar si existe en MQL5/Files
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

      // Intentar con ruta completa
      handle = OnnxCreate(fullPath, ONNX_DEFAULT);
      if(handle == INVALID_HANDLE)
      {
         Print("ERROR: Tampoco funciona con ruta completa: ", fullPath);
         return false;
      }
   }

   g_models[idx].onnxHandle = handle;
   g_models[idx].loaded = true;

   Print("Modelo ", g_models[idx].id, " cargado OK. Input size: ", g_models[idx].inputSize);
   return true;
}

//+------------------------------------------------------------------+
//| COPIAR BUFFERS DE INDICADORES                                    |
//+------------------------------------------------------------------+
bool CopyIndicatorBuffers()
{
   // Los indicadores se actualizan automaticamente por el handle
   // No necesitamos copiar manualmente, usaremos CopyBuffer en PrepareFeatures
   return true;
}

//+------------------------------------------------------------------+
//| PREPARAR FEATURES PARA UN MODELO                                 |
//+------------------------------------------------------------------+
bool PrepareFeatures(int modelIdx, double &outFeatures[])
{
   ModelConfig m = g_models[modelIdx];
   int w = m.window;

   // Necesitamos w barras de historico + buffer para calculos
   int requiredBars = w + InpATR_Period + 5;
   if(Bars(_Symbol, PERIOD_CURRENT) < requiredBars)
      return false;

   ArrayResize(outFeatures, m.inputSize);
   ArrayInitialize(outFeatures, 0.0);

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

   // Obtener ATR (para modelo D)
   double atrValues[];
   ArraySetAsSeries(atrValues, true);
   if(CopyBuffer(g_handleATR, 0, 0, requiredBars, atrValues) < requiredBars) return false;

   int featIdx = 0;

   for(int i = 0; i < w; i++)
   {
      int barIdx = i; // 0 = vela actual, 1 = anterior, etc.

      double o = opens[barIdx];
      double h = highs[barIdx];
      double l = lows[barIdx];
      double c = closes[barIdx];
      double body = c - o;
      double range = h - l;
      double rsi = rsiValues[barIdx] / 100.0; // Normalizar 0-1

      if(m.id == "A" || m.id == "B" || m.id == "C")
      {
         // STANDARD: body, range, rsi
         outFeatures[featIdx++] = body;
         outFeatures[featIdx++] = range;
         outFeatures[featIdx++] = rsi;
      }
      else if(m.id == "D")
      {
         // STRUCTURE: body_ratio, range_norm, rsi
         double bodyRatio = (range > 0.0000001) ? (body / range) : 0.0;
         double atr = atrValues[barIdx];
         double rangeNorm = (atr > 0.0000001) ? (range / atr) : 1.0;

         outFeatures[featIdx++] = bodyRatio;
         outFeatures[featIdx++] = rangeNorm;
         outFeatures[featIdx++] = rsi;
      }
      else if(m.id == "E")
      {
         // VOLATILITY: range, range_expansion, rsi
         double prevRange = (barIdx + 1 < requiredBars) ? (highs[barIdx + 1] - lows[barIdx + 1]) : range;
         double rangeExpansion = (prevRange > 0.0000001) ? (range / prevRange) : 1.0;

         outFeatures[featIdx++] = range;
         outFeatures[featIdx++] = rangeExpansion;
         outFeatures[featIdx++] = rsi;
      }
   }

   return (featIdx == m.inputSize);
}

//+------------------------------------------------------------------+
//| EJECUTAR INFERENCIA ONNX                                         |
//+------------------------------------------------------------------+
bool RunInference(int modelIdx, const double &inputData[], double &buyProbability)
{
   ModelConfig m = g_models[modelIdx];

   if(!m.loaded || m.onnxHandle == INVALID_HANDLE)
      return false;

   // Crear tensor de entrada [1, inputSize]
   // ONNX en MQL5 espera un array plano para batch=1
   long inputShape[] = {1, m.inputSize};

   // El input ya es float32 en Python, aqui usamos double y ONNX lo maneja
   // MQL5 OnnxRun acepta arrays double para inputs float

   // Output: probabilities [1, 2] -> [prob_class_0, prob_class_1]
   double outputData[2];
   long outputShape[] = {1, 2};

   // Ejecutar inferencia
   // Nota: En MQL5, OnnxRun necesita los shapes como parametros
   // La API puede variar ligeramente segun la version de MT5

   // Metodo 1: Usar OnnxRun con shapes
   bool success = OnnxRun(
      m.onnxHandle,
      ONNX_NO_CONVERSION,
      inputData,      // input
      inputShape,     // input shape [1, N]
      outputData,     // output
      outputShape     // output shape [1, 2]
   );

   if(!success)
   {
      int err = GetLastError();
      Print("ERROR ONNX Run modelo ", m.id, ": ", err);
      return false;
   }

   // outputData[0] = P(Sell/NoBuy), outputData[1] = P(Buy)
   buyProbability = outputData[1];

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
         // Quitar min y max
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
   // Verificar condiciones de trading
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

   // Obtener precios actuales
   double ask = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
   double bid = SymbolInfoDouble(_Symbol, SYMBOL_BID);
   double point = SymbolInfoDouble(_Symbol, SYMBOL_POINT);
   int digits = (int)SymbolInfoInteger(_Symbol, SYMBOL_DIGITS);

   if(ask == 0 || bid == 0)
   {
      Print("ERROR: Precios no disponibles");
      return;
   }

   // Calcular SL y TP
   double sl = (InpSL_Pips > 0) ? NormalizeDouble(ask - InpSL_Pips * point, digits) : 0.0;
   double tp = (InpTP_Pips > 0) ? NormalizeDouble(ask + InpTP_Pips * point, digits) : 0.0;

   // Verificar stops
   double minStopLevel = SymbolInfoInteger(_Symbol, SYMBOL_TRADE_STOPS_LEVEL) * point;
   if(sl > 0 && (ask - sl) < minStopLevel)
      sl = NormalizeDouble(ask - minStopLevel - point, digits);
   if(tp > 0 && (tp - ask) < minStopLevel)
      tp = NormalizeDouble(ask + minStopLevel + point, digits);

   // Preparar comentario con info del ensemble
   string comment = StringFormat("Ens:%.2f|A:%.2f B:%.2f C:%.2f D:%.2f E:%.2f",
      ensembleProb, modelProbs[0], modelProbs[1], modelProbs[2], modelProbs[3], modelProbs[4]);

   // Abrir posicion
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

      // Nuevo SL: bid - trailDistance, pero solo si mejora el actual
      double newSL = NormalizeDouble(bid - trailDistance, digits);

      // Solo mover SL hacia arriba (para BUY)
      if(newSL > openPrice && (currentSL == 0 || newSL > currentSL))
      {
         // Verificar distancia minima
         double minStopLevel = SymbolInfoInteger(_Symbol, SYMBOL_TRADE_STOPS_LEVEL) * point;
         if((bid - newSL) >= minStopLevel)
         {
            g_trade.PositionModify(ticket, newSL, currentTP);
         }
      }
   }
}

//+------------------------------------------------------------------+
//| OBTENER PESOS DESDE METADATA ONNX (opcional)                     |
//+------------------------------------------------------------------+
/*
   Esta funcion puede usarse para leer los pesos optimos desde
   los metadatos del ONNX si se guardaron alli. Requiere parsear
   el JSON de metadata_props del modelo.

   Por simplicidad, los pesos se configuran via inputs del EA.
   Para implementacion automatica, se necesitaria:
   1. Leer metadata con OnnxGetInputName/OnnxGetOutputName
   2. Parsear JSON de metadata_props
   3. Extraer validation.test_precision_buy de cada modelo
   4. Normalizar como pesos
*/
//+------------------------------------------------------------------+
