//+------------------------------------------------------------------+
//|                                            EnsembleBuyEA.mq5     |
//|  EA de Ensemble Buy-Only con 5 modelos ONNX                      |
//|  CORREGIDO: shapes ONNX, vectorf, 2 outputs, referencias         |
//+------------------------------------------------------------------+
#property copyright "Ensemble EA"
#property version   "1.10"
#property strict

#include <Trade\Trade.mqh>

#resource "\\Files\\model_A_impulse.onnx" as uchar res_model_a[];
#resource "\\Files\\model_B_swing.onnx" as uchar res_model_b[];
#resource "\\Files\\model_C_trend.onnx" as uchar res_model_c[];
#resource "\\Files\\model_D_structure.onnx" as uchar res_model_d[];
#resource "\\Files\\model_E_volatility.onnx" as uchar res_model_e[];

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
input double InpConfidenceThreshold = 0.5;  // Umbral confianza BUY (0-1)
input double InpConfidenceStep = 0.002;  // Paso de ajuste del umbral de confianza
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

input group "=== Telegram ===";
input bool   InpEnableNotifications  = true; // Enable Telegram notifications
input string InpTelegramBotToken     = "";    // Your Telegram bot bot_token
input string InpTelegramChatID       = "";    // Your Telegram chat_id

input group "=== Debug ===";
input bool   InpBacktest = false;

//+------------------------------------------------------------------+
//| ESTRUCTURA DE MODELO                                             |
//+------------------------------------------------------------------+
struct ModelConfig
  {
   string            path;           // Ruta archivo .onnx
   string            id;             // ID (A, B, C, D, E)
   string            alias;          // Nombre legible
   int               window;         // Ventana de barras
   int               featureCount;   // Features por barra
   int               inputSize;      // window * featureCount
   long              onnxHandle;     // Handle ONNX
   double            weight;         // Peso en agregacion
   bool              loaded;         // Cargado correctamente?
   string            perspective;    // Perspectiva
   int               outputCount;    // Numero de outputs del modelo ONNX
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

ulong g_magic_number = 0;

double g_confidence_threshold = 0.5; // Umbral de confianza para ejecutar operaciones

//+------------------------------------------------------------------+
//| EXPERT INITIALIZATION                                            |
//+------------------------------------------------------------------+
int OnInit()
  {
   Log("=== EnsembleBuyEA v1.10 Iniciando ===");

// Inicializar trade
   g_trade.SetDeviationInPoints(10);
   g_trade.SetTypeFilling(ORDER_FILLING_IOC);
   g_trade.SetAsyncMode(false);

// Crear handles de indicadores
   g_handleRSI = iRSI(_Symbol, PERIOD_CURRENT, InpRSI_Period, PRICE_CLOSE);
   g_handleATR = iATR(_Symbol, PERIOD_CURRENT, InpATR_Period);

   if(g_handleRSI == INVALID_HANDLE || g_handleATR == INVALID_HANDLE)
     {
      Log("ERROR: No se pudieron crear handles de indicadores");
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
      Log("ERROR CRITICO: Ningun modelo ONNX pudo cargarse. Verifique rutas.");
      return INIT_FAILED;
     }

   Log("Modelos cargados: " + IntegerToString(loadedCount) + "/5");

   if(Bars(_Symbol, PERIOD_CURRENT) < 50)
     {
      Log("Esperando mas barras de historico...");
      return INIT_SUCCEEDED;
     }

   g_lastBarTime = iTime(_Symbol, PERIOD_CURRENT, 0);
   g_confidence_threshold = InpConfidenceThreshold; // Inicializar umbral de confianza desde entrada del usuario

// Prepare the magic number
   int _some_rnd_int = MathRand();
   g_magic_number = (StringToInteger(GetYYYYMMDDHHMMSS(TimeCurrent())) + _some_rnd_int) * -1;
   g_trade.SetExpertMagicNumber(g_magic_number);

   SendInitialNotification();
   SaveCurrentExperAdvisorInputs(MQLInfoString(MQL_PROGRAM_NAME) + "_" + _Symbol + ".set");

   Log("=== EnsembleBuyEA Iniciado correctamente ===");
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

   if(g_handleRSI != INVALID_HANDLE)
      IndicatorRelease(g_handleRSI);
   if(g_handleATR != INVALID_HANDLE)
      IndicatorRelease(g_handleATR);

// Triggers specifically when inputs are changed via the GUI
   if(reason == REASON_PARAMETERS)
      SaveCurrentExperAdvisorInputs(MQLInfoString(MQL_PROGRAM_NAME) + "_" + _Symbol + ".set");

   Log("=== EnsembleBuyEA Finalizado. Razon: " + IntegerToString(reason) + " ===");
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
         Log("Esperando barras... Actual: " + IntegerToString(totalBars) + "/50");
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
            Log("WARN: No se pudieron preparar features para modelo " + g_models[i].id);
         probs[i] = 0.0;
         allOk = false;
         continue;
        }

      if(!RunInference(i, modelInput, probs[i]))
        {
         if(InpVerbose)
            Log("WARN: Inferencia fallida para modelo " + g_models[i].id);
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
                         "BAR[%d] | A:%.3f B:%.3f C:%.3f D:%.3f E:%.3f | ENSEMBLE BUY:%.3f/%.3f | SELL:%.3f | DIFF:%.3f/%.3f",
                         g_barsProcessed, probs[0], probs[1], probs[2], probs[3], probs[4],
                         ensembleProb, g_confidence_threshold, sellProb,
                         ensembleProb - sellProb, InpMinConfidenceDiff
                      );
      Log(logMsg);
     }

// Gestion de posiciones existentes
   if(InpUseTrailingStop)
      ManageTrailingStops();

// Decision de trading
   int openPositions = CountOpenPositions(POSITION_TYPE_BUY);
   double confDiff = ensembleProb - sellProb;

   string blockReason = "";
   if(ensembleProb < g_confidence_threshold)
      blockReason = StringFormat("BUY prob %.3f < umbral %.3f", ensembleProb, g_confidence_threshold);
   else
      if(confDiff < InpMinConfidenceDiff)
         blockReason = StringFormat("diff BUY-SELL %.3f < InpMinConfidenceDiff %.3f", confDiff, InpMinConfidenceDiff);
      else
         if(openPositions >= InpMaxPositions)
            blockReason = StringFormat("posiciones abiertas %d >= InpMaxPositions %d", openPositions, InpMaxPositions);
         else
            if(!allOk)
               blockReason = "inferencia incompleta en al menos un modelo (allOk=false)";

   if(blockReason == "")
     {
      OpenBuyPosition(ensembleProb, probs);
     }
   else
      if(InpVerbose)
         Log("NO BUY | " + blockReason);
  }

//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void Log(string message)
  {
   PrintFormat("%s | %s", _Symbol, message);
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
   if(sepPos == -1)
      sepPos = StringFind(fullPath, "/", 0);
   if(sepPos != -1)
      filename = StringSubstr(fullPath, sepPos + 1);

   long handle = INVALID_HANDLE;
   if(InpBacktest)
     {
      if(g_models[idx].id == "A")
         handle = OnnxCreateFromBuffer(res_model_a, ONNX_DEFAULT);
      else
         if(g_models[idx].id == "B")
            handle = OnnxCreateFromBuffer(res_model_b, ONNX_DEFAULT);
         else
            if(g_models[idx].id == "C")
               handle = OnnxCreateFromBuffer(res_model_c, ONNX_DEFAULT);
            else
               if(g_models[idx].id == "D")
                  handle = OnnxCreateFromBuffer(res_model_d, ONNX_DEFAULT);
               else
                  if(g_models[idx].id == "E")
                     handle = OnnxCreateFromBuffer(res_model_e, ONNX_DEFAULT);
                  else
                     Log("ERROR: Modelo ID desconocido: " + g_models[idx].id);
     }
   else
      handle = OnnxCreate(filename, ONNX_DEFAULT);

   if(handle == INVALID_HANDLE)
     {
      int err = GetLastError();
      Log("ERROR cargando modelo " + g_models[idx].id + " (" + filename + "): " + IntegerToString(err));

      handle = OnnxCreate(fullPath, ONNX_DEFAULT);
      if(handle == INVALID_HANDLE)
        {
         Log("ERROR: Tampoco funciona con ruta completa: " + fullPath);
         return false;
        }
     }

   g_models[idx].onnxHandle = handle;

// Verificar inputs/outputs del modelo
   long inputCount = OnnxGetInputCount(handle);
   long outputCount = OnnxGetOutputCount(handle);
   g_models[idx].outputCount = (int)outputCount;

   Log("Modelo " + g_models[idx].id + " ONNX info: inputs=" + IntegerToString(inputCount) + ", outputs=" + IntegerToString(outputCount));

// Mostrar nombres de inputs/outputs para debug
   for(int i = 0; i < (int)inputCount; i++)
     {
      string inName = OnnxGetInputName(handle, i);
      Log("  Input[" + IntegerToString(i) + "]: " + inName);
     }
   for(int i = 0; i < (int)outputCount; i++)
     {
      string outName = OnnxGetOutputName(handle, i);
      Log("  Output[" + IntegerToString(i) + "]: " + outName);
     }

// Definir shape del input [1, inputSize]
   long inputShape[] = {1, g_models[idx].inputSize};
   if(!OnnxSetInputShape(handle, 0, inputShape))
     {
      Log("ERROR OnnxSetInputShape modelo " + g_models[idx].id + ": " + IntegerToString(GetLastError()));
      OnnxRelease(handle);
      return false;
     }

// Definir shapes de outputs
// Output 0: label (int64) -> shape [1]
   long outputShape0[] = {1};
   if(!OnnxSetOutputShape(handle, 0, outputShape0))
     {
      Log("ERROR OnnxSetOutputShape[0] modelo " + g_models[idx].id + ": " + IntegerToString(GetLastError()));
      OnnxRelease(handle);
      return false;
     }

// Output 1: probabilities (float32) -> shape [1, 2]
   if(outputCount >= 2)
     {
      long outputShape1[] = {1, 2};
      if(!OnnxSetOutputShape(handle, 1, outputShape1))
        {
         Log("ERROR OnnxSetOutputShape[1] modelo " + g_models[idx].id + ": " + IntegerToString(GetLastError()));
         OnnxRelease(handle);
         return false;
        }
     }

   g_models[idx].loaded = true;
   Log("Modelo " + g_models[idx].id + " cargado OK. Input size: " + IntegerToString(g_models[idx].inputSize) +
       ", Outputs: " + IntegerToString(outputCount));
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

   if(CopyOpen(_Symbol, PERIOD_CURRENT, 0, requiredBars, opens) < requiredBars)
      return false;
   if(CopyHigh(_Symbol, PERIOD_CURRENT, 0, requiredBars, highs) < requiredBars)
      return false;
   if(CopyLow(_Symbol, PERIOD_CURRENT, 0, requiredBars, lows) < requiredBars)
      return false;
   if(CopyClose(_Symbol, PERIOD_CURRENT, 0, requiredBars, closes) < requiredBars)
      return false;

// Obtener RSI
   double rsiValues[];
   ArraySetAsSeries(rsiValues, true);
   if(CopyBuffer(g_handleRSI, 0, 0, requiredBars, rsiValues) < requiredBars)
      return false;

// Obtener ATR
   double atrValues[];
   ArraySetAsSeries(atrValues, true);
   if(CopyBuffer(g_handleATR, 0, 0, requiredBars, atrValues) < requiredBars)
      return false;

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
      else
         if(m.id == "D")
           {
            double bodyRatio = (range > 0.0000001) ? (body / range) : 0.0;
            double atr = atrValues[barIdx];
            double rangeNorm = (atr > 0.0000001) ? (range / atr) : 1.0;

            outVec[featIdx++] = (float)bodyRatio;
            outVec[featIdx++] = (float)rangeNorm;
            outVec[featIdx++] = (float)rsi;
           }
         else
            if(m.id == "E")
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
      Log("ERROR ONNX Run modelo " + m.id + ": " + IntegerToString(err));
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
         if(n == 0)
            return 0.0;
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
               if(probs[i] > 0.5)
                  buyVotes++;
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
         if(n <= 2)
            return (n > 0) ? validProbs[0] : 0.0;
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
      Log("Trading no permitido en este terminal");
      return;
     }

   if(!MQLInfoInteger(MQL_TRADE_ALLOWED))
     {
      Log("Trading no permitido para este EA");
      return;
     }

   double ask = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
   double bid = SymbolInfoDouble(_Symbol, SYMBOL_BID);
   double point = SymbolInfoDouble(_Symbol, SYMBOL_POINT);
   int digits = (int)SymbolInfoInteger(_Symbol, SYMBOL_DIGITS);

   if(ask == 0 || bid == 0)
     {
      Log("ERROR: Precios no disponibles");
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
      Log("ERROR abriendo BUY: " + IntegerToString(GetLastError()));
     }
   else
     {
      Log("BUY ABIERTO | Ask: " + DoubleToString(ask, digits) + " | SL: " + DoubleToString(sl, digits) + " | TP: " + DoubleToString(tp, digits) + " | " + comment);
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
      if(ticket <= 0)
         continue;

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
      if(ticket <= 0)
         continue;

      if(PositionGetString(POSITION_SYMBOL) != _Symbol)
         continue;
      if(PositionGetInteger(POSITION_MAGIC) != g_trade.RequestMagic())
         continue;
      if(PositionGetInteger(POSITION_TYPE) != POSITION_TYPE_BUY)
         continue;

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

//+------------------------------------------------------------------+
//|
//+------------------------------------------------------------------+
string GetYYYYMMDDHHMMSS(datetime dt)
  {
// 1. Generate YYYYDDMMHHmmss timestamp prefix
   MqlDateTime s_dt;
   TimeToStruct(dt, s_dt);

// Formats: YYYY (year), DD (day), MM (month), HH (hour), mm (minute), ss (second)
   string timestamp = StringFormat("%04d%02d%02d%02d%02d%02d_",
                                   s_dt.year,
                                   s_dt.day,
                                   s_dt.mon,
                                   s_dt.hour,
                                   s_dt.min,
                                   s_dt.sec);
   return timestamp;
  }

//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
string GetLastClosedTradeInfo()
  {
// Select the entire account history
   if(!HistorySelect(0, TimeCurrent()))
     {
      return "Error: Could not retrieve account history.";
     }

   int total_deals = HistoryDealsTotal();

// Loop backwards to find the most recent closed deal
   for(int i = total_deals - 1; i >= 0; i--)
     {
      ulong ticket = HistoryDealGetTicket(i);
      if(ticket > 0)
        {
         long entry_type = HistoryDealGetInteger(ticket, DEAL_ENTRY);

         // Check if this deal was an exit (closing a trade)
         if(entry_type == DEAL_ENTRY_OUT || entry_type == DEAL_ENTRY_INOUT)
           {
            // Extract trade information
            string symbol = HistoryDealGetString(ticket, DEAL_SYMBOL);
            double profit = HistoryDealGetDouble(ticket, DEAL_PROFIT);
            double volume = HistoryDealGetDouble(ticket, DEAL_VOLUME);
            long   deal_type = HistoryDealGetInteger(ticket, DEAL_TYPE);

            // The type of the closing deal is the opposite of the position.
            // A Sell deal closes a Buy position, and vice versa.
            string position_type = (deal_type == DEAL_TYPE_SELL) ? "BUY" : "SELL";

            // Format the final string (e.g., "BUY EURUSD | Vol: 1.00 | Profit: 50.25")
            string info = StringFormat("%s %s | Vol: %.2f | Profit: %.2f",
                                       position_type, symbol, volume, profit);
            return info;
           }
        }
     }

   return "No closed trades found.";
  }

//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void SendTelegramNotification(string bot_token, string chat_id, string msg)
  {
   if(!InpEnableNotifications)
      return;

// Construct the Telegram API URL
   string url = "https://api.telegram.org/bot" + bot_token + "/sendMessage";

// Setup headers and parameters
   string headers = "Content-Type: application/x-www-form-urlencoded\r\n";
   string params = "chat_id=" + chat_id + "&text=" + msg;

   char post_data[];
   char result_data[];
   string result_headers;

// Convert the string parameters to a char array for the WebRequest
   StringToCharArray(params, post_data, 0, WHOLE_ARRAY, CP_UTF8);

// StringToCharArray adds a null terminator (\0) at the end.
// We must remove it for the HTTP POST request to be valid.
   ArrayResize(post_data, ArraySize(post_data) - 1);

   int timeout = 5000; // 5 second timeout

// Reset the error code before making the request
   ResetLastError();

// Send the request
   int res = WebRequest("POST", url, headers, timeout, post_data, result_data, result_headers);

// Error Handling
   if(res == -1)
     {
      Log("Telegram WebRequest failed. Error code: " + GetLastError());
      Log("IMPORTANT: Have you added 'https://api.telegram.org' to the allowed URLs in Tools -> Options -> Expert Advisors?");
     }
   else
      if(res != 200)
        {
         Log("Telegram API returned an error. HTTP Code: " + res);
         Log("Response: " + CharArrayToString(result_data));
        }
      else
        {
         Log("Telegram notification sent successfully!");
        }
  }

//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool SaveCurrentExperAdvisorInputs(string file_name = "EA_Settings.set", bool common_folder = false)
  {
// 1. Generate YYYYDDMMHHmmss timestamp prefix
   string timestamp = GetYYYYMMDDHHMMSS(TimeCurrent());

// The filename
   string final_file_name = StringFormat("%s%s", timestamp, file_name);

// 2. Set file flags (Text mode, ANSI encoding, Write access)
   int flags = FILE_WRITE | FILE_TXT | FILE_ANSI;
   if(common_folder)
      flags |= FILE_COMMON;

// 3. Open the file
   int file_handle = FileOpen(final_file_name, flags);
   if(file_handle == INVALID_HANDLE)
     {
      Log(StringFormat("Error: Failed to open file '%s' for writing. Code: %d", final_file_name, GetLastError()));
      return false;
     }

// 4. Write Header
   FileWriteString(file_handle, "; Expert Advisor Saved Inputs\r\n");
   FileWriteString(file_handle, StringFormat("; Saved on: %s\r\n\r\n", TimeToString(TimeCurrent(), TIME_DATE | TIME_MINUTES)));

// 5. Write Input Variables
   FileWriteString(file_handle, StringFormat("InpModelA_Path=%s\r\n", InpModelA_Path));
   FileWriteString(file_handle, StringFormat("InpModelB_Path=%s\r\n", InpModelB_Path));
   FileWriteString(file_handle, StringFormat("InpModelC_Path=%s\r\n", InpModelC_Path));
   FileWriteString(file_handle, StringFormat("InpModelD_Path=%s\r\n", InpModelD_Path));
   FileWriteString(file_handle, StringFormat("InpModelE_Path=%s\r\n", InpModelE_Path));
   FileWriteString(file_handle, StringFormat("InpEnsembleMode=%s\r\n", InpEnsembleMode));
   FileWriteString(file_handle, StringFormat("InpWeightA=%f\r\n", InpWeightA));
   FileWriteString(file_handle, StringFormat("InpWeightB=%f\r\n", InpWeightB));
   FileWriteString(file_handle, StringFormat("InpWeightC=%f\r\n", InpWeightC));
   FileWriteString(file_handle, StringFormat("InpWeightD=%f\r\n", InpWeightD));
   FileWriteString(file_handle, StringFormat("InpWeightE=%f\r\n", InpWeightE));
   FileWriteString(file_handle, StringFormat("InpConfidenceThreshold=%f\r\n", InpConfidenceThreshold));
   FileWriteString(file_handle, StringFormat("InpMinConfidenceDiff=%f\r\n", InpMinConfidenceDiff));
   FileWriteString(file_handle, StringFormat("InpLotSize=%f\r\n", InpLotSize));
   FileWriteString(file_handle, StringFormat("InpSL_Pips=%f\r\n", InpSL_Pips));
   FileWriteString(file_handle, StringFormat("InpTP_Pips=%f\r\n", InpTP_Pips));
   FileWriteString(file_handle, StringFormat("InpMaxPositions=%d\r\n", InpMaxPositions));
   FileWriteString(file_handle, StringFormat("InpUseTrailingStop=%s\r\n", InpUseTrailingStop ? "true" : "false"));
   FileWriteString(file_handle, StringFormat("InpTrailDistance=%f\r\n", InpTrailDistance));
   FileWriteString(file_handle, StringFormat("InpRSI_Period=%d\r\n", InpRSI_Period));
   FileWriteString(file_handle, StringFormat("InpATR_Period=%d\r\n", InpATR_Period));
   FileWriteString(file_handle, StringFormat("InpVerbose=%s\r\n", InpVerbose ? "true" : "false"));
   FileWriteString(file_handle, StringFormat("InpLogEveryNBars=%d\r\n", InpLogEveryNBars));
   FileWriteString(file_handle, StringFormat("InpTelegramBotToken=%s\r\n", InpTelegramBotToken));
   FileWriteString(file_handle, StringFormat("InpTelegramChatID=%s\r\n", InpTelegramChatID));

// 6. Flush and close handle
   FileClose(file_handle);

   Log(StringFormat("Success: Current EA inputs saved to '%s'", final_file_name));
   return true;
  }
//+------------------------------------------------------------------+

//+------------------------------------------------------------------+
//| Sends initial EA startup message with all input parameters       |
//+------------------------------------------------------------------+
void SendInitialNotification()
  {
   string msg = "ENSEMBLE VOTER TRADING (BUY)\n\n";

   msg += StringFormat("Symbol: %s\n", _Symbol);
   msg += StringFormat("Magic: %d\n", g_magic_number);
   msg += StringFormat("Ensemble Mode: %s\n", EnumToString(InpEnsembleMode));
   msg += StringFormat("Confidence Threshold: %.2f\n", InpConfidenceThreshold);
   msg += StringFormat("Min Confidence Diff: %.2f\n", InpMinConfidenceDiff);
   msg += StringFormat("Lot Size: %.2f\n", InpLotSize);
   msg += StringFormat("Stop Loss (pips): %.2f\n", InpSL_Pips);
   msg += StringFormat("Take Profit (pips): %.2f\n", InpTP_Pips);
   msg += StringFormat("Max Positions: %d\n", InpMaxPositions);
   msg += StringFormat("Use Trailing Stop: %s\n", InpUseTrailingStop ? "true" : "false");
   msg += StringFormat("Trail Distance: %.2f\n", InpTrailDistance);
   msg += StringFormat("RSI Period: %d\n", InpRSI_Period);
   msg += StringFormat("ATR Period: %d\n", InpATR_Period);

   SendTelegramNotification(InpTelegramBotToken, InpTelegramChatID, msg);
  }

//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void OnTradeTransaction(const MqlTradeTransaction& trans,
                        const MqlTradeRequest& request,
                        const MqlTradeResult& result)
  {
// Only trigger when a new deal is added to the account history
   if(trans.type != TRADE_TRANSACTION_DEAL_ADD)
      return;

// Get transaction symbol from the trade transaction as all charts trigger this event
   string symbol = trans.symbol;

   ulong deal_ticket = trans.deal;

// Select the deal to read its properties
   if(!HistoryDealSelect(deal_ticket))
      return;

   long entry_type = HistoryDealGetInteger(deal_ticket, DEAL_ENTRY);
   long deal_reason = HistoryDealGetInteger(deal_ticket, DEAL_REASON);

   if(entry_type == DEAL_ENTRY_IN)
      HandleDealOpen(symbol);
   else
      if(entry_type == DEAL_ENTRY_OUT || entry_type == DEAL_ENTRY_INOUT)
        {
         if(deal_reason == DEAL_REASON_TP)
            HandleDealClosedByTP();
         else
            if(deal_reason == DEAL_REASON_SL)
               HandleDealClosedBySL(deal_ticket);
            else
               if(deal_reason == DEAL_REASON_CLIENT)
                  HandleDealClosedManually();
        }
  }

//+------------------------------------------------------------------+
//| Handles a trade open (DEAL_ENTRY_IN)                             |
//+------------------------------------------------------------------+
void HandleDealOpen(string symbol)
  {
// Plays a default MT5 sound. Replace with your custom .wav file name if needed.
   PlaySound("ok.wav");
// Send Telegram notification for trade open
   string message = "Trade Opened: " + symbol;
   SendTelegramNotification(InpTelegramBotToken, InpTelegramChatID, message);
  }

//+------------------------------------------------------------------+
//| Handles a trade closed by take profit (DEAL_REASON_TP)           |
//+------------------------------------------------------------------+
void HandleDealClosedByTP()
  {
   PlaySound("alert.wav");
// Send Telegram notification for trade close by TP
   string trade_info = GetLastClosedTradeInfo();
   string message = "Trade Closed by TP: " + trade_info;
   SendTelegramNotification(InpTelegramBotToken, InpTelegramChatID, message);
  }

//+------------------------------------------------------------------+
//| Handles a trade closed by stop loss (DEAL_REASON_SL)             |
//+------------------------------------------------------------------+
void HandleDealClosedBySL(ulong deal_ticket)
  {
   PlaySound("timeout.wav");
// Send Telegram notification for trade close by SL
   string trade_info = GetLastClosedTradeInfo();
   string message = "Trade Closed by SL: " + trade_info;
// Increase the confidence threshold slightly after a stop loss to avoid similar trades
   g_confidence_threshold += InpConfidenceStep;
// Finally, send the Telegram notification
   SendTelegramNotification(InpTelegramBotToken, InpTelegramChatID, message);
  }

//+------------------------------------------------------------------+
//| Handles a trade closed manually (DEAL_REASON_CLIENT)             |
//+------------------------------------------------------------------+
void HandleDealClosedManually()
  {
   PlaySound("close.wav");
// Send Telegram notification for trade close by manual close
   string trade_info = GetLastClosedTradeInfo();
   string message = "Trade Closed Manually: " + trade_info;
   SendTelegramNotification(InpTelegramBotToken, InpTelegramChatID, message);
  }
//+------------------------------------------------------------------+
