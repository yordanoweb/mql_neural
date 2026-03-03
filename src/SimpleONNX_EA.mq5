//+------------------------------------------------------------------+
//|                                              SimpleONNX_EA.mq5   |
//+------------------------------------------------------------------+
#property strict

#include <Trade\Trade.mqh>

//--- RECURSOS: Asegúrate de que el nombre del archivo ONNX coincida
#resource "\\Files\\model_multi.onnx" as uchar ExtModel[]

//--- ENUMERACIONES
enum ENUM_LOGIC { LOGIC_NORMAL, LOGIC_MIRROR };

//--- INPUTS
input group "Configuración IA"
input ENUM_LOGIC InpLogic      = LOGIC_MIRROR; // Lógica de ejecución
input float      InpMinConf    = 0.6;         // Confianza mínima (0.5 a 1.0)

input group "Gestión de Riesgo"
input double     InpLot        = 0.1;          // Lote
input int        InpMagic      = 123456;       // Número Mágico
input int        InpATR        = 14;           // Período ATR para SL
input double     InpMultiplier = 2.0;          // Multiplicador SL (ATR * x)

//--- VARIABLES GLOBALES
long     onnx_handle = INVALID_HANDLE;
CTrade   m_trade;
const int WINDOW_SIZE = 10;
const int FEATURES    = 3; // Cuerpo, Rango, RSI

//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit()
  {
// 1. Crear modelo
   onnx_handle = OnnxCreateFromBuffer(ExtModel, ONNX_DEFAULT);
   if(onnx_handle == INVALID_HANDLE)
     {
      Print("Error: No se pudo crear el modelo ONNX. Código: ", GetLastError());
      return(INIT_FAILED);
     }

// 2. Configurar Entrada (30 elementos: 10 velas * 3 features)
   long input_shape[] = {1, 30};
   if(!OnnxSetInputShape(onnx_handle, 0, input_shape))
     {
      Print("Error al configurar input shape.");
      return(INIT_FAILED);
     }

// 3. Configurar Salidas (Label y Probabilidades)
   long out_shape_label[] = {1};
   OnnxSetOutputShape(onnx_handle, 0, out_shape_label);
   long out_shape_probs[] = {1, 2};
   OnnxSetOutputShape(onnx_handle, 1, out_shape_probs);

   m_trade.SetExpertMagicNumber(InpMagic);
   return(INIT_SUCCEEDED);
  }

//+------------------------------------------------------------------+
//| Expert deinitialization function                                 |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
  {
   if(onnx_handle != INVALID_HANDLE)
      OnnxRelease(onnx_handle);
  }

//+------------------------------------------------------------------+
//| Expert tick function                                             |
//+------------------------------------------------------------------+
void OnTick()
  {
// Ejecutar solo al cierre de cada vela H1
   static datetime last_bar = 0;
   datetime current_bar = iTime(_Symbol, _Period, 0);
   if(current_bar == last_bar)
      return;
   last_bar = current_bar;

//--- 1. OBTENER DATOS HISTÓRICOS ---
   double close[], open[], high[], low[];
   ArraySetAsSeries(close, true);
   ArraySetAsSeries(open, true);
   ArraySetAsSeries(high, true);
   ArraySetAsSeries(low, true);

   if(CopyClose(_Symbol, _Period, 0, WINDOW_SIZE + 15, close) < WINDOW_SIZE + 15 ||
      CopyOpen(_Symbol, _Period, 0, WINDOW_SIZE, open) < WINDOW_SIZE ||
      CopyHigh(_Symbol, _Period, 0, WINDOW_SIZE, high) < WINDOW_SIZE ||
      CopyLow(_Symbol, _Period, 0, WINDOW_SIZE, low) < WINDOW_SIZE)
      return;

//--- 2. CALCULAR INDICADORES (Misma lógica que Python) ---
   int rsi_handle = iRSI(_Symbol, _Period, 14, PRICE_CLOSE);
   double rsi_buffer[];
   ArraySetAsSeries(rsi_buffer, true);
   CopyBuffer(rsi_handle, 0, 0, WINDOW_SIZE, rsi_buffer);

   int atr_handle = iATR(_Symbol, _Period, InpATR);
   double atr_buffer[];
   ArraySetAsSeries(atr_buffer, true);
   CopyBuffer(atr_handle, 0, 0, 1, atr_buffer);
   double current_atr = atr_buffer[0];

//--- 3. LLENAR BUFFER DE ENTRADA (NORMALIZADO) ---
   float input_buffer[];
   ArrayResize(input_buffer, WINDOW_SIZE * FEATURES);

// Ajuste de pips (0.0001 para EURUSD)
   double pip_unit = _Point * (_Digits == 5 || _Digits == 3 ? 10 : 1);

   for(int i=0; i < WINDOW_SIZE; i++)
     {
      int mql_idx = WINDOW_SIZE - 1 - i; // De más antiguo a más reciente

      input_buffer[i * 3 + 0] = (float)((close[mql_idx] - open[mql_idx]) / pip_unit); // Cuerpo en Pips
      input_buffer[i * 3 + 1] = (float)((high[mql_idx] - low[mql_idx]) / pip_unit);   // Rango en Pips
      input_buffer[i * 3 + 2] = (float)(rsi_buffer[mql_idx] / 100.0);               // RSI (0-1)
     }

//--- 4. INFERENCIA ---
   long output_label[];
   float output_probs[];
   ArrayResize(output_label, 1);
   ArrayResize(output_probs, 2);

   if(!OnnxRun(onnx_handle, ONNX_NO_CONVERSION, input_buffer, output_label, output_probs))
     {
      Print("Error en OnnxRun");
      return;
     }

   long  prediction = output_label[0];
   float prob_baja  = output_probs[0];
   float prob_sube  = output_probs[1];
   float confianza  = (prediction == 1) ? prob_sube : prob_baja;

// Añadir dentro de OnTick, antes de buscar nuevas operaciones:
   for(int i=PositionsTotal()-1; i>=0; i--)
     {
      if(PositionGetSymbol(i) == _Symbol)
        {
         ulong ticket = PositionGetInteger(POSITION_TICKET);
         datetime time_open = (datetime)PositionGetInteger(POSITION_TIME);

         // Si han pasado más de 48 horas, cerramos para liberar el bot
         if(TimeCurrent() - time_open > 48 * 3600)
           {
            m_trade.PositionClose(ticket);
            Print("Operación cerrada por tiempo (48h)");
           }
        }
     }

//--- 5. EJECUCIÓN DE ÓRDENES (LÓGICA ESPEJO) ---
   if(!PositionSelect(_Symbol))
     {
      double sl_dist = current_atr * InpMultiplier;
      double tp_dist = sl_dist * 1.5;

      if(confianza >= InpMinConf)
        {
         // Lógica Espejo: Invertimos la predicción de la IA
         if((InpLogic == LOGIC_MIRROR && prediction == 1) || (InpLogic == LOGIC_NORMAL && prediction == 0))
           {
            // VENDER
            double price = SymbolInfoDouble(_Symbol, SYMBOL_BID);
            m_trade.Sell(InpLot, _Symbol, price, price + sl_dist, price - tp_dist, "AI Sell");
           }
         else
           {
            // COMPRAR
            double price = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
            m_trade.Buy(InpLot, _Symbol, price, price - sl_dist, price + tp_dist, "AI Buy");
           }
        }
     }

// Comentario en pantalla para debug
   Comment("IA Predicción: ", (prediction == 1 ? "SUBE" : "BAJA"),
           "\nConfianza: ", DoubleToString(confianza*100, 2), "%",
           "\nLógica: ", EnumToString(InpLogic));
  }
//+------------------------------------------------------------------+
