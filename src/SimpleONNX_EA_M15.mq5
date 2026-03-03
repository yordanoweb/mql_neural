//+------------------------------------------------------------------+
//|                                              SimpleONNX_EA.mq5   |
//+------------------------------------------------------------------+
#property strict

#include <Trade\Trade.mqh>

//--- RECURSOS
#resource "\\Files\\model_m15.onnx" as uchar ExtModel[]

//--- ENUMERACIONES
enum ENUM_LOGIC { LOGIC_NORMAL, LOGIC_MIRROR };

//--- INPUTS
input group "Configuración IA"
input ENUM_LOGIC InpLogic      = LOGIC_MIRROR; 
input float      InpMinConf    = 0.62;         
input int        InpStartHour  = 9;            // Hora inicio (Broker)
input int        InpEndHour    = 18;           // Hora fin (Broker)

input group "Gestión de Riesgo"
input double     InpLot        = 1;          
input int        InpMagic      = 123456;       
input int        InpATR        = 6;           
input double     InpMultiplier = 1.5;          

//--- VARIABLES GLOBALES
long     onnx_handle = INVALID_HANDLE;
CTrade   m_trade;
const int WINDOW_SIZE = 20; // Para M15 usamos ventana de 20
const int FEATURES    = 3; 

int OnInit()
{
   onnx_handle = OnnxCreateFromBuffer(ExtModel, ONNX_DEFAULT);
   if(onnx_handle == INVALID_HANDLE) return(INIT_FAILED);

   long input_shape[] = {1, 60}; // 20 velas * 3 atributos
   if(!OnnxSetInputShape(onnx_handle, 0, input_shape)) return(INIT_FAILED);

   long out_shape_label[] = {1};
   OnnxSetOutputShape(onnx_handle, 0, out_shape_label);
   long out_shape_probs[] = {1, 2};
   OnnxSetOutputShape(onnx_handle, 1, out_shape_probs);

   m_trade.SetExpertMagicNumber(InpMagic);
   return(INIT_SUCCEEDED);
}

void OnDeinit(const int reason) { if(onnx_handle != INVALID_HANDLE) OnnxRelease(onnx_handle); }

void OnTick()
{
   // 1. FILTRO HORARIO CORRECTO
   MqlDateTime dt;
   TimeCurrent(dt); 
   bool horario_valido = (dt.hour >= InpStartHour && dt.hour < InpEndHour);

   // 2. CONTROL DE VELA
   static datetime last_bar = 0;
   datetime current_bar = iTime(_Symbol, _Period, 0);
   if(current_bar == last_bar) return;
   last_bar = current_bar;

   // 3. DATOS
   double close[], open[], high[], low[];
   ArraySetAsSeries(close, true); ArraySetAsSeries(open, true);
   ArraySetAsSeries(high, true);  ArraySetAsSeries(low, true);

   if(CopyClose(_Symbol, _Period, 0, WINDOW_SIZE + 15, close) < WINDOW_SIZE + 15 ||
      CopyOpen(_Symbol, _Period, 0, WINDOW_SIZE, open) < WINDOW_SIZE) return;

   // 4. INDICADORES
   int rsi_handle = iRSI(_Symbol, _Period, 14, PRICE_CLOSE);
   double rsi_buffer[];
   ArraySetAsSeries(rsi_buffer, true);
   CopyBuffer(rsi_handle, 0, 0, WINDOW_SIZE, rsi_buffer);

   int atr_handle = iATR(_Symbol, _Period, InpATR);
   double atr_buffer[];
   ArraySetAsSeries(atr_buffer, true);
   CopyBuffer(atr_handle, 0, 0, 1, atr_buffer);
   double current_atr = atr_buffer[0];

   // 5. INPUT BUFFER CON NORMALIZACIÓN POR _Digits
   float input_buffer[];
   ArrayResize(input_buffer, WINDOW_SIZE * FEATURES);
   
   // _Digits es la variable correcta. Si es 5 o 3 decimales, ajustamos a pips (x10).
   double pip_unit = _Point * (_Digits == 5 || _Digits == 3 ? 10 : 1);

   for(int i=0; i < WINDOW_SIZE; i++)
   {
      int mql_idx = WINDOW_SIZE - 1 - i;
      input_buffer[i * 3 + 0] = (float)((close[mql_idx] - open[mql_idx]) / pip_unit);
      input_buffer[i * 3 + 1] = (float)((iHigh(_Symbol, _Period, mql_idx) - iLow(_Symbol, _Period, mql_idx)) / pip_unit);
      input_buffer[i * 3 + 2] = (float)(rsi_buffer[mql_idx] / 100.0);
   }

   // 6. INFERENCIA
   long output_label[]; float output_probs[];
   ArrayResize(output_label, 1); ArrayResize(output_probs, 2);
   if(!OnnxRun(onnx_handle, ONNX_NO_CONVERSION, input_buffer, output_label, output_probs)) return;

   long  prediction = output_label[0];
   float confianza  = (prediction == 1) ? output_probs[1] : output_probs[0];

   // 7. EJECUCIÓN CON FILTRO HORARIO
   if(!PositionSelect(_Symbol) && horario_valido && confianza >= InpMinConf)
   {
      double sl_dist = current_atr * InpMultiplier;
      double tp_dist = sl_dist * 1.5;

      if((InpLogic == LOGIC_MIRROR && prediction == 1) || (InpLogic == LOGIC_NORMAL && prediction == 0))
      {
         double price = SymbolInfoDouble(_Symbol, SYMBOL_BID);
         m_trade.Sell(InpLot, _Symbol, price, price + sl_dist, price - tp_dist, "AI M15");
      }
      else
      {
         double price = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
         m_trade.Buy(InpLot, _Symbol, price, price - sl_dist, price + tp_dist, "AI M15");
      }
   }
   
   Comment("IA M15 | Confianza: ", DoubleToString(confianza*100, 2), "%",
           "\nHorario: ", (horario_valido ? "ACTIVO" : "RESTRINGIDO"));
}