# Fix para Error ONNX "parameter is empty" - Cambio Mínimo

## ❌ El Problema Original

Tu EA funcionaba perfectamente en **M5** con un modelo entrenado en **M5**, pero fallaba con:

```
ONNX: parameter is empty, inspect code (270:65)
❌ ERROR: ONNX inference failed
```

---

## ✅ La Solución (SOLO 2 LÍNEAS)

En la función `RunInference()`, agregar **SOLO estas 2 líneas** antes de `OnnxRun()`:

### **ANTES (líneas 267-270 de tu código original):**

```cpp
//--- Run inference
float output_proba[];
long  output_label[];

if(!OnnxRun(g_onnx_handle, ONNX_NO_CONVERSION, input_buffer, output_proba, output_label)) {
   Print("❌ ERROR: ONNX inference failed");
   return;
}
```

### **DESPUÉS (con las 2 líneas agregadas):**

```cpp
//--- Run inference
float output_proba[];
long  output_label[];

// ⬇️ AGREGAR ESTAS 2 LÍNEAS ⬇️
ArrayResize(output_proba, 3);  // 3 classes: HOLD, BUY, SELL
ArrayResize(output_label, 1);  // 1 predicted class

if(!OnnxRun(g_onnx_handle, ONNX_NO_CONVERSION, input_buffer, output_proba, output_label)) {
   Print("❌ ERROR: ONNX inference failed");
   return;
}
```

---

## 🎯 Eso es TODO lo que necesitas cambiar

**NO necesitas:**
- ❌ Cambiar el timeframe (M5 está bien)
- ❌ Reentrenar el modelo
- ❌ Modificar parámetros
- ❌ Cambiar PrepareInput (opcional, solo para mejores logs)

**Solo necesitas:**
- ✅ Agregar esas 2 líneas de `ArrayResize`

---

## 📝 Si Quieres Copiar y Pegar

Busca en tu EA la función `RunInference()` y reemplaza esta sección:

```cpp
void RunInference()
{
   //--- Check if already traded this bar
   datetime current_bar = iTime(_Symbol, _Period, 0);
   if(InpOneTradePerBar && current_bar == g_last_trade_bar) {
      return;
   }
   
   //--- Prepare input
   float input_buffer[];
   if(!PrepareInput(input_buffer)) {
      Print("⚠ Warning: Cannot prepare input data");
      return;
   }
   
   //--- Run inference
   float output_proba[];
   long  output_label[];
   
   // ⬇️⬇️⬇️ AGREGAR ESTAS 2 LÍNEAS AQUÍ ⬇️⬇️⬇️
   ArrayResize(output_proba, 3);  // 3 classes: HOLD, BUY, SELL
   ArrayResize(output_label, 1);  // 1 predicted class
   // ⬆️⬆️⬆️ FIN DE LAS LÍNEAS NUEVAS ⬆️⬆️⬆️
   
   if(!OnnxRun(g_onnx_handle, ONNX_NO_CONVERSION, input_buffer, output_proba, output_label)) {
      Print("❌ ERROR: ONNX inference failed");
      return;
   }
   
   // ... resto del código sin cambios
```

---

## 🚀 Resumen

1. **El timeframe M5 está bien** - No hay problema con eso
2. **Solo faltan 2 líneas** - Pre-dimensionar los arrays de salida
3. **Todo lo demás funciona** - Tu EA original era correcto

El archivo `EA_SGRADT60_ONNX_FIXED.mq5` tiene este cambio más algunas mejoras opcionales en los logs, pero **el cambio crítico son solo esas 2 líneas**.

---

## 💡 Por Qué Esto Pasa

MQL5 requiere que arrays pasados como parámetros de salida estén pre-dimensionados. Es un requisito del lenguaje, no un bug en tu lógica.

---

**Usa tu EA original + estas 2 líneas = Problema resuelto** ✅
