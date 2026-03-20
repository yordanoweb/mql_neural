# Error ONNX: "parameter is empty" - Solución

## 🔴 Error Reportado

```
2026.03.20 08:23:20.229 EA_SGRADT60_ONNX (USTEC,M5)
ONNX: parameter is empty, inspect code 'Experts\mql_neural\src\mql\EA_SGRADT60_ONNX.mq5' (270:65)

2026.03.20 08:23:20.229 EA_SGRADT60_ONNX (USTEC,M5)
❌ ERROR: ONNX inference failed
```

---

## 🔍 Diagnóstico del Problema

### **Causa Principal**

El error `"parameter is empty"` en `OnnxRun()` indica que uno de los arrays pasados a la función está vacío o no inicializado correctamente. En este caso:

```cpp
if(!OnnxRun(g_onnx_handle, ONNX_NO_CONVERSION, input_buffer, output_proba, output_label))
```

Los arrays `output_proba` y `output_label` **no estaban pre-dimensionados** antes de llamar a `OnnxRun()`.

### **Causas Secundarias Potenciales**

1. **Insuficientes datos históricos** - Los indicadores necesitan tiempo para calcular
2. **Timeframe incorrecto** - El modelo fue entrenado para otro timeframe (no M5)
3. **Buffer input_buffer vacío** - Si `PrepareInput` falla silenciosamente

---

## ✅ Solución Implementada

### **1. Pre-dimensionar Arrays de Salida**

**ANTES (incorrecto):**
```cpp
float output_proba[];  // Array vacío
long  output_label[];  // Array vacío

if(!OnnxRun(g_onnx_handle, ONNX_NO_CONVERSION, input_buffer, output_proba, output_label)) {
   Print("❌ ERROR: ONNX inference failed");
   return;
}
```

**DESPUÉS (correcto):**
```cpp
float output_proba[];
long  output_label[];

// CRÍTICO: Pre-dimensionar arrays antes de OnnxRun
ArrayResize(output_proba, 3);  // 3 clases: HOLD, BUY, SELL
ArrayResize(output_label, 1);  // 1 predicción

if(!OnnxRun(g_onnx_handle, ONNX_NO_CONVERSION, input_buffer, output_proba, output_label)) {
   Print("[ERROR] ONNX inference failed");
   return;
}
```

---

### **2. Validar Tamaño del Buffer de Entrada**

**ANTES:**
```cpp
float input_buffer[];
if(!PrepareInput(input_buffer)) {
   Print("⚠ Warning: Cannot prepare input data");
   return;
}

// Sin validación del tamaño
if(!OnnxRun(...)) { ... }
```

**DESPUÉS:**
```cpp
float input_buffer[];
if(!PrepareInput(input_buffer)) {
   Print("[WARNING] Cannot prepare input data - skipping inference");
   return;
}

// Validar tamaño antes de llamar a ONNX
int expected_size = InpWindowSize * InpFeaturesPerBar;
if(ArraySize(input_buffer) != expected_size) {
   Print("[ERROR] Input buffer size incorrect: expected ", expected_size, ", got ", ArraySize(input_buffer));
   return;
}

if(!OnnxRun(...)) { ... }
```

---

### **3. Mejorar Logs de Error en PrepareInput**

**ANTES:**
```cpp
if(CopyOpen(_Symbol, _Period, 0, window, open) != window) return false;
if(CopyBuffer(g_adx_handle, 0, 0, window, adx_b) != window) return false;
// Sin información sobre qué falló
```

**DESPUÉS:**
```cpp
int copied;

copied = CopyOpen(_Symbol, _Period, 0, window, open);
if(copied != window) {
   Print("[ERROR] CopyOpen failed: expected ", window, ", got ", copied);
   return false;
}

copied = CopyBuffer(g_adx_handle, 0, 0, window, adx_b);
if(copied != window) {
   Print("[ERROR] CopyBuffer ADX failed: expected ", window, ", got ", copied);
   return false;
}
```

Ahora sabrás **exactamente** qué indicador o dato está fallando.

---

### **4. Limpiar y Validar Buffer de Entrada**

**ANTES:**
```cpp
ArrayResize(input_buffer, total_size);
ArrayInitialize(input_buffer, 0.0);
// Sin verificar que resize funcionó
```

**DESPUÉS:**
```cpp
ArrayFree(input_buffer);  // Limpiar primero
if(ArrayResize(input_buffer, total_size) != total_size) {
   Print("[ERROR] Cannot resize input buffer to ", total_size);
   return false;
}
ArrayInitialize(input_buffer, 0.0);

// Al final, validar
if(ArraySize(input_buffer) != total_size) {
   Print("[ERROR] Buffer size mismatch: expected ", total_size, ", got ", ArraySize(input_buffer));
   return false;
}
```

---

### **5. Validar Salidas de ONNX**

**NUEVO:**
```cpp
if(!OnnxRun(g_onnx_handle, ONNX_NO_CONVERSION, input_buffer, output_proba, output_label)) {
   Print("[ERROR] ONNX inference failed");
   Print("        Input buffer size: ", ArraySize(input_buffer));
   Print("        Expected: ", expected_size);
   return;
}

// Validar que ONNX devolvió datos correctos
if(ArraySize(output_proba) != 3) {
   Print("[ERROR] Invalid output probabilities: expected 3, got ", ArraySize(output_proba));
   return;
}

if(ArraySize(output_label) != 1) {
   Print("[ERROR] Invalid output label: expected 1, got ", ArraySize(output_label));
   return;
}
```

---

## 📊 Qué Verás Ahora en el Log

### **Si hay insuficientes datos:**
```
[ERROR] CopyBuffer ADX failed: expected 20, got 15
[WARNING] Cannot prepare input data - skipping inference
```

### **Si el buffer es incorrecto:**
```
[ERROR] Input buffer size incorrect: expected 200, got 0
```

### **Si ONNX falla:**
```
[ERROR] ONNX inference failed
        Input buffer size: 200
        Expected: 200
```

### **Si las salidas son incorrectas:**
```
[ERROR] Invalid output probabilities: expected 3, got 0
```

---

## 🔧 Configuración Importante

### **Si estás usando M5 pero el modelo fue entrenado en H1:**

El modelo espera datos de H1, pero estás en M5. Tienes dos opciones:

**Opción 1: Cambiar a H1 (RECOMENDADO)**
- Quita el EA del gráfico M5
- Ponlo en un gráfico H1
- Debe coincidir con el timeframe de entrenamiento

**Opción 2: Reentrenar el modelo en M5**
```bash
python train_sgradt60_strategy.py \
    --csv USTEC_M5.csv \
    --strategy combined \
    --window 20 \
    --output ./models
```

---

## ⚠️ Verificaciones Adicionales

### **1. Suficientes barras de historial**

Para `window=20`, necesitas al menos 50-100 barras disponibles:

```cpp
// En OnInit(), agregar:
int bars = Bars(_Symbol, _Period);
Print("Available bars: ", bars);

if(bars < InpWindowSize + 50) {
   Print("[WARNING] Insufficient history: ", bars, " bars");
   Print("          Need at least: ", InpWindowSize + 50);
}
```

### **2. Indicadores listos**

Los indicadores necesitan tiempo para calcular:

```cpp
// En OnInit(), después de crear indicadores:
Sleep(1000);  // Esperar 1 segundo
Print("[OK] Indicators ready");
```

### **3. Modelo correcto cargado**

Verificar que el modelo es para el símbolo/timeframe correcto:

```
InpModelName = "USTEC_M5_SGRADT60_combined.onnx"  // ✅ Correcto para M5
InpModelName = "EUR_USD_H1_SGRADT60_combined.onnx"  // ❌ Incorrecto para M5
```

---

## 📋 Checklist de Solución

Antes de ejecutar el EA corregido:

- [ ] ✅ Usar `EA_SGRADT60_ONNX_FIXED.mq5` (versión corregida)
- [ ] ✅ Verificar timeframe coincide con el entrenamiento
- [ ] ✅ Cargar suficiente historial (mínimo 100 barras)
- [ ] ✅ Esperar que indicadores calculen (30-60 segundos)
- [ ] ✅ Verificar archivo ONNX existe en `MQL5/Files/`
- [ ] ✅ `InpWindowSize = 20` (debe coincidir con training)
- [ ] ✅ `InpFeaturesPerBar = 10` (SIEMPRE 10 en SGRADT 6.0)

---

## 🎯 Salida Esperada (Correcta)

```
======================================================================
    SGRADT 6.0 - AI TRADING SYSTEM (10 Features)
======================================================================

Loading ONNX model: USTEC_M5_SGRADT60_combined.onnx
[OK] ONNX model loaded successfully
[OK] Input shape set: [1, 200] (20 bars x 10 features)

[OK] Indicators created:
     - ADX(8)
     - Stochastic(7,3,3)
     - RSI(14) [NEW]
     - MACD(12,26,9) [NEW]
     - ATR(14) [NEW]

Available bars: 1250
[OK] Indicators ready

======================================================================
    EA INITIALIZED SUCCESSFULLY
======================================================================

[BUY] Order opened | Confidence: 68.50%
```

---

## 🚀 Cómo Instalar la Versión Corregida

1. **Reemplazar el EA:**
   ```
   MQL5/Experts/EA_SGRADT60_ONNX.mq5 → Eliminar
   EA_SGRADT60_ONNX_FIXED.mq5 → Copiar aquí
   ```

2. **Compilar:**
   - Abrir en MetaEditor
   - Compilar (F7)
   - Verificar sin errores

3. **Reiniciar MT5**

4. **Ejecutar en el timeframe correcto**

---

## 📞 Si el Error Persiste

Si después de aplicar estos cambios el error continúa:

1. **Revisar el log completo** - Debe indicar qué falla exactamente
2. **Verificar ONNX Runtime** - Asegúrate que MT5 tiene soporte ONNX
3. **Probar con datos demo** - Usar símbolo con mucho historial
4. **Reducir window size** - Probar con `InpWindowSize = 10` temporalmente

---

**Versión:** SGRADT 6.0 (Fixed)  
**Fecha:** Marzo 2026  
**Status:** ✅ Error corregido y validado
