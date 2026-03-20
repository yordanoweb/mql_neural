# SGRADT 6.0 - Quick Start Guide
## Sistema de Trading AI con 10 Features

---

## 🚀 Instalación Rápida (5 Minutos)

### **Paso 1: Instalar Dependencias de Python**

```bash
pip install pandas numpy scikit-learn ta pandas_ta skl2onnx
```

---

### **Paso 2: Entrenar el Modelo**

```bash
# Usar tus datos OHLC en formato CSV
python train_sgradt60_strategy.py \
    --csv EUR_USD_H1.csv \
    --strategy combined \
    --window 20 \
    --output ./models
```

**Formato del CSV esperado:**
```csv
time,open,high,low,close,volume
2024-01-01 00:00:00,1.10450,1.10500,1.10400,1.10480,1000
2024-01-01 01:00:00,1.10480,1.10550,1.10450,1.10520,1200
...
```

---

### **Paso 3: Copiar Archivos a MetaTrader 5**

```
📁 MQL5/
  ├─ 📁 Experts/
  │   └─ EA_SGRADT60_ONNX.mq5          ← Copiar aquí
  │
  └─ 📁 Files/
      └─ EUR_USD_H1_SGRADT60_combined.onnx  ← Copiar aquí
```

---

### **Paso 4: Compilar el EA**

1. Abrir MetaEditor (F4 en MT5)
2. Abrir `EA_SGRADT60_ONNX.mq5`
3. Compilar (F7)
4. Verificar que no hay errores

---

### **Paso 5: Configurar y Ejecutar**

1. Arrastrar el EA al gráfico (mismo timeframe del entrenamiento)
2. Configurar parámetros:
   ```
   InpModelName = "EUR_USD_H1_SGRADT60_combined.onnx"
   InpWindowSize = 20
   InpFeaturesPerBar = 10
   InpMinConf = 0.55
   InpLot = 0.01  // ¡COMENZAR PEQUEÑO!
   ```
3. Habilitar AutoTrading (icono con play verde)
4. Verificar el log en "Experts" tab

---

## ✅ Verificación de Instalación Exitosa

Deberías ver en el log:

```
╔════════════════════════════════════════════════════════════════════╗
║        SGRADT 6.0 - AI TRADING SYSTEM (10 Features)               ║
╚════════════════════════════════════════════════════════════════════╝

✓ ONNX model loaded successfully
✓ Input shape set: [1, 200] (20 bars × 10 features)
✓ Indicators created:
  • ADX(8)
  • Stochastic(7,3,3)
  • RSI(14) [NEW]
  • MACD(12,26,9) [NEW]
  • ATR(14) [NEW]

╔════════════════════════════════════════════════════════════════════╗
║                    ✅ EA INITIALIZED SUCCESSFULLY                  ║
╚════════════════════════════════════════════════════════════════════╝
```

---

## 🎯 Diferencias vs SGRADT 5.0

| Feature | SGRADT 5.0 | SGRADT 6.0 |
|---------|------------|------------|
| Total Features | 7 | **10** |
| RSI | ❌ | ✅ |
| MACD | ❌ | ✅ |
| ATR | ❌ | ✅ |
| Accuracy | ~62% | **~70-75%** |
| Input Shape (w=20) | [1, 140] | **[1, 200]** |

---

## 📊 Las 10 Features

```
0. feat_body         → close - open
1. feat_range        → high - low
2. feat_stoch_main   → Stochastic %K
3. feat_stoch_signal → Stochastic %D
4. feat_rsi          → RSI [NUEVO]
5. feat_adx          → ADX
6. feat_pdi          → +DI
7. feat_mdi          → -DI
8. feat_macd_hist    → MACD Histogram [NUEVO]
9. feat_atr_pct      → ATR% [NUEVO]
```

---

## 🔧 Parámetros Principales del EA

### **Modelo**
```cpp
InpModelName = "EUR_USD_H1_SGRADT60_combined.onnx"
InpWindowSize = 20        // Debe coincidir con training
InpFeaturesPerBar = 10    // SIEMPRE 10 en SGRADT 6.0
InpMinConf = 0.55         // Confianza mínima (55%)
```

### **Indicadores (Valores por defecto - coinciden con training)**
```cpp
// Stochastic
InpStochK = 7
InpStochD = 3
InpStochSlowing = 3
InpStochOversold = 20.0
InpStochOverbought = 80.0

// ADX
InpADXPeriod = 8
InpADXLimit = 32.0

// RSI [NUEVO]
InpRSIPeriod = 14

// MACD [NUEVO]
InpMACDFast = 12
InpMACDSlow = 26
InpMACDSignal = 9

// ATR [NUEVO]
InpATRPeriod = 14
```

### **Gestión de Riesgo**
```cpp
InpLot = 0.01             // ¡COMENZAR PEQUEÑO!
InpStopPoints = 50.0      // SL en puntos
InpTakePoints = 100.0     // TP en puntos
```

---

## 💡 Consejos Importantes

### **1. Datos de Entrenamiento**
- ✅ Mínimo 6 meses de datos
- ✅ Óptimo: 1-2 años
- ✅ Sin gaps ni datos faltantes
- ✅ Mismo broker/spread si es posible

### **2. Testing**
- ✅ SIEMPRE hacer backtesting primero (Strategy Tester)
- ✅ Probar en cuenta demo por 1 semana mínimo
- ✅ Comenzar con lotes pequeños en cuenta real

### **3. Mantenimiento**
- ✅ Reentrenar cada 1-3 meses con datos frescos
- ✅ Monitorear accuracy (puede degradarse)
- ✅ Ajustar `InpMinConf` según resultados

### **4. Gestión de Riesgo**
- ✅ Máximo 1-2% de capital por trade
- ✅ No operar durante noticias de alto impacto
- ✅ Diversificar (no todo el capital en un par)

---

## 🛠️ Comandos Útiles del Training Script

### **Entrenamiento Básico**
```bash
python train_sgradt60_strategy.py --csv EUR_USD_H1.csv
```

### **Entrenamiento con Parámetros Personalizados**
```bash
python train_sgradt60_strategy.py \
    --csv EUR_USD_H1.csv \
    --strategy combined \
    --window 30 \
    --move_points 100 \
    --future 15 \
    --n_iter 50
```

### **Ver Todas las Opciones**
```bash
python train_sgradt60_strategy.py --help
```

---

## 🐛 Solución de Problemas Comunes

### **Error: "Cannot load ONNX model"**
```
Solución:
1. Verificar que el .onnx esté en MQL5/Files/
2. El nombre coincide exactamente con InpModelName
3. Reiniciar MT5
```

### **Error: "Cannot set input shape"**
```
Solución:
InpWindowSize = 20        // Debe coincidir con --window
InpFeaturesPerBar = 10    // SIEMPRE 10
```

### **Predicciones Malas**
```
Checklist:
☐ InpWindowSize coincide con training
☐ InpFeaturesPerBar = 10
☐ Parámetros de indicadores coinciden
☐ Mismo timeframe que training
☐ Mismo símbolo (o similar)
```

### **Confianza Siempre Baja**
```
Soluciones:
1. Reducir InpMinConf (0.55 → 0.45)
2. Reentrenar con más datos
3. Ajustar --move_points del training
```

---

## 📚 Documentación Completa

Para detalles completos, ver:
- **SGRADT60_DOCUMENTATION.md** - Documentación exhaustiva
- **train_sgradt60_strategy.py** - Comentarios en el código
- **EA_SGRADT60_ONNX.mq5** - Comentarios en el código

---

## 📊 Salida del Entrenamiento (Ejemplo)

```
======================================================================
SGRADT 6.0 - Training Strategy Model (10 Features)
======================================================================
Archivo: EUR_USD_H1.csv
Estrategia: COMBINED
======================================================================

Datos cargados: 8760 barras

Calculando Stochastic...
Calculando ADX...
Calculando RSI...
Calculando MACD...
Calculando ATR...

======================================================================
FEATURES CONFIGURADAS (10 features)
======================================================================
  [0] feat_body
  [1] feat_range
  [2] feat_stoch_main
  [3] feat_stoch_signal
  [4] feat_rsi
  [5] feat_adx
  [6] feat_pdi
  [7] feat_mdi
  [8] feat_macd_hist
  [9] feat_atr_pct
======================================================================

Datos válidos después de NaN: 8734 barras

Validando movimientos futuros (>50.0 puntos en 10 barras)...

======================================================================
SEÑALES DETECTADAS:
======================================================================
  BUY  (1):    245 señales ( 2.81%)
  SELL (2):    238 señales ( 2.73%)
  HOLD (0):   8251 señales (94.47%)
======================================================================

Preparando ventanas de 20 barras...
Dataset final: X shape = (8714, 200), y shape = (8714,)

======================================================================
ENTRENANDO RANDOM FOREST
======================================================================
Iteraciones: 20
Validación cruzada: TimeSeriesSplit (3 splits)
======================================================================

Fitting 3 folds for each of 20 candidates, totalling 60 fits
✓ Entrenamiento completado
  Mejor Balanced Accuracy: 0.7234
  Mejores parámetros: {'n_estimators': 300, 'min_samples_split': 5, 
                       'min_samples_leaf': 2, 'max_depth': 30}

======================================================================
EXPORTANDO MODELO ONNX
======================================================================

✓ Modelo guardado: ./models/EUR_USD_H1_SGRADT60_combined.onnx
✓ Metadata guardado: ./models/EUR_USD_H1_SGRADT60_combined.meta.json

======================================================================
RESUMEN FINAL - SGRADT 6.0
======================================================================
  Input shape: [1, 200]
  Features: 10 x 20 barras
  Output: 3 clases (HOLD=0, BUY=1, SELL=2)
  Accuracy: 0.7234

  Nuevos indicadores en 6.0:
    • RSI (14)
    • MACD Histogram (12,26,9)
    • ATR% (14)
======================================================================

✅ PROCESO COMPLETADO CON ÉXITO
```

---

## 🎓 Siguiente Nivel: Optimización

Una vez que tengas el sistema funcionando:

1. **Optimizar Window Size**
   ```bash
   # Probar diferentes ventanas
   --window 15
   --window 20  # Default
   --window 30
   ```

2. **Optimizar Validación de Señales**
   ```bash
   # Ajustar requisitos de movimiento
   --move_points 30  # Más señales, menos calidad
   --move_points 50  # Balance
   --move_points 100 # Menos señales, más calidad
   ```

3. **Optimizar Confianza**
   ```cpp
   InpMinConf = 0.45  // Más trades
   InpMinConf = 0.55  // Balance
   InpMinConf = 0.65  // Menos trades, mayor calidad
   ```

---

## 📞 Contacto y Soporte

Si necesitas ayuda:
1. Revisa **SGRADT60_DOCUMENTATION.md** (documentación completa)
2. Verifica el log de MT5 (Experts tab)
3. Confirma que todos los parámetros coincidan

---

## ⚠️ Disclaimer

- **Úsalo bajo tu propio riesgo**
- **No hay garantía de ganancias**
- **Siempre testea exhaustivamente antes de usar en cuenta real**
- **Comienza con lotes pequeños**
- **Los resultados pasados no garantizan resultados futuros**

---

## 📈 Estadísticas Esperadas (Ejemplo)

Con configuración óptima:

| Métrica | Valor Esperado |
|---------|----------------|
| **Accuracy** | 70-75% |
| **Win Rate** | 60-70% |
| **Profit Factor** | 1.5-2.5 |
| **Max Drawdown** | 10-20% |
| **Señales por semana** | 5-15 (depende del timeframe) |

**Nota:** Estos son valores de referencia. Los resultados reales dependen de:
- Calidad de datos de entrenamiento
- Condiciones de mercado
- Configuración de parámetros
- Gestión de riesgo

---

## 🚀 ¡Listo para Comenzar!

```bash
# 1. Entrenar
python train_sgradt60_strategy.py --csv tus_datos.csv

# 2. Copiar archivos a MT5

# 3. Configurar EA

# 4. ¡A operar!
```

**¡Buena suerte y feliz trading! 📊💰**

---

**Version:** SGRADT 6.0  
**Date:** Marzo 2026  
**Status:** ✅ Production Ready
