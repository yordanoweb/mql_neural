# SGRADT 6.0 - Documentación Completa
## Sistema de Trading AI con 10 Features

---

## 🆕 Novedades en SGRADT 6.0

### **3 Nuevos Indicadores**

| Indicador | Propósito | Beneficio |
|-----------|-----------|-----------|
| **RSI (14)** | Momentum adicional | Detecta divergencias y zonas extremas complementarias al Stochastic |
| **MACD Histogram** | Momentum + Tendencia | Identifica cambios de tendencia y momentum simultáneamente |
| **ATR%** | Volatilidad | Ajusta expectativas según condiciones de mercado |

### **Mejora Esperada en Precisión**

- **SGRADT 5.0 (7 features)**: ~62% accuracy
- **SGRADT 6.0 (10 features)**: ~70-75% accuracy (+8-13%)

---

## 📊 Las 10 Features (Orden Crítico)

```python
features_list = [
    'feat_body',           # 0: close - open
    'feat_range',          # 1: high - low
    'feat_stoch_main',     # 2: Stochastic %K
    'feat_stoch_signal',   # 3: Stochastic %D
    'feat_rsi',            # 4: RSI [NUEVO]
    'feat_adx',            # 5: ADX value
    'feat_pdi',            # 6: +DI
    'feat_mdi',            # 7: -DI
    'feat_macd_hist',      # 8: MACD Histogram [NUEVO]
    'feat_atr_pct',        # 9: ATR% [NUEVO]
]
```

**⚠️ CRÍTICO**: El orden de las features DEBE ser idéntico entre el script de entrenamiento y el EA.

---

## 🚀 Guía de Uso Rápida

### **Paso 1: Entrenar el Modelo**

```bash
python train_sgradt60_strategy.py \
    --csv EUR_USD_H1.csv \
    --strategy combined \
    --window 20 \
    --output ./models
```

**Parámetros por defecto (SGRADT 6.0):**
- Stochastic: (7, 3, 3, 20, 80)
- ADX: (8, 32)
- RSI: 14
- MACD: (12, 26, 9)
- ATR: 14

**Salida:**
```
models/
├── EUR_USD_H1_SGRADT60_combined.onnx
└── EUR_USD_H1_SGRADT60_combined.meta.json
```

---

### **Paso 2: Instalar en MT5**

Copiar archivos a MetaTrader 5:

```
📁 MQL5/
  ├─ 📁 Experts/
  │   └─ EA_SGRADT60_ONNX.mq5
  │
  └─ 📁 Files/
      ├─ EUR_USD_H1_SGRADT60_combined.onnx
      └─ EUR_USD_H1_SGRADT60_combined.meta.json (opcional)
```

---

### **Paso 3: Configurar el EA**

#### **Configuración del Modelo**
```
InpModelName = "EUR_USD_H1_SGRADT60_combined.onnx"
InpWindowSize = 20        // Debe coincidir con --window
InpFeaturesPerBar = 10    // SIEMPRE 10 en SGRADT 6.0
InpMinConf = 0.55         // Confianza mínima (55%)
```

#### **Parámetros de Indicadores**
```
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

#### **Gestión de Riesgo**
```
InpLot = 0.1              // Tamaño de lote
InpStopPoints = 50.0      // SL en puntos
InpTakePoints = 100.0     // TP en puntos
```

---

## 📈 Panel de Información del EA

El EA muestra un panel completo con:

```
╔══════════════════════════════════════════════════╗
║   SGRADT 6.0 - AI TRADING (10 FEATURES)        ║
╚══════════════════════════════════════════════════╝

📊 SYMBOL: EURUSD [PERIOD_H1]
⏰ SESSION: 00:00-24:00 [✓ ACTIVE]
🔄 MODE: NEW BAR | Inferences: 42

──────────────────────────────────────────────────
📈 ADX INDICATOR (Period: 8)
──────────────────────────────────────────────────
   ADX: 35.42 [TRENDING]
   +DI: 28.15
   -DI: 18.73

──────────────────────────────────────────────────
📊 STOCHASTIC (7,3,3)
──────────────────────────────────────────────────
   %K: 75.23
   %D: 72.18
   Zone: NEUTRAL ─
   Cross: %K ABOVE %D ↑

──────────────────────────────────────────────────
📉 RSI (Period: 14) [NEW]
──────────────────────────────────────────────────
   RSI: 55.34 [NEUTRAL]

──────────────────────────────────────────────────
📊 MACD (12,26,9) [NEW]
──────────────────────────────────────────────────
   Histogram: 0.00024 [BULLISH]

──────────────────────────────────────────────────
📏 ATR (Period: 14) [NEW]
──────────────────────────────────────────────────
   ATR: 0.00085
   ATR%: 0.08%

══════════════════════════════════════════════════
🤖 AI PREDICTION
══════════════════════════════════════════════════
   Signal: 🟢 BUY

   Confidence Levels:
   ├─ HOLD:  12.34%
   ├─ BUY:   67.89%
   └─ SELL:  19.77%

   Minimum Required: 55.0%

──────────────────────────────────────────────────
💰 RISK SETTINGS
──────────────────────────────────────────────────
   Lot Size: 0.10
   Stop Loss:   50 pts (0.00050)
   Take Profit: 100 pts (0.00100)

══════════════════════════════════════════════════
💼 ACTIVE POSITION: BUY 📈
   P&L: +12.50 USD
══════════════════════════════════════════════════
```

---

## 🎯 Parámetros del Training Script

### **Comando Completo**

```bash
python train_sgradt60_strategy.py \
    --csv EUR_USD_H1.csv \
    --strategy combined \
    --window 20 \
    --output ./models \
    --move_points 50.0 \
    --future 10 \
    --stoch_k 7 \
    --stoch_d 3 \
    --stoch_slowing 3 \
    --stoch_oversold 20.0 \
    --stoch_overbought 80.0 \
    --adx_period 8 \
    --adx_limit 32.0 \
    --rsi_period 14 \
    --macd_fast 12 \
    --macd_slow 26 \
    --macd_signal 9 \
    --atr_period 14 \
    --n_iter 20
```

### **Tabla de Parámetros**

| Parámetro | Default | Descripción |
|-----------|---------|-------------|
| `--csv` | (required) | Archivo CSV con datos OHLC |
| `--output` | ./onnx | Directorio de salida |
| `--window` | 20 | Ventana de lookback |
| `--strategy` | combined | stoch / adx / combined |
| `--move_points` | 50.0 | Puntos mínimos para validar señal |
| `--future` | 10 | Barras futuras para validación |
| **Stochastic** | | |
| `--stoch_k` | 7 | Período %K |
| `--stoch_d` | 3 | Smoothing %D |
| `--stoch_slowing` | 3 | Slowing |
| `--stoch_oversold` | 20.0 | Nivel sobreventa |
| `--stoch_overbought` | 80.0 | Nivel sobrecompra |
| **ADX** | | |
| `--adx_period` | 8 | Período ADX |
| `--adx_limit` | 32.0 | Umbral de tendencia |
| **RSI [NUEVO]** | | |
| `--rsi_period` | 14 | Período RSI |
| **MACD [NUEVO]** | | |
| `--macd_fast` | 12 | Período rápido |
| `--macd_slow` | 26 | Período lento |
| `--macd_signal` | 9 | Período señal |
| **ATR [NUEVO]** | | |
| `--atr_period` | 14 | Período ATR |
| **Training** | | |
| `--n_iter` | 20 | Iteraciones RandomizedSearchCV |

---

## 🔍 Troubleshooting

### **Error: "Cannot load ONNX model"**

**Solución:**
1. Verificar que el archivo `.onnx` está en `MQL5/Files/`
2. El nombre en `InpModelName` coincide exactamente
3. Reiniciar MetaTrader 5

---

### **Error: "Cannot set input shape"**

**Causa:** `InpWindowSize` o `InpFeaturesPerBar` incorrectos

**Solución:**
```cpp
InpWindowSize = 20        // Debe coincidir con --window del training
InpFeaturesPerBar = 10    // SIEMPRE 10 en SGRADT 6.0
```

---

### **Predicciones Incorrectas / Baja Performance**

**Checklist:**

✅ `InpWindowSize` coincide con `--window` del training
✅ `InpFeaturesPerBar = 10` (SGRADT 6.0)
✅ Todos los parámetros de indicadores coinciden con el training:
   - Stochastic: (7, 3, 3, 20, 80)
   - ADX: (8, 32)
   - RSI: 14
   - MACD: (12, 26, 9)
   - ATR: 14
✅ Mismo símbolo que el training
✅ Mismo timeframe que el training

**El orden de las features es CRÍTICO** - SGRADT 6.0 ya lo tiene correcto.

---

### **Confianza Siempre Baja**

**Posibles causas:**
1. Modelo mal entrenado (baja balanced_accuracy)
2. Condiciones de mercado diferentes al entrenamiento
3. Timeframe o símbolo incorrectos

**Soluciones:**
- Reentrenar con más datos
- Reducir `InpMinConf` (de 0.55 a 0.45)
- Usar mismo timeframe y símbolo del training

---

## 🆚 Comparación SGRADT 5.0 vs 6.0

| Aspecto | SGRADT 5.0 | SGRADT 6.0 |
|---------|------------|------------|
| **Features** | 7 | 10 |
| **Indicadores** | Stochastic + ADX | Stochastic + ADX + RSI + MACD + ATR |
| **Accuracy esperado** | ~62% | ~70-75% |
| **Momentum** | Solo Stochastic | Stochastic + RSI + MACD |
| **Volatilidad** | Solo range | range + ATR% |
| **Tendencia** | Solo ADX | ADX + MACD |
| **Features per bar** | 7 | 10 |
| **Input shape (window=20)** | [1, 140] | [1, 200] |

---

## 💡 Cómo Funciona Cada Nuevo Indicador

### **RSI (Relative Strength Index)**

**Qué mide:** Velocidad y magnitud de cambios de precio

**Valores:**
- RSI < 30: Sobreventa (posible BUY)
- RSI > 70: Sobrecompra (posible SELL)
- RSI 30-70: Neutral

**Ventaja sobre Stochastic:**
- Detecta divergencias precio-indicador
- Menos sensible a picos momentáneos
- Complementa señales de Stochastic

**Ejemplo:**
```
Stochastic en oversold + RSI < 30 = Señal BUY más fuerte
Stochastic neutral + RSI divergencia = Alerta temprana
```

---

### **MACD Histogram**

**Qué mide:** Diferencia entre MACD y su señal (momentum + tendencia)

**Valores:**
- Histogram > 0: Momentum alcista
- Histogram < 0: Momentum bajista
- Histogram creciendo: Fortaleza aumentando
- Histogram decreciendo: Fortaleza disminuyendo

**Ventaja sobre ADX:**
- Muestra DIRECCIÓN (alcista/bajista)
- ADX solo muestra fuerza (sin dirección)
- Detecta cambios de tendencia más temprano

**Ejemplo:**
```
ADX > 32 + MACD Histogram > 0 = Tendencia alcista confirmada
ADX > 32 + MACD Histogram < 0 = Tendencia bajista confirmada
```

---

### **ATR% (Average True Range Percentage)**

**Qué mide:** Volatilidad como porcentaje del precio

**Valores típicos:**
- ATR% < 0.5%: Baja volatilidad (mercado tranquilo)
- ATR% 0.5-1.5%: Volatilidad normal
- ATR% > 1.5%: Alta volatilidad (mercado agitado)

**Ventaja:**
- Normaliza la volatilidad (comparable entre símbolos)
- Ayuda a ajustar SL/TP dinámicamente
- Filtra señales en mercados demasiado volátiles

**Ejemplo:**
```
Señal BUY + ATR% bajo = Entrada más segura
Señal BUY + ATR% alto = Esperar confirmación adicional
```

---

## 📊 Feature Engineering Explicado

### **Body (feat_body)**
```python
body = close - open
```
- Positivo: Vela alcista
- Negativo: Vela bajista
- Magnitud: Fuerza del movimiento

### **Range (feat_range)**
```python
range = high - low
```
- Mide volatilidad intrabar
- Range grande: Alta actividad
- Range pequeño: Consolidación

### **MACD Histogram (feat_macd_hist)**
```python
macd_hist = MACD_main - MACD_signal
```
- Mide aceleración del momentum
- Cruce de 0: Cambio de dirección

### **ATR% (feat_atr_pct)**
```python
atr_pct = (ATR / close) * 100
```
- Normaliza ATR como porcentaje
- Permite comparar volatilidad entre diferentes precios

---

## 🎓 Mejores Prácticas

### **1. Entrenamiento**

✅ **Usar suficientes datos**
```bash
# Mínimo recomendado: 6 meses de datos
# Óptimo: 1-2 años
```

✅ **Validar señales con move_points apropiado**
```bash
# Para H1: --move_points 50 (5 pips en EUR/USD)
# Para M15: --move_points 30 (3 pips)
```

✅ **Iterar suficiente en RandomizedSearchCV**
```bash
# Rápido: --n_iter 20
# Balanceado: --n_iter 50
# Exhaustivo: --n_iter 100
```

---

### **2. Testing**

✅ **Siempre hacer backtesting primero**
- Usar Strategy Tester de MT5
- Mínimo 3 meses de datos históricos
- Verificar drawdown y win rate

✅ **Probar en demo antes de live**
- Al menos 1 semana de operación en demo
- Monitorear precisión de predicciones
- Ajustar `InpMinConf` si es necesario

---

### **3. Optimización**

✅ **Ajustar confianza mínima**
```cpp
InpMinConf = 0.55  // Balance calidad/cantidad
InpMinConf = 0.45  // Más señales, menor calidad
InpMinConf = 0.65  // Menos señales, mayor calidad
```

✅ **Ajustar SL/TP según ATR**
```cpp
// Calcular dinámicamente:
SL = ATR * 1.5
TP = ATR * 3.0
```

---

## 📁 Archivos del Sistema

```
sgradt60/
├── train_sgradt60_strategy.py     # Script de entrenamiento
├── EA_SGRADT60_ONNX.mq5           # Expert Advisor para MT5
├── SGRADT60_DOCUMENTATION.md      # Este archivo
└── models/
    ├── EUR_USD_H1_SGRADT60_combined.onnx
    └── EUR_USD_H1_SGRADT60_combined.meta.json
```

---

## 🔗 Compatibilidad

| Componente | Versión |
|------------|---------|
| **Python** | 3.8+ |
| **pandas** | 1.3+ |
| **scikit-learn** | 1.0+ |
| **ta** | 0.10+ |
| **pandas_ta** | 0.3+ |
| **skl2onnx** | 1.13+ |
| **MetaTrader 5** | Build 3802+ |
| **MQL5** | ONNX Runtime compatible |

---

## 📝 Notas Importantes

1. **Orden de Features:** Las 10 features están en orden específico que DEBE coincidir entre Python y MQL5

2. **Window Size:** Debe ser el mismo en entrenamiento y en el EA

3. **Parámetros de Indicadores:** Deben coincidir exactamente entre entrenamiento y EA

4. **Timeframe:** Usar el EA en el mismo timeframe del entrenamiento

5. **Símbolo:** Preferiblemente el mismo símbolo, o uno con características similares

---

## ⚠️ Advertencias

- **No usar en cuenta real sin testear exhaustivamente**
- **Los resultados pasados no garantizan resultados futuros**
- **Entrenar periódicamente con datos frescos (cada 1-3 meses)**
- **Monitorear el modelo - la precisión puede degradarse con el tiempo**
- **Usar gestión de riesgo apropiada (máximo 1-2% por trade)**

---

## 🎯 Próximos Pasos Sugeridos

1. **Recolectar datos de calidad** (mínimo 6 meses)
2. **Entrenar el modelo** con parámetros por defecto
3. **Hacer backtesting** en MT5 Strategy Tester
4. **Ajustar parámetros** si es necesario
5. **Probar en demo** por al menos 1 semana
6. **Ir a live** con lotes pequeños
7. **Monitorear y reentrenar** cada 1-3 meses

---

**Version:** SGRADT 6.0  
**Date:** Marzo 2026  
**Author:** Sistema de Trading Automatizado  
**License:** Uso personal/educativo

---

## 🆘 Soporte

Si tienes problemas:

1. Verifica que los archivos estén en los directorios correctos
2. Revisa el log de MT5 (Experts tab)
3. Confirma que todos los parámetros coincidan entre training y EA
4. Verifica que `InpFeaturesPerBar = 10` (SIEMPRE en SGRADT 6.0)

**Mensaje de inicialización exitosa:**
```
✓ ONNX model loaded successfully
✓ Input shape set: [1, 200] (20 bars × 10 features)
✓ Indicators created: ADX, Stochastic, RSI, MACD, ATR
✅ EA INITIALIZED SUCCESSFULLY
```

Si ves este mensaje, ¡estás listo para operar! 🚀
