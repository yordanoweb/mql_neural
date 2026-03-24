# SGRADT 7.0 v3 - Cambios Implementados

## Resumen de Modificaciones

**SGRADT 7.0 v3** es una evolución de v2 que simplifica el modelo eliminando el **EMA Gate**, reduciendo las características de 7 a 6. Esta versión mantiene la arquitectura 100% dirigida por red neuronal, donde el modelo ONNX toma TODAS las decisiones de trading basándose únicamente en indicadores técnicos puros y contexto de volumen.

### 🔄 Cambios de v2 a v3

| Aspecto | v2 | v3 |
|---------|----|----|
| **Número de Features** | 7 | 6 |
| **EMA Gate** | ✅ Incluido | ❌ Removido |
| **Features Técnicas** | 5 (stoch_k, stoch_d, adx, pdi, mdi) | 5 (stoch_k, stoch_d, adx, pdi, mdi) |
| **Features de Contexto** | 2 (ema_gate, volume_gate) | 1 (volume_gate) |
| **Magic Number** | 7072 | 7073 |
| **Indicador EMA** | Requerido | No requerido |

**Filosofía de v3:** La red neuronal debe aprender patrones de precio directamente desde los indicadores técnicos sin depender de filtros de tendencia basados en EMA. El modelo se concentra en momentum (Stochastic), fuerza direccional (ADX/DI+/DI-) y contexto de volumen.

---

## 📊 CAMBIOS EN EL SCRIPT DE ENTRENAMIENTO (Python)

### Archivo: `train_sgradt70_strategy_v3.py`

#### 1. **Reducción de Features: De 7 a 6**

**v2 (7 features):**
- `feat_stoch_main` (Stochastic %K)
- `feat_stoch_signal` (Stochastic %D)
- `feat_adx` (ADX)
- `feat_pdi` (Positive Directional Indicator)
- `feat_mdi` (Minus Directional Indicator)
- `feat_ema_gate` ← **REMOVIDO EN v3**
- `feat_volume_gate`

**v3 (6 features):**
- `feat_stoch_main` (Stochastic %K)
- `feat_stoch_signal` (Stochastic %D)
- `feat_adx` (ADX)
- `feat_pdi` (Positive Directional Indicator)
- `feat_mdi` (Minus Directional Indicator)
- `feat_volume_gate` (ratio del volumen vs promedio de 10 barras)

#### 2. **Eliminación del Cálculo de EMA**

**v2:**
```python
# Calcular EMA 9
print("Calculando EMA 9...")
ema_inst = EMAIndicator(close=df['close'], window=args.ema_period)
df['ema9'] = ema_inst.ema_indicator()

# Calcular EMA Gate
df['ema_gate'] = 0.0
df.loc[df['open'] > df['ema9'], 'ema_gate'] = 1.0
df.loc[df['open'] < df['ema9'], 'ema_gate'] = -1.0
```

**v3:**
```python
# EMA y EMA Gate completamente removidos
# No se importa EMAIndicator
# No se calcula feat_ema_gate
```

#### 3. **Input Shape Actualizado**

El modelo ahora recibe `6 features × window_size` elementos:

```python
num_features_per_bar = 6  # Antes era 7
num_inputs = num_features_per_bar * args.window
# Ejemplo: 6 × 20 = 120 inputs
```

#### 4. **Metadata Actualizada**

```json
{
  "version": "SGRADT 7.0 v3",
  "strategy": "Neural Network Driven (6 Features - No EMA Gate)",
  "features_per_bar": 6,
  "feature_descriptions": {
    "feat_stoch_main": "Stochastic %K",
    "feat_stoch_signal": "Stochastic %D",
    "feat_adx": "Average Directional Index",
    "feat_pdi": "Positive Directional Indicator",
    "feat_mdi": "Minus Directional Indicator",
    "feat_volume_gate": "Volume ratio vs 10-bar average"
  },
  "notes": [
    "NN-driven strategy: model makes ALL decisions",
    "6 features (no EMA gate - removed in v3)",
    "Volume gate is the only context feature",
    "ATR-based SL/TP in MT5 code"
  ]
}
```

---

## 🤖 CAMBIOS EN EL EA DE MT5

### Archivo: `EA_SGRADT70_ONNX_v3.mq5`

#### 1. **Parámetros de Entrada Actualizados**

```mql5
// v2
input string InpModelName = "USTEC_M5_SGRADT70_v2.onnx";
input int    InpFeaturesPerBar = 7;    // 7 features
input int    InpMagic = 7072;
input int    InpEMAPeriod = 9;         // Requerido en v2

// v3
input string InpModelName = "USTEC_M5_SGRADT70_v3.onnx";
input int    InpFeaturesPerBar = 6;    // 6 features
input int    InpMagic = 7073;          // Nuevo magic number
// InpEMAPeriod REMOVIDO - no se necesita EMA
```

#### 2. **Eliminación del Handle de EMA**

**v2:**
```mql5
int g_ema_handle = INVALID_HANDLE;

// En OnInit()
g_ema_handle = iMA(_Symbol, _Period, InpEMAPeriod, 0, MODE_EMA, PRICE_CLOSE);
if(g_ema_handle == INVALID_HANDLE)
{
   Print("[ERROR] Cannot create EMA indicator");
   return INIT_FAILED;
}

// En OnDeinit()
if(g_ema_handle != INVALID_HANDLE)
   IndicatorRelease(g_ema_handle);
```

**v3:**
```mql5
// g_ema_handle completamente removido
// No se crea, no se libera, no se usa
```

#### 3. **PrepareInput() Simplificado**

La función ahora prepara solo **6 features** por barra:

**v2 (7 features):**
```mql5
for(int i = 0; i < window; i++)
{
   int offset = i * 7;  // 7 features
   
   input_buffer[offset + 0] = (float)stoch_k_b[i];
   input_buffer[offset + 1] = (float)stoch_d_b[i];
   input_buffer[offset + 2] = (float)adx_b[i];
   input_buffer[offset + 3] = (float)di_plus_b[i];
   input_buffer[offset + 4] = (float)di_minus_b[i];
   
   // Feature 6: EMA Gate
   double ema_gate = 0.0;
   if(open_b[i] > ema_b[i])
      ema_gate = 1.0;
   else if(open_b[i] < ema_b[i])
      ema_gate = -1.0;
   input_buffer[offset + 5] = (float)ema_gate;
   
   // Feature 7: Volume Gate
   double volume_gate = ...;
   input_buffer[offset + 6] = (float)volume_gate;
}
```

**v3 (6 features):**
```mql5
for(int i = 0; i < window; i++)
{
   int offset = i * 6;  // 6 features
   
   input_buffer[offset + 0] = (float)stoch_k_b[i];    // feat_stoch_main
   input_buffer[offset + 1] = (float)stoch_d_b[i];    // feat_stoch_signal
   input_buffer[offset + 2] = (float)adx_b[i];        // feat_adx
   input_buffer[offset + 3] = (float)di_plus_b[i];    // feat_pdi
   input_buffer[offset + 4] = (float)di_minus_b[i];   // feat_mdi
   
   // Feature 6: Volume Gate (único feature de contexto)
   double volume_gate = ...;
   input_buffer[offset + 5] = (float)volume_gate;     // feat_volume_gate
   
   // EMA Gate REMOVIDO - ya no existe
}
```

#### 4. **UpdatePanel() Sin EMA Gate**

**v2:**
```mql5
// EMA Gate display
string ema_status = "";
if(open_current > ema[0])
   ema_status = "ABOVE (+1.0)";
else if(open_current < ema[0])
   ema_status = "BELOW (-1.0)";
else
   ema_status = "EQUAL (0.0)";
panel += "   EMA Gate: " + ema_status + "\n";
```

**v3:**
```mql5
// EMA Gate display completamente removido
// Solo se muestra Volume Gate
panel += "   Volume Gate: " + DoubleToString(vol_ratio, 2) + "x avg\n";
```

#### 5. **Inicialización de Indicadores Simplificada**

**v3 solo requiere:**
- ADX (con DI+/DI-)
- Stochastic
- ATR

**No requiere:**
- EMA ❌

---

## 📈 FLUJO COMPLETO DE LA ESTRATEGIA v3

### Entrenamiento (Python):

1. **Carga de datos** → Debe incluir columna `tick_volume`
2. **Cálculo de indicadores** → ADX (con DI+/DI-), Stochastic
3. **Generación de features** → 6 features (sin ema_gate)
4. **Labeling forward-looking** → Busca profit >= min_profit_points en ventana futura
5. **Entrenamiento Random Forest** → Aprende patrones de las 6 features
6. **Export ONNX** → Modelo listo para MT5

### Ejecución en MT5:

1. **Inicialización** → Carga modelo ONNX + indicadores (ADX, Stoch, ATR)
2. **OnTick()** → Nueva barra o intervalo de tiempo
3. **PrepareInput()** → Calcula 6 features para ventana de lookback
4. **OnnxRun()** → Red neuronal predice: HOLD(0), BUY(1), o SELL(2)
5. **Verificación de confianza** → Si probabilidad >= InpMinConf
6. **Ejecución de trade** → 
   - BUY/SELL según decisión de la NN
   - SL/TP basado en ATR
   - Sin filtros manuales

---

## 🎯 VENTAJAS DE v3 SOBRE v2

### 1. **Modelo Más Simple y Directo**
- ✅ Menos features = menos complejidad
- ✅ Menos probabilidad de overfitting
- ✅ Entrenamiento más rápido
- ✅ Inferencia más rápida

### 2. **Independencia de Filtros de Tendencia**
- ❌ v2: Dependía del EMA gate para contexto de tendencia
- ✅ v3: La NN aprende tendencia directamente de ADX/DI+/DI-

### 3. **Menos Dependencias en MT5**
- ❌ v2: Requería 4 indicadores (EMA, ADX, Stoch, ATR)
- ✅ v3: Requiere solo 3 indicadores (ADX, Stoch, ATR)

### 4. **Features Más Puras**
- Stochastic: Momentum puro
- ADX/DI+/DI-: Fuerza y dirección de tendencia
- Volume Gate: Contexto de participación del mercado

La NN no está sesgada por un filtro de tendencia específico (EMA) - aprende patrones de tendencia de forma más general.

---

## 📋 CHECKLIST DE USO

### Para Entrenar:

```bash
python train_sgradt70_strategy_v3.py \
    --csv data/USTEC_M5.csv \
    --output ./onnx \
    --window 20 \
    --min_profit_points 20.0 \
    --future 50 \
    --stoch_k 7 \
    --stoch_d 3 \
    --adx_period 8 \
    --n_iter 10
```

**IMPORTANTE:** 
- El CSV debe tener la columna `tick_volume`
- No hay parámetro `--ema_period` (removido en v3)

### Para MT5:

1. Copiar `USTEC_M5_SGRADT70_v3.onnx` a `MQL5/Files/`
2. Compilar `EA_SGRADT70_ONNX_v3.mq5`
3. Configurar parámetros:
   - `InpFeaturesPerBar = 6` (CRÍTICO - era 7 en v2)
   - `InpWindowSize = 20` (debe coincidir con entrenamiento)
   - `InpMagic = 7073` (diferente de v2)
   - `InpATRMultiplierSL = 2.0` (ajustar según volatilidad)
   - `InpATRMultiplierTP = 3.0` (ajustar según ratio R:R deseado)
   - `InpMinConf = 0.55` (ajustar según precisión del modelo)

**NO configurar:**
- `InpEMAPeriod` (parámetro no existe en v3)

---

## ⚠️ DIFERENCIAS CLAVE: v1 vs v2 vs v3

| Aspecto | v1 (Original) | v2 | v3 |
|---------|--------------|----|----|
| **Features** | 5 (body, range, stoch_k, stoch_d, adx) | 7 (stoch_k, stoch_d, adx, pdi, mdi, ema_gate, volume_gate) | 6 (stoch_k, stoch_d, adx, pdi, mdi, volume_gate) |
| **EMA** | Filtro externo | Feature de entrada | No usado |
| **Volume** | Filtro externo | Feature de entrada | Feature de entrada |
| **Decisión** | NN + Filtros manuales | 100% NN | 100% NN |
| **Exit** | Cruce de EMA | SL/TP (ATR-based) | SL/TP (ATR-based) |
| **Magic** | 7070 | 7072 | 7073 |
| **Complejidad** | Media | Alta (7 features) | Media (6 features) |
| **Filosofía** | Híbrida | NN total con contexto de tendencia | NN total con contexto de volumen |

---

## 🎓 CUÁNDO USAR v3 vs v2

### Usar v3 si:
- ✅ Quieres un modelo más simple y rápido
- ✅ El mercado tiene patrones de tendencia diversos (no solo EMA 9)
- ✅ Prefieres que la NN aprenda tendencia de forma general
- ✅ Quieres menos dependencias de indicadores
- ✅ Buscas reducir riesgo de overfitting

### Usar v2 si:
- ✅ El mercado respeta fuertemente el EMA 9
- ✅ Quieres dar contexto explícito de tendencia a la NN
- ✅ Tienes suficientes datos de entrenamiento (v2 necesita más por tener más features)
- ✅ La validación de v2 muestra mejor rendimiento en tu instrumento específico

---

## 📊 VALIDACIÓN DE LABELS (Igual que v2)

La lógica de labeling permanece idéntica a v2:

```python
# BUY: Si sube >= threshold Y sube mucho más de lo que baja
if (max_up_move >= args.min_profit_points and 
    max_up_move >= max_down_move * args.min_profit_ratio):
    labels[i] = 1  # BUY signal

# SELL: Si baja >= threshold Y baja mucho más de lo que sube  
elif (max_down_move >= args.min_profit_points and
      max_down_move >= max_up_move * args.min_profit_ratio):
    labels[i] = 2  # SELL signal

# else: HOLD
```

### 🎯 Parámetros Recomendados por Mercado

Consulta la [tabla completa en la documentación de v2](README_SGRADT70_v2.md#-parámetros-recomendados-para-balance-saludable).

**Resumen rápido:**

| Mercado | min_profit_points | future | ratio |
|---------|------------------|--------|-------|
| **NAS100/S&P500** | 50 | 20 | 2.0 |
| **Forex Majors** | 15 | 30 | 1.8 |
| **BTC/ETH** | 80 | 15 | 2.5 |
| **FTSE/DAX** | 30 | 25 | 2.0 |

---

## 🔧 TROUBLESHOOTING

### Error: "Input buffer size incorrect"
**Solución:** Verificar que `InpFeaturesPerBar = 6` en MT5 (NO 7 como en v2).

### Error: "Cannot create EMA indicator"
**Solución:** Estás usando código de v2. En v3 no se usa EMA. Verifica que estés usando `EA_SGRADT70_ONNX_v3.mq5`.

### Modelo entrenado en v2 no funciona en v3
**Solución:** Los modelos NO son compatibles entre versiones. Debes:
1. Re-entrenar con `train_sgradt70_strategy_v3.py`
2. Usar el nuevo modelo `.onnx` generado
3. Configurar `InpFeaturesPerBar = 6` en MT5

### Modelo no genera señales
**Posibles causas:**
1. Confianza muy alta (`InpMinConf` → reducir a 0.50)
2. Datos insuficientes en entrenamiento
3. `min_profit_points` muy alto → reducir a 15-20

---

## 📊 COMPARACIÓN DE RENDIMIENTO v2 vs v3

**Hipótesis:**
- v3 puede tener **ligeramente menor precisión** en mercados que respetan fuertemente el EMA 9
- v3 puede tener **mejor generalización** en mercados con patrones de tendencia diversos
- v3 es **más rápido** en entrenamiento e inferencia
- v3 tiene **menor riesgo de overfitting** por tener menos features

**Recomendación:** Entrenar ambos modelos (v2 y v3) con los mismos datos y comparar métricas:
- Balanced Accuracy en validación
- Sharpe Ratio en backtest
- Drawdown máximo
- Win rate y profit factor

El modelo con mejores métricas en TU instrumento específico es el que debes usar.

---

## 🔄 MIGRACIÓN DE v2 a v3

### Script Python:

1. Descargar `train_sgradt70_strategy_v3.py`
2. Ejecutar con los mismos parámetros que v2 **EXCEPTO:**
   - Remover `--ema_period` (ya no existe)
3. Los archivos generados tendrán sufijo `_v3.onnx`

### EA de MT5:

1. Descargar `EA_SGRADT70_ONNX_v3.mq5`
2. Copiar modelo `_v3.onnx` a `MQL5/Files/`
3. Configurar:
   - `InpModelName` → archivo `_v3.onnx`
   - `InpFeaturesPerBar = 6`
   - `InpMagic = 7073`
4. Compilar y verificar sin errores

**IMPORTANTE:** No puedes usar un modelo v2 con el EA v3, ni viceversa. Los input shapes son diferentes.

---

## 📝 EJEMPLO DE USO COMPLETO

### 1. Entrenamiento

```bash
# Entrenar modelo v3
python train_sgradt70_strategy_v3.py \
    --csv data/EURUSD_M5.csv \
    --output ./onnx \
    --window 20 \
    --min_profit_points 15.0 \
    --future 30 \
    --min_profit_ratio 1.8 \
    --stoch_k 7 \
    --stoch_d 3 \
    --adx_period 8 \
    --n_iter 10
```

**Output esperado:**
```
SGRADT 7.0 v3 - NN-Driven Strategy (6 Features - No EMA Gate)
======================================================================
Archivos a procesar: 1
  - EURUSD_M5.csv
======================================================================

PROCESANDO: EURUSD_M5.csv
======================================================================
Datos cargados: 65000 barras
Calculando ADX...
Calculando Stochastic...
Calculando Volume Gate...
Datos válidos después de NaN: 64950 barras

SEÑALES DETECTADAS: EURUSD_M5.csv
======================================================================
  BUY  (1):   8234 señales ( 12.67%)
  SELL (2):   8891 señales ( 13.69%)
  HOLD (0):  47825 señales ( 73.64%)
======================================================================

ENTRENANDO RANDOM FOREST — EURUSD_M5.csv
======================================================================

Entrenamiento completado
  Mejor Balanced Accuracy: 0.6832
  Mejores parametros: {'n_estimators': 300, 'max_depth': 20, ...}

EXPORTANDO MODELO ONNX — EURUSD_M5.csv
======================================================================

Modelo guardado:   ./onnx/EURUSD_M5_SGRADT70_v3.onnx
Metadata guardado: ./onnx/EURUSD_M5_SGRADT70_v3.meta.json

RESUMEN — EURUSD_M5.csv
======================================================================
  Input shape: [1, 120]
  Features: 6 x 20 barras
  Output: 3 clases (HOLD=0, BUY=1, SELL=2)
  Accuracy: 0.6832

  Estrategia:
    - NN-driven: modelo decide basándose en 6 features
    - Features: stoch_k, stoch_d, adx, pdi, mdi, volume_gate
    - No EMA gate (removido en v3)
    - Exit: ATR-based SL/TP (configurado en MT5)
======================================================================
```

### 2. Configuración en MT5

**Parámetros del EA:**

```
======== AI MODEL ========
InpModelName = "EURUSD_M5_SGRADT70_v3.onnx"
InpMetaFile  = "EURUSD_M5_SGRADT70_v3.meta.json"
InpMinConf   = 0.55
InpWindowSize = 20
InpFeaturesPerBar = 6  ← CRÍTICO: 6 features (no 7)

======== INFERENCE ========
InpInferSeconds = 0
InpOneTradePerBar = true

======== SESSION ========
InpStartHour = 0
InpEndHour   = 24

======== INDICATORS ========
InpStochK = 7
InpStochD = 3
InpADXPeriod = 8

======== RISK (ATR-BASED) ========
InpLot = 0.1
InpMagic = 7073
InpATRPeriod = 14
InpATRMultiplierSL = 2.0
InpATRMultiplierTP = 3.0
```

### 3. Verificación en Logs de MT5

```
----------------------------------------------------------------------
    SGRADT 7.0 v3 - NN-DRIVEN STRATEGY (6 Features)
    No EMA Gate - Volume Gate Only
----------------------------------------------------------------------

Loading ONNX model: EURUSD_M5_SGRADT70_v3.onnx
[OK] ONNX model loaded successfully
[OK] Input shape set: [1, 120] (20 bars x 6 features)

Initializing indicators...
[OK] Indicators created:
     - ADX(8)
     - Stochastic(7,3)
     - ATR(14)

[OK] All indicators ready with 50 bars

=== SYMBOL INFORMATION ===
Symbol: EURUSD
Timeframe: M5
Digits: 5
Point: 0.00001

=== INDICATOR PARAMETERS ===
Stochastic: (7,3)
ADX: Period=8
ATR: Period=14

=== STRATEGY ===
Type: NN-Driven (Neural Network decides everything)
Features: 6 (stoch_k, stoch_d, adx, pdi, mdi, volume_gate)
Exit: ATR-based SL/TP (SL=2.0xATR, TP=3.0xATR)
No EMA gate - removed in v3
No manual gates - NN makes all decisions

Inference mode: New bar only

----------------------------------------------------------------------
    EA INITIALIZED SUCCESSFULLY
----------------------------------------------------------------------
```

---

## 📊 ESTRUCTURA DE ARCHIVOS

```
project/
├── train_sgradt70_strategy_v3.py  ← Script de entrenamiento
├── EA_SGRADT70_ONNX_v3.mq5        ← Expert Advisor para MT5
├── README_SGRADT70_v3.md          ← Esta documentación
│
├── data/
│   └── EURUSD_M5.csv              ← Datos de entrenamiento (con tick_volume)
│
├── onnx/
│   ├── EURUSD_M5_SGRADT70_v3.onnx      ← Modelo entrenado
│   └── EURUSD_M5_SGRADT70_v3.meta.json ← Metadata
│
└── MQL5/Files/  (en MT5)
    └── EURUSD_M5_SGRADT70_v3.onnx ← Copiar aquí para usar en MT5
```

---

## 🧪 TESTING Y VALIDACIÓN

### 1. Validación del Modelo (Python)

El script genera automáticamente métricas de validación:

```
Mejor Balanced Accuracy: 0.6832
```

**Interpretación:**
- < 0.60: Modelo débil, considerar ajustar parámetros
- 0.60 - 0.70: Modelo aceptable
- 0.70 - 0.80: Modelo bueno
- > 0.80: Modelo excelente (cuidado con overfitting)

### 2. Backtest en MT5

Antes de usar en vivo:

1. **Strategy Tester** → Modo "Every tick"
2. Periodo: Mínimo 3 meses de datos out-of-sample
3. Métricas a revisar:
   - Profit Factor > 1.5
   - Sharpe Ratio > 1.0
   - Max Drawdown < 20%
   - Win Rate > 50%
   - Recovery Factor > 2.0

### 3. Forward Testing en Demo

1. Ejecutar en cuenta demo por 2-4 semanas
2. Monitorear:
   - Frecuencia de trades (debe ser razonable)
   - Tamaño de SL/TP (debe ser proporcional a ATR)
   - Comportamiento en diferentes condiciones de mercado

---

## 🚀 OPTIMIZACIÓN AVANZADA

### 1. Ensemble de Modelos

Combinar v3 con otros timeframes:

```bash
# Entrenar modelos para diferentes timeframes
python train_sgradt70_strategy_v3.py --csv EURUSD_M5.csv  --window 20
python train_sgradt70_strategy_v3.py --csv EURUSD_M15.csv --window 20
python train_sgradt70_strategy_v3.py --csv EURUSD_H1.csv  --window 15
```

En MT5, usar 3 EAs diferentes (uno por timeframe) con diferentes magic numbers.

### 2. Optimización de Hiperparámetros

```bash
# Explorar diferentes configuraciones
for MIN_PROFIT in 10 15 20 25; do
  for FUTURE in 20 30 40 50; do
    for RATIO in 1.5 1.8 2.0 2.5; do
      python train_sgradt70_strategy_v3.py \
        --csv data.csv \
        --min_profit_points $MIN_PROFIT \
        --future $FUTURE \
        --min_profit_ratio $RATIO \
        --output ./onnx/test_${MIN_PROFIT}_${FUTURE}_${RATIO}/
    done
  done
done
```

Comparar balanced_accuracy y seleccionar la mejor combinación.

### 3. Feature Engineering Adicional

Aunque v3 tiene 6 features, puedes experimentar agregando:
- RSI (Relative Strength Index)
- Bollinger Bands width
- ATR normalization
- Price distance from recent high/low

**IMPORTANTE:** Cada feature adicional requiere:
1. Actualizar el script Python
2. Actualizar el EA de MT5 (PrepareInput)
3. Actualizar `InpFeaturesPerBar`

---

## 📖 GLOSARIO DE TÉRMINOS

| Término | Definición |
|---------|-----------|
| **Feature** | Variable de entrada para la red neuronal |
| **Label** | Clase objetivo (HOLD/BUY/SELL) para entrenamiento |
| **Window** | Número de barras históricas que ve el modelo |
| **Balanced Accuracy** | Métrica de precisión balanceada entre clases |
| **Forward-looking** | Mirar hacia el futuro para crear labels |
| **ATR** | Average True Range - medida de volatilidad |
| **Gate** | Filtro o feature de contexto (en v3 solo volume_gate) |
| **ONNX** | Formato de modelo neural network para MT5 |
| **Inference** | Proceso de predicción usando el modelo |

---

## ❓ FAQ - Preguntas Frecuentes

### P: ¿Por qué se eliminó el EMA gate en v3?

**R:** Para simplificar el modelo y permitir que la NN aprenda patrones de tendencia de forma más general. El ADX con DI+/DI- ya proporciona información de tendencia sin sesgar el modelo hacia un indicador específico (EMA 9).

### P: ¿v3 es mejor que v2?

**R:** Depende del mercado. v3 es más simple y general. v2 puede funcionar mejor en mercados que respetan fuertemente el EMA 9. Debes probar ambos con tus datos.

### P: ¿Puedo usar un modelo v2 en el EA v3?

**R:** NO. Los input shapes son diferentes (7 features vs 6 features). Debes usar modelos v3 con EA v3.

### P: ¿Cómo sé si mi modelo está en overfitting?

**R:** Señales de overfitting:
- Balanced accuracy > 0.85 en entrenamiento
- Rendimiento excelente en backtest pero malo en forward testing
- Muchas features relativas a pocas muestras de datos
- Modelo funciona solo en periodo específico de entrenamiento

### P: ¿Cuántos datos necesito para entrenar?

**R:** Mínimo recomendado:
- **Datos de entrenamiento:** 6 meses de datos M5 (~50,000 barras)
- **Datos de validación:** 2 meses adicionales
- **Datos de testing:** 2 meses adicionales

Total: ~10 meses de datos históricos.

### P: ¿Qué hacer si el modelo no genera suficientes señales?

**R:** Posibles soluciones:
1. Reducir `min_profit_points`
2. Aumentar `future` (ventana de búsqueda)
3. Reducir `min_profit_ratio`
4. Reducir `InpMinConf` en MT5

### P: ¿Puedo usar v3 en múltiples pares simultáneamente?

**R:** Sí, pero entrena un modelo SEPARADO para cada par:
- EURUSD → modelo específico para EURUSD
- GBPUSD → modelo específico para GBPUSD
- Etc.

Cada instrumento tiene patrones únicos que el modelo debe aprender.

---

## 📚 RECURSOS ADICIONALES

### Documentación Relacionada
- [README_SGRADT70_v2.md](README_SGRADT70_v2.md) - Versión anterior con 7 features
- Documentación oficial de ONNX: https://onnx.ai/
- MQL5 ONNX Integration: https://www.mql5.com/en/docs/integration/onnx

### Librerías Python Usadas
- `pandas` - Manipulación de datos
- `numpy` - Operaciones numéricas
- `scikit-learn` - Machine learning (Random Forest)
- `skl2onnx` - Exportación a ONNX
- `ta` - Indicadores técnicos

### Instalación de Dependencias

```bash
pip install pandas numpy scikit-learn skl2onnx ta
```

---

## 🔔 CHANGELOG

### v3.0.0 (2025)
- ✅ **NUEVO:** Reducción a 6 features (eliminado EMA gate)
- ✅ **NUEVO:** Simplificación del modelo
- ✅ **NUEVO:** Magic number actualizado a 7073
- ❌ **REMOVIDO:** EMA indicator y EMA gate feature
- ❌ **REMOVIDO:** Parámetro `--ema_period` en script Python
- ❌ **REMOVIDO:** Parámetro `InpEMAPeriod` en EA de MT5
- 🔧 **CAMBIADO:** `InpFeaturesPerBar` de 7 a 6
- 🔧 **CAMBIADO:** Input shape de [1, 140] a [1, 120] (con window=20)

---

## 📧 SOPORTE Y CONTACTO

Para preguntas, reportes de bugs o sugerencias:

1. Revisar esta documentación completa
2. Verificar la sección de Troubleshooting
3. Revisar el FAQ
4. Comparar con documentación de v2 si vienes de esa versión

---

## ⚖️ DISCLAIMER

Este software es para fines educativos y de investigación. El trading automatizado implica riesgos significativos:

- ⚠️ **Riesgo de pérdida:** Puedes perder todo tu capital
- ⚠️ **Sin garantías:** Los resultados pasados no garantizan resultados futuros
- ⚠️ **Responsabilidad:** El usuario es responsable de sus decisiones de trading
- ⚠️ **Testing:** Siempre prueba exhaustivamente en demo antes de usar en vivo

**RECOMENDACIÓN:** 
1. Entrena con datos de calidad (mínimo 10 meses)
2. Backtest exhaustivo (mínimo 3 meses out-of-sample)
3. Forward test en demo (mínimo 1 mes)
4. Empieza con lotes pequeños en vivo
5. Monitorea constantemente el rendimiento

---

## 📝 NOTAS FINALES

**SGRADT 7.0 v3** representa una evolución hacia la simplificación:

**Filosofía:**
- **Menos es más:** 6 features bien seleccionadas > 7 features con redundancia
- **Generalización:** Dejar que la NN aprenda patrones sin sesgos de filtros específicos
- **Eficiencia:** Modelo más rápido en entrenamiento e inferencia
- **Pureza:** Features técnicas puras + contexto de volumen

**Comparación de enfoques:**
- **v1:** "La NN sugiere, las reglas deciden"
- **v2:** "La NN ve todo (incluido EMA gate) y decide todo"
- **v3:** "La NN ve indicadores puros (sin EMA gate) y decide todo"

**PRÓXIMOS PASOS SUGERIDOS:**
1. Entrenar modelos v3 con tus datos históricos
2. Comparar rendimiento v3 vs v2 en backtest
3. Forward testing en demo por 1-2 meses
4. Ajustar parámetros según resultados
5. Considerar ensemble multi-timeframe

**El mejor modelo es el que funciona mejor en TU mercado con TUS datos.**

---

**Versión:** 7.0.3  
**Fecha:** Marzo 2025  
**Autor:** SGRADT Team  
**Licencia:** Educational Use Only

---

*"Simplicity is the ultimate sophistication." - Leonardo da Vinci*
