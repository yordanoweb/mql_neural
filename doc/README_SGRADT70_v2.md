# SGRADT 7.0 v2 - Cambios Implementados

## Resumen de Modificaciones

Se ha transformado completamente la arquitectura de la estrategia SGRADT 7.0 para convertirla en una estrategia 100% dirigida por red neuronal, donde el modelo ONNX toma TODAS las decisiones de trading.

---

## 📊 CAMBIOS EN EL SCRIPT DE ENTRENAMIENTO (Python)

### Archivo: `train_sgradt70_strategy_v2.py`

#### 1. **Nuevas Features: De 5 a 7**

**ANTES (5 features):**
- `feat_body` (close - open)
- `feat_range` (high - low)  
- `feat_stoch_main` (Stochastic %K)
- `feat_stoch_signal` (Stochastic %D)
- `feat_adx` (ADX)

**AHORA (7 features):**
- `feat_stoch_main` (Stochastic %K)
- `feat_stoch_signal` (Stochastic %D)
- `feat_adx` (ADX)
- `feat_pdi` (Positive Directional Indicator)
- `feat_mdi` (Minus Directional Indicator)
- **`feat_ema_gate`** ← NUEVO (1.0 si Open > EMA9, -1.0 si Open < EMA9, 0.0 si igual)
- **`feat_volume_gate`** ← NUEVO (ratio del volumen actual vs promedio de 10 barras)

#### 2. **Los "Gates" Ahora son Features**

**CONCEPTO CLAVE:** Los gates que antes eran filtros externos (EMA Gate y Volume Gate) ahora son **features de entrada** para la red neuronal.

```python
# EMA Gate como feature
df['ema_gate'] = 0.0
df.loc[df['open'] > df['ema9'], 'ema_gate'] = 1.0   # Above EMA
df.loc[df['open'] < df['ema9'], 'ema_gate'] = -1.0  # Below EMA

# Volume Gate como feature  
df['volume_avg_10'] = df['tick_volume'].rolling(window=10).mean()
df['volume_gate'] = df['tick_volume'] / df['volume_avg_10']
```

#### 3. **Lógica de Labeling Simplificada**

**ANTES:** Lógica compleja con condiciones manuales de ADX, Stochastic, cruces de EMA, etc.

**AHORA:** Forward-looking puro y simple:
```python
# Buscar el mejor movimiento en la ventana futura
for j in range(i+1, min(i+args.future+1, len(df))):
    future_price = df['close'].iloc[j]
    profit_points = (future_price - entry_price) / entry_price * 10000
    
    if profit_points >= args.min_profit_points:
        labels[i] = 1  # BUY
    elif loss_points >= args.min_profit_points:
        labels[i] = 2  # SELL
```

**Filosofía:** La red neuronal aprende los patrones de los gates y todas las condiciones de forma automática. No hay lógica manual de entrada/salida.

#### 4. **Requisito Nuevo: Columna `tick_volume`**

El CSV debe incluir la columna `tick_volume` para calcular el volume_gate.

---

## 🤖 CAMBIOS EN EL EA DE MT5

### Archivo: `EA_SGRADT70_ONNX_v2.mq5`

#### 1. **Eliminación Completa de Gates como Filtros**

**ANTES:**
```mql5
// Filtros manuales
if(!HasPosition(POSITION_TYPE_BUY) && 
   EMAGateAllows(predicted_class) &&    // ← Gate manual
   VolumeGateAllows())                   // ← Gate manual
{
    // Open BUY
}
```

**AHORA:**
```mql5
// Solo confianza del modelo
if(predicted_class == 1)  // BUY signal from NN
{
    if(!HasPosition(POSITION_TYPE_BUY))
    {
        // Open BUY - sin gates, la NN decide todo
    }
}
```

**FILOSOFÍA:** El modelo ya recibió los gates como features durante el entrenamiento. No necesitamos filtrarlos manualmente.

#### 2. **SL/TP Basado en ATR**

**ANTES:**
```mql5
input double InpStopPoints = 50.0;  // Stop Loss in POINTS
input double InpTakePoints = 100.0; // Take Profit in POINTS

double sl = ask - InpStopPoints * _Point;
double tp = ask + InpTakePoints * _Point;
```

**AHORA:**
```mql5
input int    InpATRPeriod     = 14;    // ATR period
input double InpATRMultiplierSL = 2.0; // ATR multiplier for Stop Loss
input double InpATRMultiplierTP = 3.0; // ATR multiplier for Take Profit

// Dinámico basado en volatilidad
double atr_value = atr_buf[0];
double sl = ask - (atr_value * InpATRMultiplierSL);
double tp = ask + (atr_value * InpATRMultiplierTP);
```

**VENTAJAS:**
- ✅ Se adapta automáticamente a la volatilidad del mercado
- ✅ Más riesgo cuando el mercado es volátil
- ✅ Menos riesgo cuando el mercado está tranquilo

#### 3. **Nuevas Features en PrepareInput()**

La función ahora calcula 7 features por barra:

```mql5
for(int i = 0; i < window; i++)
{
    int offset = i * 7;  // 7 features per bar
    
    input_buffer[offset + 0] = (float)stoch_k_b[i];    // Stochastic K
    input_buffer[offset + 1] = (float)stoch_d_b[i];    // Stochastic D
    input_buffer[offset + 2] = (float)adx_b[i];        // ADX
    input_buffer[offset + 3] = (float)di_plus_b[i];    // DI+
    input_buffer[offset + 4] = (float)di_minus_b[i];   // DI-
    
    // Feature 6: EMA Gate
    double ema_gate = 0.0;
    if(open_b[i] > ema_b[i])
        ema_gate = 1.0;   // Above EMA
    else if(open_b[i] < ema_b[i])
        ema_gate = -1.0;  // Below EMA
    input_buffer[offset + 5] = (float)ema_gate;
    
    // Feature 7: Volume Gate
    double vol_avg = 0.0;
    for(int j = i; j < i + 10; j++)
        vol_avg += (double)volume_b[j];
    vol_avg /= 10.0;
    double volume_gate = (vol_avg > 0) ? (double)volume_b[i] / vol_avg : 1.0;
    input_buffer[offset + 6] = (float)volume_gate;
}
```

#### 4. **Eliminación de Lógica de Exit Manual**

**ANTES:** Función `CheckEMAExit()` que cerraba posiciones cuando el precio cruzaba la EMA.

**AHORA:** Exit por SL/TP automático (ATR-based). El modelo podría aprender patrones de exit si se entrena adecuadamente.

#### 5. **Parámetros Actualizados**

```mql5
input int InpFeaturesPerBar = 7;  // Era 5, ahora es 7
input int InpMagic = 7072;        // Cambiado de 7070 a 7072 para diferenciar
```

---

## 📈 FLUJO COMPLETO DE LA ESTRATEGIA v2

### Entrenamiento (Python):

1. **Carga de datos** → Debe incluir columna `tick_volume`
2. **Cálculo de indicadores** → EMA, ADX (con DI+/DI-), Stochastic
3. **Generación de features** → 7 features (incluye ema_gate y volume_gate)
4. **Labeling forward-looking** → Busca profit >= min_profit_points en ventana futura
5. **Entrenamiento Random Forest** → Aprende patrones de las 7 features
6. **Export ONNX** → Modelo listo para MT5

### Ejecución en MT5:

1. **Inicialización** → Carga modelo ONNX + indicadores (EMA, ADX, Stoch, ATR)
2. **OnTick()** → Nueva barra o intervalo de tiempo
3. **PrepareInput()** → Calcula 7 features para ventana de lookback
4. **OnnxRun()** → Red neuronal predice: HOLD(0), BUY(1), o SELL(2)
5. **Verificación de confianza** → Si probabilidad >= InpMinConf
6. **Ejecución de trade** → 
   - BUY/SELL según decisión de la NN
   - SL/TP basado en ATR
   - Sin filtros manuales

---

## 🎯 VENTAJAS DE LA NUEVA ARQUITECTURA

### 1. **Decisión 100% NN-Driven**
- ❌ ANTES: Modelo predecía + filtros manuales decidían
- ✅ AHORA: Modelo ve todos los datos y decide todo

### 2. **Gates como Contexto**
- La NN aprende **cuándo** los gates son importantes
- No hay reglas rígidas tipo "si volume < 0.8, no tradear"
- El modelo puede ignorar el volume gate si otros factores son fuertes

### 3. **Risk Management Adaptativo**
- ATR se ajusta automáticamente a la volatilidad
- En mercados volátiles: SL/TP más amplios
- En mercados tranquilos: SL/TP más ajustados

### 4. **Más Features, Mejor Contexto**
- DI+/DI- dan dirección de la tendencia
- EMA Gate da posición estructural del precio
- Volume Gate da contexto de participación del mercado

---

## 📋 CHECKLIST DE USO

### Para Entrenar:

```bash
python train_sgradt70_strategy_v2.py \
    --csv data/USTEC_M5.csv \
    --output ./onnx \
    --window 20 \
    --min_profit_points 20.0 \
    --future 50 \
    --ema_period 9 \
    --stoch_k 7 \
    --stoch_d 3 \
    --adx_period 8 \
    --n_iter 10
```

**IMPORTANTE:** El CSV debe tener la columna `tick_volume`.

### Para MT5:

1. Copiar `USTEC_M5_SGRADT70_v2.onnx` a `MQL5/Files/`
2. Compilar `EA_SGRADT70_ONNX_v2.mq5`
3. Configurar parámetros:
   - `InpFeaturesPerBar = 7` (CRÍTICO)
   - `InpWindowSize = 20` (debe coincidir con entrenamiento)
   - `InpATRMultiplierSL = 2.0` (ajustar según volatilidad)
   - `InpATRMultiplierTP = 3.0` (ajustar según ratio R:R deseado)
   - `InpMinConf = 0.55` (ajustar según precisión del modelo)

---

## ⚠️ DIFERENCIAS CLAVE vs v1

| Aspecto | v1 (Original) | v2 (Nueva) |
|---------|--------------|------------|
| **Features** | 5 (body, range, stoch_k, stoch_d, adx) | 7 (stoch_k, stoch_d, adx, pdi, mdi, ema_gate, volume_gate) |
| **Gates** | Filtros externos en MT5 | Features de entrada para NN |
| **Decisión** | NN + Filtros manuales | 100% NN |
| **Exit** | Cruce de EMA | SL/TP (ATR-based) |
| **SL/TP** | Puntos fijos | Dinámico (ATR) |
| **Complejidad** | Media | Más simple (NN decide todo) |

---

## 🤔 ¿POR QUÉ YA NO SE NECESITAN STOCH_OVERSOLD Y STOCH_OVERBOUGHT?

### Cambio Fundamental: De Reglas Manuales a Aprendizaje Automático

**VERSIÓN ANTERIOR (v1) - Lógica Manual con Thresholds:**

El script original usaba estos parámetros para crear reglas manuales de entrada:

```python
# v1: Reglas codificadas manualmente
stoch_oversold_cross = (
    (stoch_k_1 < stoch_d_1) and 
    (stoch_k_0 > stoch_d_0) and 
    (stoch_k_0 <= args.stoch_oversold)  # ← Threshold fijo en 20
)

# Solo marcar BUY si Stochastic está en zona de sobreventa
buy_signal = above_ema and adx_strong and stoch_oversold_cross
```

**Problema:** Esto limita el modelo a casos muy específicos y puede perder señales válidas.

**VERSIÓN NUEVA (v2) - El Modelo Aprende los Thresholds:**

Ahora simplemente damos el valor crudo del Stochastic:

```python
# v2: Valores crudos sin filtros
df['feat_stoch_main'] = df['stoch_k']    # Valor 0-100 completo
df['feat_stoch_signal'] = df['stoch_d']  # Sin restricciones

# La red neuronal recibe TODOS los valores posibles
```

### ¿Por Qué Esto Es Mejor?

#### 1. **El Random Forest Aprende Automáticamente los Niveles Óptimos**

Cuando entrenas un Random Forest, internamente hace esto:

```
Árbol 1: Si stoch_k < 25 → BUY (73% accuracy)
Árbol 2: Si stoch_k < 18 AND volume_gate > 1.2 → BUY (81% accuracy)
Árbol 3: Si stoch_k < 30 AND pdi > mdi → BUY (76% accuracy)
... (hasta 500 árboles)
```

**El modelo encuentra sus propios thresholds óptimos:**
- ✅ Puede descubrir que 18 es mejor que 20
- ✅ Puede descubrir que en ciertos contextos, 35 es el nivel clave
- ✅ Los thresholds cambian según otros indicadores (ADX, volumen, etc.)

#### 2. **Más Oportunidades de Trade**

```python
# v1: Solo señales en extremos (oversold/overbought)
Resultado: 500 señales BUY, 480 señales SELL en 10,000 barras

# v2: Señales basadas en profit futuro sin restricciones
Resultado: 1,200 señales BUY, 1,150 señales SELL en 10,000 barras
```

**Más datos de entrenamiento = Mejor modelo**

#### 3. **Captura Patrones Complejos**

Ejemplo de patrón que v1 NUNCA detectaría:

```
Stochastic en 55 (neutral, ignorado por v1)
+ Rebotando desde 30 en últimas 3 barras
+ ADX creciendo
+ Volumen alto
= Señal fuerte de continuación
```

v2 puede aprender esto porque ve la ventana completa de valores.

#### 4. **Menos Overfitting a "Números Mágicos"**

Los niveles 20/80 son **convenciones**, no leyes físicas:
- En cripto, 30/70 podría funcionar mejor
- En Forex, 15/85 podría ser óptimo  
- En índices volátiles, 25/75

**v2 deja que el modelo descubra qué funciona en TUS datos específicos**

### Ejemplo Práctico: Señal Perdida en v1

**Escenario:**
- Stochastic K = 45 (zona neutral)
- ADX = 35 (tendencia fuerte)
- DI+ = 30, DI- = 10 (tendencia alcista clara)
- Open > EMA9
- Volumen 1.5x promedio

**v1:** ❌ Ignora la señal (Stoch no está en oversold)  
**v2:** ✅ El modelo ve el patrón completo y puede predecir BUY con 75% de confianza

### Comparación Resumida

| Aspecto | v1 (Con Thresholds) | v2 (Sin Thresholds) |
|---------|---------------------|---------------------|
| **Flexibilidad** | Baja (reglas fijas 20/80) | Alta (modelo decide) |
| **Datos de entrenamiento** | Menos (filtrado estricto) | Más (todos los casos) |
| **Generalización** | Puede overfit a 20/80 | Aprende patrones reales |
| **Oportunidades** | Solo extremos | Todo el rango |
| **Adaptabilidad** | Manual (cambiar código) | Automática (re-entrenar) |

### Conclusión

No necesitamos `stoch_oversold` y `stoch_overbought` en v2 porque:

1. ✅ El Random Forest aprende automáticamente qué niveles importan
2. ✅ Puede descubrir thresholds mejores que 20/80
3. ✅ Considera el contexto (otros indicadores) al evaluar Stochastic
4. ✅ No perdemos señales válidas por reglas arbitrarias
5. ✅ El modelo es más robusto en diferentes condiciones de mercado

**Filosofía v2:** Dale al modelo la información cruda y déjalo aprender. Es más inteligente de lo que creemos. 🧠

---

## ⚠️ PROBLEMA COMÚN: OVER-LABELING (DEMASIADAS SEÑALES)

### Síntoma

Al entrenar el modelo, ves algo como esto:

```
BUY  (1):  33,730 señales (50.93%)  ← ¡DEMASIADO!
SELL (2):  21,266 señales (32.11%)  ← ¡DEMASIADO!
HOLD (0):  11,235 señales (16.96%)  ← ¡Solo 17% HOLD!
```

**Total de señales de trading: 83% de las barras = PROBLEMA GRAVE**

### ¿Por Qué Sucede?

Con parámetros muy permisivos:
- `--min_profit_points 5` (muy bajo)
- `--future 8` (ventana muy corta)

El algoritmo dice: *"Si en las próximas 8 barras el precio se mueve 5 puntos en cualquier dirección → SEÑAL"*

En mercados volátiles, esto pasa **casi siempre**, resultando en señales en el 80-90% de las barras.

### ❌ Consecuencias

1. **Overtrading extremo** - El EA abrirá posición tras posición
2. **Modelo sesgado** - Aprende que "siempre hay que estar en el mercado"
3. **Costos de spread** - Con 1 pip de spread, harás ~83 trades por cada 100 barras
4. **Pérdida garantizada** - Los costos de transacción destruyen la cuenta

### ✅ Solución: Validación Balanceada

La **v2 corregida** usa esta lógica:

```python
# Buscar el MEJOR profit y el PEOR loss en la ventana futura
for j in range(i+1, i+future+1):
    profit = (future_price - entry_price) / entry_price * 10000
    loss = (entry_price - future_price) / entry_price * 10000
    
    max_profit = max(max_profit, profit)
    max_loss = max(max_loss, loss)

# Validar dirección dominante
if (max_profit >= min_profit_points and 
    max_profit > max_loss * min_profit_ratio):  # Profit debe ser mayor que loss
    labels[i] = 1  # BUY - dirección alcista clara
    
elif (max_loss >= min_profit_points and
      max_loss > max_profit * min_profit_ratio):  # Loss debe ser mayor que profit
    labels[i] = 2  # SELL - dirección bajista clara
    
# else: HOLD - no hay dirección clara o movimiento bidireccional
```

**Clave:** Solo etiquetamos cuando hay **dirección dominante**, no cuando el precio oscila en ambas direcciones.

### 🎯 Parámetros Recomendados para Balance Saludable

**Para M5 (5 minutos):**
```bash
python train_sgradt70_strategy_v2.py \
    --csv data.csv \
    --min_profit_points 10 \       # Mínimo 10 puntos de profit
    --future 20 \                  # Ventana de ~2 horas
    --min_profit_ratio 1.3 \       # Profit debe ser 1.3x el loss
    --window 20
```

**Resultado esperado:** 
```
BUY  (1):  5,000-8,000 señales (15-25%)
SELL (2):  4,800-7,500 señales (15-25%)  
HOLD (0): 50,000-55,000 señales (50-70%)  ← Balance saludable
```

**Para M15 (15 minutos):**
```bash
--min_profit_points 15 \
--future 30 \
--min_profit_ratio 1.4
```

**Para H1 (1 hora):**
```bash
--min_profit_points 30 \
--future 40 \
--min_profit_ratio 1.5
```

### 📊 Ajuste Dinámico Según Resultados

**Si obtienes 0% señales o muy pocas (< 5%):**
```bash
# Opción 1: Reducir threshold de profit
--min_profit_points 8  # Reducir de 10 a 8

# Opción 2: Reducir ratio requerido
--min_profit_ratio 1.1  # Reducir de 1.3 a 1.1

# Opción 3: Aumentar ventana de búsqueda
--future 30  # Aumentar de 20 a 30
```

**Si obtienes > 40% señales (demasiadas):**
```bash
# Opción 1: Aumentar threshold
--min_profit_points 15  # Aumentar de 10 a 15

# Opción 2: Aumentar ratio
--min_profit_ratio 1.5  # Aumentar de 1.3 a 1.5

# Opción 3: Reducir ventana
--future 15  # Reducir de 20 a 15
```

### 📊 Guía de Distribución de Señales

| % HOLD | % BUY+SELL | Diagnóstico | Acción |
|--------|------------|-------------|--------|
| 50-70% | 30-50% | ✅ **ÓPTIMO** | Balance perfecto para trading activo |
| 70-85% | 15-30% | ✅ **CONSERVADOR** | Bueno para trading selectivo |
| 40-50% | 50-60% | ⚠️ **ACTIVO** | Verificar que no sea overtrading |
| 85-95% | 5-15% | ⚠️ **MUY SELECTIVO** | Pocas oportunidades, pero alta calidad |
| < 40% | > 60% | 🚨 **OVER-LABELING** | Aumentar min_profit_ratio |
| > 95% | < 5% | 🚨 **SUB-LABELING** | Reducir thresholds |

### 💡 Regla de Oro

> **"Un buen modelo de trading encuentra dirección clara en 20-40% de las barras. Si predice señal en > 60% o < 5%, algo está mal"**

El objetivo es capturar movimientos direccionales genuinos, no ruido del mercado.

---

## 🔧 TROUBLESHOOTING

### Error: "Columna 'tick_volume' no encontrada"
**Solución:** Exportar datos de MT5 con volumen de tick incluido.

### Error: "Input buffer size incorrect"
**Solución:** Verificar que `InpFeaturesPerBar = 7` y coincida con el modelo entrenado.

### Modelo no genera señales
**Posibles causas:**
1. Confianza muy alta (`InpMinConf` → reducir a 0.50)
2. Datos insuficientes en entrenamiento
3. `min_profit_points` muy alto → reducir a 15-20

---

## 📊 RECOMENDACIONES DE OPTIMIZACIÓN

1. **Entrenar con diferentes timeframes**
   - M5 para scalping
   - M15 para swing
   - H1 para position trading

2. **Ajustar ATR multipliers según mercado**
   - Forex: SL=1.5, TP=2.5
   - Índices: SL=2.0, TP=3.0
   - Cripto: SL=3.0, TP=5.0 (más volátil)

3. **Experimentar con window size**
   - Window pequeño (10-15): Más reactivo
   - Window medio (20-30): Balanceado
   - Window grande (40-50): Más conservador

4. **Validación cruzada**
   - Train: 70% datos más antiguos
   - Validation: 15% datos medios
   - Test: 15% datos más recientes

---

## 📝 NOTAS FINALES

Esta versión v2 representa un cambio fundamental en la filosofía de la estrategia:

**v1:** "La NN sugiere, las reglas deciden"
**v2:** "La NN ve todo y decide todo"

El modelo ahora tiene **más responsabilidad** pero también **más contexto** para tomar decisiones informadas.

**PRÓXIMOS PASOS SUGERIDOS:**
1. Entrenar con datos de al menos 6 meses
2. Backtesting exhaustivo en diferentes condiciones de mercado
3. Forward testing en demo por 1-2 meses
4. Ajustar ATR multipliers según resultados
5. Considerar ensemble de modelos (múltiples timeframes)

---

**Versión:** 7.0.2
**Fecha:** 2025
**Autor:** SGRADT Team
