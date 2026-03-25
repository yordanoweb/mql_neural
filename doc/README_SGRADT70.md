# SGRADT 7.0 - Simplificación Máxima

## Resumen de Modificaciones

**SGRADT 7.0** es una versión radicalmente simplificada que elimina el **EMA completamente** y reduce las características a solo **5 features esenciales**. Esta versión mantiene la arquitectura 100% dirigida por red neuronal, donde el modelo ONNX toma TODAS las decisiones de trading basándose únicamente en **ADX + Stochastic + estructura de precios**.

### 🔄 Cambios Principales

| Aspecto | Descripción |
|---------|------------|
| **Número de Features** | 5 (máxima simplificación) |
| **EMA** | ❌ Completamente removido |
| **Indicadores** | ADX + Stochastic solamente |
| **Features** | body, range, stoch_k, stoch_d, adx |
| **Magic Number** | 7070 |
| **Entry Logic** | ADX > limit + Stochastic crossover |
| **Exit Logic** | Fixed bar horizon (future bars) |

**Filosofía:** Machine learning puro. Sin EMA, sin filtros manuales. Solo indicadores de momentum y fuerza direccional. La NN aprende patrones directamente de ADX y Stochastic en una ventana histórica.

---

## 📊 CAMBIOS EN EL SCRIPT DE ENTRENAMIENTO (Python)

### Archivo: `train_sgradt70_strategy.py`

#### 1. **Features Simplificadas: Solo 5 Features Esenciales**

**SGRADT 7.0 (5 features - máxima simplificación):**
- `feat_body` = close - open (cuerpo de la vela)
- `feat_range` = high - low (rango de la vela)
- `feat_stoch_main` (Stochastic %K)
- `feat_stoch_signal` (Stochastic %D)
- `feat_adx` (Average Directional Index)

**Comparación:**
- v2: 7 features (incluía PDI, MDI, EMA Gate, Volume Gate)
- v3: 6 features (eliminaba EMA Gate)
- **v7.0: 5 features (eliminados PDI, MDI, Volume Gate; solo lo esencial)**

#### 2. **Eliminación Completa de EMA**

```python
# ❌ NO SE IMPORTA
from ta.trend import EMAIndicator

# ❌ NO SE CALCULA
ema_inst = EMAIndicator(close=df['close'], window=args.ema_period)
df['ema9'] = ema_inst.ema_indicator()

# ✅ SE USA SOLO ADX + STOCHASTIC
adx_inst = ADXIndicator(high=df['high'], low=df['low'], close=df['close'], window=args.adx_period)
df['adx'] = adx_inst.adx()

stoch = StochasticOscillator(high=df['high'], low=df['low'], close=df['close'], 
                              window=args.stoch_k, smooth_window=args.stoch_d)
df['stoch_k'] = stoch.stoch()
df['stoch_d'] = stoch.stoch_signal()
```

#### 3. **Input Shape Actualizado**

El modelo ahora recibe `5 features × window_size` elementos:

```python
num_features_per_bar = 5  # Antes era 7, luego 6
num_inputs = num_features_per_bar * args.window
# Ejemplo con window=20: 5 × 20 = 100 inputs
```

#### 4. **Entry Logic Puro (Sin Filtros)**

```python
# Entry signals basados SOLO en ADX + Stochastic
adx_strong = df['adx'].iloc[i] > args.adx_limit

# BUY: ADX fuerte + Stochastic oversold cruzando arriba
stoch_buy = ((stoch_k_1 < stoch_d_1) and (stoch_k_0 > stoch_d_0) and (stoch_k_0 <= args.stoch_oversold)) \
            or (stoch_k_0 > stoch_k_1 + 7)
buy_signal = adx_strong and stoch_buy

# SELL: ADX fuerte + Stochastic overbought cruzando abajo
stoch_sell = ((stoch_k_1 > stoch_d_1) and (stoch_k_0 < stoch_d_0) and (stoch_k_0 >= args.stoch_overbought)) \
             or (stoch_k_0 < stoch_k_1 - 7)
sell_signal = adx_strong and stoch_sell
```

#### 5. **Exit Logic: Fixed Bar Horizon**

```python
# No gestión dinámmica de stops. La NN aprende si la señal llega a profit
# dentro de 'future' barras (parámetro: --future)

entry_price = df['open'].iloc[i+1]

if buy_signal:
    for j in range(i+1, i + args.future + 1):
        max_reach = df['high'].iloc[j]
        profit = (max_reach - entry_price) / df['close'].iloc[i] * 10000
        if profit >= args.min_profit_points:
            labels[i] = 1  # BUY label
            break

elif sell_signal:
    for j in range(i+1, i + args.future + 1):
        min_reach = df['low'].iloc[j]
        profit = (entry_price - min_reach) / df['close'].iloc[i] * 10000
        if profit >= args.min_profit_points:
            labels[i] = 2  # SELL label
            break
```

#### 6. **Metadata Actualizada**

```json
{
  "version": "SGRADT 7.0",
  "strategy": "Machine Learning + ADX + Stochastic",
  "features_per_bar": 5,
  "feature_descriptions": {
    "feat_body": "Close - Open (candlestick body)",
    "feat_range": "High - Low (candlestick range)",
    "feat_stoch_main": "Stochastic %K",
    "feat_stoch_signal": "Stochastic %D",
    "feat_adx": "Average Directional Index"
  },
  "notes": [
    "ML-driven strategy: model makes ALL decisions",
    "5 features only (no EMA, no PDI, no MDI, no volume gate)",
    "Entry: ADX > limit + Stochastic crossover",
    "Exit: Fixed bar horizon (profit-based labeling)",
    "Window: 20 bars lookback (example)",
    "No manual filters or gates"
  ]
}
```

---

## 🤖 CAMBIOS EN EL EA DE MT5

### Archivo: `EA_SGRADT70_ONNX.ex5` (o .mq5 source)

#### 1. **Parámetros de Entrada Simplificados**

```mql5
// SGRADT 7.0 - Solo lo esencial
input string InpModelName = "model_SGRADT70.onnx";
input int    InpFeaturesPerBar = 5;    // 5 features solamente
input int    InpWindow = 20;           // 20 barras de lookback
input int    InpMagic = 7070;
input int    InpADXPeriod = 8;
input int    InpStochK = 7;
input int    InpStochD = 3;
input int    InpADXLimit = 25.0;

// ❌ Removidos de versiones anteriores:
// - InpEMAPeriod (ya no existe EMA)
// - InpVolumeWindow (ya no existe Volume Gate)
// - Cualquier parámetro de "gate" manual
```

#### 2. **Indicadores Necesarios**

```mql5
// SGRADT 7.0 SOLO REQUIERE:

int g_adx_handle = INVALID_HANDLE;      // ADX
int g_stoch_handle = INVALID_HANDLE;    // Stochastic
int g_atr_handle = INVALID_HANDLE;      // Para SL/TP (opcional pero recomendado)

// ❌ NO REQUIERE:
// - EMA indicator
// - Volume indicator
// - Cualquier otro indicador
```

#### 3. **PrepareInput() Simplificado**

La función ahora prepara EXACTAMENTE **5 features** por barra:

```mql5
// SGRADT 7.0: 5 features solamente
bool PrepareInput(double& input_buffer[], int window)
{
    // Obtener datos históricos
    double open_b[],    close_b[],   high_b[],   low_b[];
    double adx_b[],     stoch_k_b[], stoch_d_b[];
    
    if(!GetIndicators(open_b, close_b, high_b, low_b, adx_b, stoch_k_b, stoch_d_b, window))
        return false;
    
    // Preparar features
    for(int i = 0; i < window; i++)
    {
        int offset = i * 5;  // 5 features por barra
        
        // Feature 0: Body (close - open)
        input_buffer[offset + 0] = (float)(close_b[i] - open_b[i]);
        
        // Feature 1: Range (high - low)
        input_buffer[offset + 1] = (float)(high_b[i] - low_b[i]);
        
        // Feature 2: Stochastic %K
        input_buffer[offset + 2] = (float)stoch_k_b[i];
        
        // Feature 3: Stochastic %D
        input_buffer[offset + 3] = (float)stoch_d_b[i];
        
        // Feature 4: ADX
        input_buffer[offset + 4] = (float)adx_b[i];
        
        // ✅ FIN. Solo 5 features. Sin EMA, sin PDI, sin MDI, sin volume gate.
    }
    
    return true;
}
```

#### 4. **OnTick() Logic**

```mql5
void OnTick()
{
    // 1. Verificar si es nueva barra
    if(!IsNewBar())
        return;
    
    // 2. Preparar inputs para la NN
    double input_buffer[100];  // 5 features × 20 window = 100 elements
    if(!PrepareInput(input_buffer, InpWindow))
        return;
    
    // 3. Run ONNX model
    double output[3];  // [HOLD, BUY, SELL]
    if(!OnnxRun(g_onnx_handle, input_buffer, output))
        return;
    
    // 4. Obtener predicción (índice de confianza máxima)
    int prediction = ArrayMaximum(output);
    double confidence = output[prediction];
    
    // 5. Filtro de confianza mínima
    if(confidence < InpMinConf)
        return;
    
    // 6. Ejecutar trade según predicción
    if(prediction == 1)  // BUY
        OpenBuy();
    else if(prediction == 2)  // SELL
        OpenSell();
    
    // 0 = HOLD (no hacer nada)
}
```

#### 5. **Panel de Información Simplificado**

```mql5
// No hay EMA gate, no hay volume gate. Solo lo esencial.
void UpdatePanel()
{
    string panel = "═══════════════════════════════\n";
    panel += "SGRADT 7.0 Status\n";
    panel += "═══════════════════════════════\n";
    panel += StringFormat("ADX: %.2f (limit: %.0f)\n", adx[0], InpADXLimit);
    panel += StringFormat("Stoch %%K: %.2f\n", stoch_k[0]);
    panel += StringFormat("Stoch %%D: %.2f\n", stoch_d[0]);
    panel += StringFormat("Body: %.5f\n", close_current - open_current);
    panel += StringFormat("Range: %.5f\n", high_current - low_current);
    panel += "═══════════════════════════════\n";
    
    Comment(panel);
}

---

## 📈 FLUJO COMPLETO DE LA ESTRATEGIA

### Entrenamiento (Python):

1. **Carga de datos** → CSV con columnas: open, high, low, close, tick_volume
2. **Cálculo de indicadores** → ADX (period=8), Stochastic (%K=7, %D=3)
3. **Generación de features** → 5 features: body, range, stoch_k, stoch_d, adx
4. **Labeling forward-looking** → Busca si precio alcanza profit_target en 'future' barras
5. **Entrenamiento Random Forest** → Aprende patrones de las 5 features usando ventana de 20 barras
6. **Export ONNX** → Modelo listo para MT5

### Ejecución en MT5:

1. **Inicialización** → Carga modelo ONNX + indicadores (ADX, Stochastic)
2. **OnTick()** → Cada nueva barra
3. **PrepareInput()** → Calcula 5 features para ventana de lookback
4. **OnnxRun()** → Red neuronal predice: HOLD(0), BUY(1), o SELL(2)
5. **Verificación de confianza** → Si probabilidad >= InpMinConf
6. **Ejecución de trade** → 
   - BUY/SELL según decisión de la NN
   - SL/TP basado en ATR
   - Sin filtros manuales

---

## 🎯 VENTAJAS DE SGRADT 7.0 SIMPLIFICADO

### 1. **Máxima Simplicidad**
- ✅ Solo 5 features esenciales
- ✅ Modelo ultra-lightweight
- ✅ Mínimo riesgo de overfitting
- ✅ Entrenamiento muy rápido
- ✅ Inferencia ultra-rápida en MT5

### 2. **Sin EMA = Mayor Flexibilidad**
- ✅ No depende de un indicador específico
- ✅ La NN aprende directamente de ADX + Stochastic
- ✅ Funciona en diferentes timeframes sin ajuste de EMA
- ✅ Más adaptable a cambios de mercado

### 3. **Menos Dependencias**
- ✅ Solo 2 indicadores requeridos: ADX + Stochastic
- ✅ Sin manejo de volume
- ✅ Sin gates manuales
- ✅ Máxima pureza en el machine learning

### 4. **Features Puras y Directas**
- **Body** y **Range**: Estructura de precio bruto
- **Stochastic**: Momentum oscilador
- **ADX**: Fuerza y dirección de tendencia
- Cada feature tiene un propósito específico, sin redundancias

### 5. **Entrada y Salida Claras**
- **Entrada**: ADX fuerte + Stochastic en extremos
- **Salida**: Fixed bar horizon (la NN predice si llegará al target)
- **Etiquetado**: Binario y determinístico (alcanzó profit o no)
- ADX/DI+/DI-: Fuerza y dirección de tendencia
- Volume Gate: Contexto de participación del mercado

La NN no está sesgada por un filtro de tendencia específico (EMA) - aprende patrones de tendencia de forma más general.

---

## 📋 CHECKLIST DE USO

### Para Entrenar:

```bash
python train_sgradt70_strategy.py \
    --csv data/EURUSD_M5.csv \
    --output ./onnx \
    --window 20 \
    --min_profit_points 20.0 \
    --future 50 \
    --stoch_k 7 \
    --stoch_d 3 \
    --adx_period 8 \
    --adx_limit 25.0 \
    --n_iter 7
```

**IMPORTANTE:** 
- El CSV debe tener columnas: open, high, low, close, tick_volume
- NO hay parámetro `--ema_period` (completamente removido)
- NO hay parámetro `--min_profit_ratio` (removido en v7.0)

### Para MT5:

1. Copiar `EURUSD_M5_SGRADT70.onnx` a `MQL5/Files/`
2. Compilar `EA_SGRADT70_ONNX.mq5` (o usar el .ex5 compilado)
3. Configurar parámetros:
   - `InpFeaturesPerBar = 5` (CRÍTICO - solo 5 features)
   - `InpWindow = 20` (debe coincidir con entrenamiento)
   - `InpMagic = 7070` (magic number para v7.0)
   - `InpADXPeriod = 8` (coincidir con entrenamiento)
   - `InpStochK = 7` (coincidir con entrenamiento)
   - `InpStochD = 3` (coincidir con entrenamiento)
   - `InpATRMultiplierSL = 2.0` (ajustar según volatilidad)
   - `InpATRMultiplierTP = 3.0` (ajustar según ratio R:R deseado)
   - `InpMinConf = 0.55` (ajustar según precisión del modelo)

**NO configurar ninguno de estos (ya no existen):**
- `InpEMAPeriod`
- `InpVolumeWindow`
- Cualquier parámetro de "gate"

---

## ⚠️ TABLA COMPARATIVA: Evolución del Modelo

| Aspecto | v1 | v2 | v3 | **v7.0** |
|---------|----|----|----|----|
| **Features** | 5 base | 7 (pdi, mdi, ema_gate, volume_gate) | 6 (sin ema_gate) | **5 esenciales (body, range, stoch_k, stoch_d, adx)** |
| **Indicadores** | ADX, Stoch | ADX, DI+/-, Stoch, EMA | ADX, DI+/-, Stoch | **ADX, Stoch solamente** |
| **EMA** | Sí (filtro externo) | Sí (feature) | No | **No** |
| **PDI / MDI** | Sí | Sí | Sí | **No** |
| **Volume Gate** | No | Sí | Sí | **No** |
| **Decisión** | NN + Filtros manuales | 100% NN | 100% NN | **100% NN (puro ML)** |
| **Complexity** | Baja | Alta | Media | **Muy baja (máxima simplicidad)** |
| **Input Shape** | [1, 100] (5×20) | [1, 140] (7×20) | [1, 120] (6×20) | **[1, 100] (5×20)** |
| **Magic** | 7070 | 7072 | 7073 | **7070** |
| **Filosofía** | Híbrida | NN con contexto total | NN con momentum+fuerza | **ML minimalista** |

**Conclusión:** v7.0 = máxima simplicidad, máxima pureza en ML, mínimo overfitting.

---

## 🎓 CUÁNDO USAR v7.0 vs v2 vs v3

### Usar v7.0 si:
- ✅ Quieres el modelo MÁS SIMPLE y RÁPIDO
- ✅ Buscas máxima generalización entre mercados
- ✅ Prefieres features "puras" sin redundancias
- ✅ Tienes datos limitados (menos features = menos datos necesarios)
- ✅ Quieres menor riesgo de overfitting
- ✅ Buscas máxima pureza en machine learning (sin gates, sin filtros)

### Usar v2 o v3 si:
- ✅ Tus datos históricos muestran mejor rendimiento con más features
- ✅ Necesitas contexto explícito de tendencia (PDI/MDI)
- ✅ El mercado responde bien a volume-gate filtering
- ✅ Prefieres más contexto antes de simplificar

---

## 📊 LABELING EN v7.0: Fixed Bar Horizon

La lógica de labeling en v7.0 es **más simple y directa**:

```python
# Entry conditions (ADX + Stochastic only)
adx_strong = df['adx'].iloc[i] > args.adx_limit

# BUY: ADX fuerte + Stochastic oversold cruzando arriba
stoch_buy = ((stoch_k_1 < stoch_d_1) and (stoch_k_0 > stoch_d_0) and (stoch_k_0 <= args.stoch_oversold)) \
            or (stoch_k_0 > stoch_k_1 + 7)
buy_signal = adx_strong and stoch_buy

# SELL: ADX fuerte + Stochastic overbought cruzando abajo
stoch_sell = ((stoch_k_1 > stoch_d_1) and (stoch_k_0 < stoch_d_0) and (stoch_k_0 >= args.stoch_overbought)) \
             or (stoch_k_0 < stoch_k_1 - 7)
sell_signal = adx_strong and stoch_sell

# Exit: ¿Alcanzó el profit target dentro de 'future' barras?
if buy_signal:
    for j in range(i+1, i + args.future + 1):
        max_reach = df['high'].iloc[j]
        profit = (max_reach - entry_price) / df['close'].iloc[i] * 10000
        if profit >= args.min_profit_points:
            labels[i] = 1  # BUY label
            break

elif sell_signal:
    for j in range(i+1, i + args.future + 1):
        min_reach = df['low'].iloc[j]
        profit = (entry_price - min_reach) / df['close'].iloc[i] * 10000
        if profit >= args.min_profit_points:
            labels[i] = 2  # SELL label
            break

# else: labels[i] = 0 (HOLD)
```

**Diferencia clave con v2/v3:**
- ❌ v2/v3: Usaban `min_profit_ratio` para validar balance entre up/down moves
- ✅ v7.0: Solo busca si alcanza `min_profit_points` en ventana futura (más simple)

### 🎯 Parámetros Recomendados para v7.0

| Instrumento | min_profit_points | future | adx_limit | Notas |
|-------------|-------------------|--------|-----------|-------|
| **NAS100/S&P500** | 40-60 | 20-30 | 25 | Volatilidad media-alta |
| **Forex Majors** | 15-25 | 30-50 | 25 | Menos volátil, requiere ventana mayor |
| **BTC/ETH** | 60-100 | 15-25 | 20 | Alta volatilidad, profit target grande |
| **GBPUSD** | 20-30 | 40-60 | 25 | Muy volátil, necesita ajuste fino |
| **FTSE/DAX** | 25-40 | 25-40 | 25 | Media volatilidad |

**Recomendación:** Comienza con `--min_profit_points 20 --future 50 --adx_limit 25` y ajusta según tus resultados.

---

## 🔧 TROUBLESHOOTING

### Error: "Input buffer size incorrect"
**Solución:** Verificar que `InpFeaturesPerBar = 5` en MT5 (5 features solamente en v7.0).

### Error: "Model expects 100 inputs but got X"
**Solución:** El mismatch ocurre cuando window no coincide:
- v7.0 usa: `5 features × window_size = total inputs`
- Ejemplo: 5 × 20 = 100 inputs
- Verifica que `--window` en entrenamiento = `InpWindow` en MT5

### Error: "Cannot find ADX/Stochastic indicator"
**Solución:** v7.0 SOLO requiere ADX y Stochastic. Asegúrate de que:
1. Los handles se crean correctamente en OnInit()
2. Los indicadores tienen suficientes barras históricas
3. Los parámetros coinciden (ADX period=8, Stoch K=7, D=3)

### Modelo entrenado en v2/v3 no funciona en v7.0
**Solución:** Los modelos NO son compatibles entre versiones. Debes:
1. Re-entrenar con `train_sgradt70_strategy.py` (v7.0)
2. Usar el nuevo modelo `.onnx` generado
3. Configurar `InpFeaturesPerBar = 5` en MT5

### Modelo no genera suficientes señales
**Posibles causas y soluciones:**
1. `--min_profit_points` muy alto → reducir de 20 a 15
2. `--future` muy pequeño → aumentar de 50 a 70
3. `--adx_limit` muy alto → reducir de 25 a 20
4. Datos insuficientes en entrenamiento (< 6 meses)
5. Stochastic threshold muy restrictivo → ajustar `--stoch_oversold` a 25 en lugar de 20

### Modelo en backtest es excelente pero en forward test es malo
**Problema:** Overfitting probable.
**Soluciones:**
1. Reduce `--n_iter` en RandomizedSearchCV (menos búsqueda = menos overfitting)
2. Usa más datos de entrenamiento (mínimo 9 meses)
3. Prueba con `--min_profit_points` 5-10 puntos MAYOR (requiere signals más fuertes)
4. Aumenta `--future` para ser más conservador

### "Warnings: Balanced Accuracy muy baja (< 0.55)"
**Significado:** El modelo no aprende bien los patrones.
**Soluciones:**
1. Los parámetros de entrada no son óptimos → prueba diferentes combinations
2. Los datos son muy ruidosos → filtra datos de baja volatilidad/volumen
3. El mercado es muy aleatorio en ese período → prueba otro timeframe
4. Necesitas más datos de entrenamiento

---

## 📊 CARACTERÍSTICAS DE v7.0 vs VERSIONES ANTERIORES

**v7.0: Máxima Simplicidad**
- ✅ 5 features solamente (no hay redundancias)
- ✅ Solo ADX + Stochastic (dos indicadores puros)
- ✅ Exit logic simple: fixed bar horizon
- ✅ Mínimo riesgo de overfitting
- ✅ Entrenamiento ultra-rápido
- ✅ Inferencia ultra-rápida en MT5

**Ventajas de v7.0:**
- Generaliza mejor (menos features = menos específico al período de entrenamiento)
- Requiere menos datos para entrenar (5 features vs 7)
- Más estable en diferentes condiciones de mercado
- Código más mantenible (sin EMA, sin PDI/MDI, sin volume gate)

**Comparativa rápida:**
| Métrica | v2 | v3 | **v7.0** |
|---------|----|----|---------|
| Features | 7 | 6 | **5** |
| Indicadores | 4+ | 3+ | **2** |
| Input Shape (window=20) | 140 | 120 | **100** |
| Velocidad Entrenamiento | Media | Rápida | **Muy rápida** |
| Riesgo Overfitting | Alto | Medio | **Bajo** |
| Generalización | Media | Buena | **Excelente** |

---

## 🔄 MIGRACIÓN DE v3 o v2 a v7.0

### Script Python:

1. Descargar `train_sgradt70_strategy.py` (v7.0)
2. Ejecutar con parámetros simplificados:
   ```bash
   python train_sgradt70_strategy.py \
       --csv data.csv \
       --window 20 \
       --min_profit_points 20 \
       --future 50 \
       --adx_period 8 \
       --stoch_k 7 \
       --stoch_d 3 \
       --adx_limit 25 \
       --n_iter 7
   ```

3. **Parámetros eliminados (ya no existen):**
   - `--ema_period`
   - `--min_profit_ratio`
   - Cualquier parámetro de volume gate

### EA de MT5:

1. Descargar `EA_SGRADT70_ONNX.mq5` (v7.0)
2. Copiar modelo `_SGRADT70.onnx` a `MQL5/Files/`
3. Configurar parámetros:
   - `InpModelName` → archivo `_SGRADT70.onnx`
   - `InpFeaturesPerBar = 5` (CRÍTICO)
   - `InpWindow = 20` (debe coincidir con --window en entrenamiento)
   - `InpMagic = 7070`
   - `InpADXPeriod = 8`
   - `InpStochK = 7`
   - `InpStochD = 3`
4. Compilar y verificar sin errores

**IMPORTANTE:** Los modelos NO son compatibles entre versiones. Debes:
- Entrenar nuevo modelo con v7.0
- Usar el nuevo `.onnx` generado
- Configurar EA v7.0 correctamente

---

## 📝 EJEMPLO DE USO COMPLETO

### 1. Entrenamiento

```bash
# Entrenar modelo v7.0
python train_sgradt70_strategy.py \
    --csv data/EURUSD_M5.csv \
    --output ./onnx \
    --window 20 \
    --min_profit_points 20.0 \
    --future 50 \
    --stoch_k 7 \
    --stoch_d 3 \
    --adx_period 8 \
    --adx_limit 25.0 \
    --n_iter 7
```

**Output esperado:**
```
[INFO] Starting SGRADT 7.0 training
======================================================================
[INFO] Arguments: {'csv': ['data/EURUSD_M5.csv'], 'output': './onnx', ...}
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
InpModelName = "EURUSD_M5_SGRADT70.onnx"
InpMinConf   = 0.55
InpWindow = 20
InpFeaturesPerBar = 5  ← CRÍTICO: 5 features (v7.0 simplificado)

======== INFERENCE ========
InpInferSeconds = 0
InpOneTradePerBar = true

======== SESSION ========
InpStartHour = 0
InpEndHour   = 24

======== INDICATORS ========
InpADXPeriod = 8
InpStochK = 7
InpStochD = 3
InpADXLimit = 25.0

======== RISK (ATR-BASED) ========
InpLot = 0.1
InpMagic = 7070
InpATRPeriod = 14
InpATRMultiplierSL = 2.0
InpATRMultiplierTP = 3.0
```

### 3. Verificación en Logs de MT5

```
----------------------------------------------------------------------
    SGRADT 7.0 - NN-DRIVEN STRATEGY (5 Features - Pure ML)
    ADX + Stochastic Only - No EMA, No PDI/MDI, No Volume Gate
----------------------------------------------------------------------

Loading ONNX model: EURUSD_M5_SGRADT70.onnx
[OK] ONNX model loaded successfully
[OK] Input shape set: [1, 100] (20 bars x 5 features)

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
ADX: Period=8, Limit=25.0
ATR: Period=14

=== STRATEGY ===
Type: NN-Driven (Neural Network decides everything)
Features: 5 (body, range, stoch_k, stoch_d, adx)
Exit: Fixed bar horizon (labeled during training)
Entry: ADX > limit + Stochastic crossover
No EMA - removed in v7.0
No PDI/MDI - removed in v7.0
No Volume Gate - removed in v7.0
Pure machine learning - only essential indicators

Inference mode: New bar only

----------------------------------------------------------------------
    EA INITIALIZED SUCCESSFULLY
----------------------------------------------------------------------
```

---

## 📊 ESTRUCTURA DE ARCHIVOS

```
project/
├── train_sgradt70_strategy.py     ← Script de entrenamiento (simplificado)
├── EA_SGRADT70_ONNX.ex5           ← Expert Advisor compilado para MT5
├── README_SGRADT70.md             ← Esta documentación
│
├── data/
│   └── EURUSD_M5.csv              ← Datos de entrenamiento (open, high, low, close, tick_volume)
│
├── onnx/
│   ├── EURUSD_M5_SGRADT70.onnx         ← Modelo entrenado
│   └── EURUSD_M5_SGRADT70.meta.json    ← Metadata
│
└── MQL5/Files/  (en MT5)
    └── EURUSD_M5_SGRADT70.onnx ← Copiar aquí para usar en MT5
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

## 🚀 OPTIMIZACIÓN Y EXTENSIÓN

Dado que SGRADT 7.0 es maximalista en simplicidad, las opciones de optimización son limitadas:

### 1. Ajuste de Parámetros del Indicador

```bash
# Probar diferentes períodos de ADX y Stochastic
for ADX in 6 8 10 12; do
  for STOCH_K in 5 7 9 11; do
    python train_sgradt70_strategy.py \
      --csv data.csv \
      --adx_period $ADX \
      --stoch_k $STOCH_K \
      --output ./onnx/test_adx${ADX}_stoch${STOCH_K}/
  done
done
```

### 2. Ajuste de Parámetros de Labeling

```bash
# Probar diferentes profit targets y horizontes
for MIN_PROFIT in 15 20 25 30; do
  for FUTURE in 30 50 70; do
    python train_sgradt70_strategy.py \
      --csv data.csv \
      --min_profit_points $MIN_PROFIT \
      --future $FUTURE \
      --output ./onnx/test_profit${MIN_PROFIT}_future${FUTURE}/
  done
done
```

Comparar balanced_accuracy y seleccionar la mejor combinación.

### 3. Ensemble Multi-Timeframe

Entrenar modelos separados para diferentes timeframes:

```bash
python train_sgradt70_strategy.py --csv EURUSD_M5.csv  --window 20
python train_sgradt70_strategy.py --csv EURUSD_M15.csv --window 15
python train_sgradt70_strategy.py --csv EURUSD_H1.csv  --window 10
```

En MT5: usar 3 EAs diferentes (uno por timeframe) con diferentes magic numbers.

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
| **Gate** | Filtro o feature de contexto (removido en v7.0 para máxima simplicidad) |
| **ONNX** | Formato de modelo neural network para MT5 |
| **Inference** | Proceso de predicción usando el modelo |

---

## ❓ FAQ - Preguntas Frecuentes

### P: ¿Por qué solo 5 features?

**R:** Máxima simplicidad = máxima generalización. Cada feature tiene un propósito claro:
- Body + Range: Estructura de precio
- Stoch K + D: Momentum
- ADX: Fuerza de tendencia

Menos features = menos overfitting, más velocidad, mayor robustez.

### P: ¿Cómo puede funcionar sin EMA?

**R:** ADX ya mide fuerza y dirección de tendencia. Stochastic mide momentum. Juntos, proporcionan toda la información que un EMA pediría. La NN aprende a combinarlos directamente.

### P: ¿Puedo usar un modelo v3 en la versión 7.0?

**R:** NO. Los input shapes son diferentes (6 features vs 5 features). Debes entrenar un modelo 7.0 nuevo.

### P: ¿Cómo sé si mi modelo está en overfitting?

**R:** Señales de overfitting:
- Balanced accuracy > 0.85 en entrenamiento pero mucho peor en backtest
- Rendimiento perfecto en el período de entrenamiento pero malo en periodos nuevos
- Modelo funciona SOLO en el timeframe/par específico de entrenamiento

### P: ¿Cuántos datos necesito para entrenar?

**R:** Mínimo recomendado:
- **Datos de entrenamiento:** 6 meses de datos M5 (~50,000 barras)
- **Datos de validación/test:** Otros 3 meses (~25,000 barras)

Total: ~9 meses de datos históricos (mínimo).

### P: ¿Qué hacer si el modelo no genera suficientes señales?

**R:** Posibles soluciones:
1. Reducir `--min_profit_points` (ej: 20 → 15)
2. Aumentar `--future` (ej: 50 → 70)
3. Reducir `--adx_limit` (ej: 25 → 20)
4. Reducir umbral de Stochastic

### P: ¿Puedo usar el modelo en múltiples pares simultáneamente?

**R:** Sí, pero entrena un modelo SEPARADO para cada par:
- EURUSD → modelo específico para EURUSD
- GBPUSD → modelo específico para GBPUSD
- USDJPY → modelo específico para USDJPY

Cada instrumento tiene patrones únicos que el modelo debe aprender.

### P: ¿Cuál es la diferencia entre window y future?

**R:** 
- **window**: Cuántas barras históricas ve el modelo para hacer la predicción (ej: 20 barras)
- **future**: Cuántas barras futuras se buscan para encontrar el profit target (ej: 50 barras)

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

### v7.0.0 (2025) - Simplificación Máxima
- ✅ **NUEVO:** Reducción a 5 features (eliminados PDI, MDI, Volume Gate, EMA Gate)
- ✅ **NUEVO:** Entry logic basada SOLO en ADX + Stochastic
- ✅ **NUEVO:** Exit logic basada en fixed bar horizon
- ✅ **NUEVO:** Magic number 7070
- ❌ **REMOVIDO:** EMA indicator y EMA gate
- ❌ **REMOVIDO:** PDI y MDI indicators
- ❌ **REMOVIDO:** Volume Gate feature
- ❌ **REMOVIDO:** Parámetro `--ema_period`
- ❌ **REMOVIDO:** Parámetro `InpEMAPeriod`
- ❌ **REMOVIDO:** Parámetro `--min_profit_ratio`
- 🔧 **CAMBIADO:** `InpFeaturesPerBar` de 6 a 5
- 🔧 **CAMBIADO:** Input shape de [1, 120] a [1, 100] (con window=20)
- 🔧 **CAMBIADO:** Script name de `train_sgradt70_strategy_v3.py` a `train_sgradt70_strategy.py`

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

**SGRADT 7.0** representa el máximo nivel de simplificación:

**Filosofía:**
- **Purismo en Machine Learning:** Sin ruido, sin filtros manuales
- **5 features > cualquier combinación más compleja:** Cada feature = propósito específico
- **ADX + Stochastic:** La combinación mínima viable para trading
- **Eficiencia máxima:** Entrenamiento e inferencia ultra-rápida
- **Robustez:** Menos features = menos overfitting

**Comparación de enfoques:**
- **v1:** Estrategia híbrida (NN + reglas manuales)
- **v2:** NN pura pero con 7 features (PDI, MDI, EMA Gate, Volume Gate)
- **v3:** NN pura con 6 features (sin EMA Gate)
- **v7.0:** NN pura minimalista con 5 features (esencial = máxima potencia)

**PRÓXIMOS PASOS SUGERIDOS:**
1. Entrenar modelo con tus datos usando `train_sgradt70_strategy.py`
2. Validar balanced_accuracy >= 0.60 (mínimo aceptable)
3. Backtest exhaustivo en MT5 (mínimo 3 meses out-of-sample)
4. Forward test en demo por 2-4 semanas
5. Si resultados OK, empezar en vivo con lotes pequeños

**El mejor modelo es el que funciona mejor en TU mercado con TUS datos.**

---

**Versión:** 7.0.0  
**Fecha:** Marzo 2025  
**Estrategia:** Machine Learning + ADX + Stochastic  
**Features:** 5 (body, range, stoch_k, stoch_d, adx)  
**Indicadores:** ADX, Stochastic  
**Licencia:** Educational Use Only

---

*"Everything should be made as simple as possible, but not simpler." - Albert Einstein*
