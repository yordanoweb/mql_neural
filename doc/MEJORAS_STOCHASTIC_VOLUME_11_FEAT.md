# MEJORAS EN CARACTERÍSTICAS STOCHASTIC Y VOLUME

## 📊 COMPARACIÓN: VERSIÓN SIMPLE VS MEJORADA

### ❌ VERSIÓN ORIGINAL (Demasiado Simple)

#### Stochastic (1 característica)
```python
# Solo la diferencia entre K y D
feat_stoch = (stoch.stoch() - stoch.stoch_signal()) / 100.0
```
**Problemas:**
- Solo captura UN aspecto del estocástico
- No detecta zonas de sobrecompra/sobreventa
- No mide la velocidad del cambio
- Pierde información crítica sobre reversiones

#### Volume (1 característica)
```python
# Solo el ratio vs media móvil
feat_vol = tick_volume / tick_volume.rolling(window=20).mean()
```
**Problemas:**
- Ignora la tendencia del volumen
- No detecta divergencias precio-volumen
- No mide si el volumen es anormalmente alto/bajo
- No captura acumulación/distribución

---

### ✅ VERSIÓN MEJORADA (Robusta y Completa)

## 1️⃣ STOCHASTIC - 4 CARACTERÍSTICAS AVANZADAS

### Feature 1: **Momentum** (K - D normalizado)
```python
feat_stoch_momentum = (stoch_k - stoch_d) / 100.0
```
- **Rango:** [-1, 1]
- **Propósito:** Detecta cruce de líneas y fuerza del movimiento
- **Señal:** Positivo = bullish, Negativo = bearish

### Feature 2: **Position** (Posición relativa)
```python
feat_stoch_position = (stoch_k - 50.0) / 50.0
```
- **Rango:** [-1, 1]
- **Propósito:** Indica si estamos en zona alta o baja
- **Señal:** 
  - > 0.6 = Zona de sobrecompra
  - < -0.6 = Zona de sobreventa

### Feature 3: **Velocity** (Velocidad de cambio)
```python
feat_stoch_velocity = stoch_k.diff() / 100.0
```
- **Rango:** Variable
- **Propósito:** Detecta aceleración/desaceleración
- **Señal:** 
  - Positivo = Acelerando hacia arriba
  - Negativo = Acelerando hacia abajo

### Feature 4: **Divergence Pressure** (Presión de reversión)
```python
# Zonas de presión
overbought_pressure = np.where(stoch_k > 80, -(stoch_k - 80) / 20.0, 0)
oversold_pressure = np.where(stoch_k < 20, (20 - stoch_k) / 20.0, 0)
feat_stoch_divergence = overbought_pressure + oversold_pressure
```
- **Rango:** [-1, 1]
- **Propósito:** Detecta presión de reversión en extremos
- **Señal:**
  - Negativo en K>80 = Presión bajista
  - Positivo en K<20 = Presión alcista

---

## 2️⃣ VOLUME - 5 CARACTERÍSTICAS AVANZADAS

### Feature 1: **Ratio** (Volumen vs Media)
```python
feat_vol_ratio = tick_volume / vol_ma
```
- **Rango:** > 0
- **Propósito:** Volumen relativo actual
- **Señal:** >1.5 = Alto volumen, <0.5 = Bajo volumen

### Feature 2: **Momentum** (Tendencia del volumen)
```python
vol_ema_fast = tick_volume.ewm(span=5).mean()
vol_ema_slow = tick_volume.ewm(span=20).mean()
feat_vol_momentum = (vol_ema_fast - vol_ema_slow) / vol_ema_slow
```
- **Rango:** Variable
- **Propósito:** Detecta si el volumen está aumentando o disminuyendo
- **Señal:** 
  - Positivo = Volumen creciente (confirmación de tendencia)
  - Negativo = Volumen decreciente (debilitamiento)

### Feature 3: **Price Divergence** (Divergencia Precio-Volumen)
```python
price_change = close.pct_change().abs()
vol_change = tick_volume.pct_change().abs()
feat_vol_price_div = vol_change - price_change
```
- **Rango:** Variable
- **Propósito:** Detecta acumulación/distribución
- **Señal:**
  - Positivo alto = Volumen sube más que precio (acumulación)
  - Negativo = Precio se mueve sin volumen (debilidad)

### Feature 4: **Percentile** (Percentil de volumen)
```python
feat_vol_percentile = tick_volume.rolling(20).apply(
    lambda x: pd.Series(x).rank(pct=True).iloc[-1]
)
feat_vol_percentile = (feat_vol_percentile - 0.5) * 2  # [-1, 1]
```
- **Rango:** [-1, 1]
- **Propósito:** Posición del volumen actual en la distribución reciente
- **Señal:**
  - +1 = Volumen máximo en ventana de 20
  - -1 = Volumen mínimo en ventana de 20

### Feature 5: **Z-Score** (Detección de anomalías)
```python
feat_vol_zscore = (tick_volume - vol_ma) / vol_std
feat_vol_zscore = np.clip(feat_vol_zscore, -3, 3) / 3.0  # [-1, 1]
```
- **Rango:** [-1, 1]
- **Propósito:** Detecta volumen anormalmente alto/bajo
- **Señal:**
  - +1 = 3+ desviaciones estándar por encima (anomalía alta)
  - -1 = 3+ desviaciones estándar por debajo (anomalía baja)

---

## 📈 IMPACTO EN EL MODELO

### Antes (2 características básicas + 2 simples)
```
Total features: 4
- feat_body
- feat_range
- feat_stoch (simple)
- feat_vol (simple)

Input shape: [1, 80]  (20 ventanas × 4 features)
```

### Después (2 básicas + 4 stoch + 5 vol)
```
Total features: 11
- feat_body
- feat_range
- feat_stoch_momentum
- feat_stoch_position
- feat_stoch_velocity
- feat_stoch_divergence
- feat_vol_ratio
- feat_vol_momentum
- feat_vol_price_div
- feat_vol_percentile
- feat_vol_zscore

Input shape: [1, 220]  (20 ventanas × 11 features)
```

---

## 🎯 VENTAJAS DE LA VERSIÓN MEJORADA

### 1. **Captura Multidimensional**
- Cada indicador es analizado desde múltiples ángulos
- Stochastic: momentum, posición, velocidad, presión
- Volume: ratio, tendencia, divergencia, percentil, anomalías

### 2. **Detección de Patrones Complejos**
- Divergencias precio-volumen (acumulación/distribución)
- Zonas de reversión con presión
- Anomalías estadísticas en volumen

### 3. **Mejor Contexto para el ML**
- El modelo RF puede aprender relaciones más sofisticadas
- Más información = mejor capacidad predictiva
- Features normalizadas y balanceadas

### 4. **Sincronización Python-MT5**
- Cálculos idénticos en ambos lados
- Mismo orden de features
- Misma normalización

---

## 🔧 USO

### Entrenar el Modelo Mejorado
```bash
python train_onnx_enhanced_features.py \
    --input_csv ndx100_m5_rates.csv \
    --output_dir ./models \
    --window 20 \
    --future 10 \
    --atr_period 14 \
    --min_profit_atr 1.5 \
    --stoch_window 14 \
    --vol_window 20 \
    --n_iter 10
```

### Parámetros del EA
```
FEATURES = 11  // CRÍTICO: debe coincidir con Python
InpWindow = 20
InpStochPeriod = 14  // Debe coincidir con --stoch_window
InpVolWindow = 20    // Debe coincidir con --vol_window
```

---

## ⚠️ PUNTOS CRÍTICOS DE SINCRONIZACIÓN

### 1. Orden de Features
El orden DEBE ser exactamente el mismo:
```cpp
// MQL5
input_buffer[base_idx + 0] = feat_body;
input_buffer[base_idx + 1] = feat_range;
input_buffer[base_idx + 2] = stoch_momentum;
input_buffer[base_idx + 3] = stoch_position;
input_buffer[base_idx + 4] = stoch_velocity;
input_buffer[base_idx + 5] = stoch_divergence;
input_buffer[base_idx + 6] = vol_ratio;
input_buffer[base_idx + 7] = vol_momentum;
input_buffer[base_idx + 8] = vol_price_div;
input_buffer[base_idx + 9] = vol_percentile;
input_buffer[base_idx + 10] = vol_zscore;
```

```python
# Python
features = [
    'feat_body',
    'feat_range',
    'feat_stoch_momentum',
    'feat_stoch_position',
    'feat_stoch_velocity',
    'feat_stoch_divergence',
    'feat_vol_ratio',
    'feat_vol_momentum',
    'feat_vol_price_div',
    'feat_vol_percentile',
    'feat_vol_zscore'
]
```

### 2. Parámetros de Indicadores
- Stochastic: Período 14, K=3, D=3
- Volume Window: 20 barras
- ATR: 14 períodos

### 3. Normalización
- Todas las features deben estar aproximadamente en [-1, 1] o [0, 2]
- Evita valores extremos con clipping

---

## 📊 EJEMPLO DE FEATURES CALCULADAS

Para una vela específica:
```
BASIC:
  feat_body = 0.45         (cierre 0.45 ATRs por encima del open)
  feat_range = 1.2         (rango de 1.2 ATRs)

STOCHASTIC:
  feat_stoch_momentum = 0.15   (K está 15 puntos por encima de D)
  feat_stoch_position = 0.6    (K=80, zona de sobrecompra)
  feat_stoch_velocity = -0.05  (K cayendo ligeramente)
  feat_stoch_divergence = -0.5 (presión bajista en sobrecompra)

VOLUME:
  feat_vol_ratio = 1.8         (80% más volumen que promedio)
  feat_vol_momentum = 0.3      (volumen creciente)
  feat_vol_price_div = 0.4     (volumen sube más que precio)
  feat_vol_percentile = 0.9    (volumen en percentil 95)
  feat_vol_zscore = 0.8        (2.4 desv. estándar arriba)

Interpretación: Alta volatilidad con volumen anómalo en zona de 
sobrecompra con presión bajista = Posible reversión bajista
```

---

## 🚀 MEJORAS ADICIONALES OPCIONALES

### Si quieres ir más allá:

1. **RSI Features**
   - RSI value
   - RSI velocity
   - RSI divergence

2. **MACD Features**
   - MACD histogram
   - MACD crossover
   - MACD divergence

3. **Volume Profile**
   - Volume at price levels
   - VWAP distance

4. **Time Features**
   - Hour of day
   - Day of week
   - Session (Asia/Europe/US)

---

## ✅ CHECKLIST DE IMPLEMENTACIÓN

- [ ] Entrenar modelo con script mejorado
- [ ] Verificar que genera archivo .onnx
- [ ] Copiar .onnx a MT5/Files/
- [ ] Actualizar #resource en EA
- [ ] Compilar EA mejorado
- [ ] Verificar FEATURES = 11 en EA
- [ ] Probar en Strategy Tester
- [ ] Comparar resultados con versión simple

---

## 📝 NOTAS FINALES

Esta versión mejorada proporciona al modelo ML una vista mucho más completa 
del mercado. En lugar de solo 4 features simples, ahora tiene 11 features 
que capturan:
- Precio (body, range)
- Momentum multidimensional (stochastic 4D)
- Volumen completo (5D con tendencia, divergencias, anomalías)

Esto debería mejorar significativamente la capacidad predictiva del modelo.
