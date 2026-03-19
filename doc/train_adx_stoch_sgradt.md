# SGRADT 5.0 - Script de Entrenamiento Re-adaptado

## 📋 Descripción

Este script implementa **exactamente** las condiciones de trading descritas en el documento `stoch_adx_strategy.md` para entrenar un modelo de Machine Learning que replica la lógica del robot SGRADT 5.0.

## 🔄 Diferencias principales con el script original

### Script Original (`train_price_action_adx_points.py`)
- ❌ Usaba crossover simple de Stochastic (K cruza D)
- ❌ Solo verificaba zona oversold/overbought en la barra anterior
- ❌ No implementaba momentum fuerte (+7/-7)
- ❌ ADX solo verificaba threshold simple
- ❌ No implementaba reversales de DI

### Nuevo Script (`train_sgradt_strategy.py`)
- ✅ Implementa **todas** las condiciones del markdown
- ✅ Crossover con lookback múltiple (barras 1-2 y 2-3)
- ✅ Momentum fuerte de Stochastic (+7/-7 en dos barras consecutivas)
- ✅ Pre-condición de ADX (threshold + movimiento +5)
- ✅ Condiciones de tendencia DI+ y DI- (trending y reversal)
- ✅ Tres modos de estrategia: `stoch`, `adx`, `combined`

## 🎯 Condiciones Implementadas

### Stochastic - Señales BUY

**Condición 1: Oversold Crossover**
```
Lookback 1-2:
  main[2] < signal[2] AND
  main[1] > signal[1] AND
  main[1] <= 20.0

Lookback 2-3:
  main[3] < signal[3] AND
  main[2] > signal[2] AND
  main[2] <= 20.0
```

**Condición 2: Strong Upward Momentum**
```
main[0] > main[1] + 7 AND
main[1] > main[2] + 7
```

### Stochastic - Señales SELL

**Condición 1: Overbought Crossover**
```
Lookback 1-2:
  main[2] > signal[2] AND
  main[1] < signal[1] AND
  main[1] >= 80.0

Lookback 2-3:
  main[3] > signal[3] AND
  main[2] < signal[2] AND
  main[2] >= 80.0
```

**Condición 2: Strong Downward Momentum**
```
main[0] < main[1] - 7 AND
main[1] < main[2] - 7
```

### ADX - Pre-condición (Mercado en Tendencia)

```
adx[0] > 32 OR
adx[1] > 32 OR
(adx[1] - adx[2]) > 5 OR
(adx[0] - adx[1]) > 5
```

### ADX - Señales BUY (si pre-condición cumple)

**Condición 1: DI+ trending up, DI- trending down**
```
pdi[0] > pdi[2] AND
pdi[1] > pdi[2] AND
pdi[0] > pdi[1] AND
mdi[0] < mdi[1] AND
mdi[1] < mdi[2]
```

**Condición 2: -DI Reversal**
```
mdi[2] < mdi[3] AND
mdi[1] < mdi[2] AND
mdi[0] < mdi[1] AND
pdi[0] > pdi[2]
```

### ADX - Señales SELL (si pre-condición cumple)

**Condición 1: DI- trending up, DI+ trending down**
```
mdi[0] > mdi[2] AND
mdi[1] > mdi[2] AND
mdi[0] > mdi[1] AND
pdi[0] < pdi[1] AND
pdi[1] < pdi[2]
```

**Condición 2: +DI Reversal**
```
pdi[2] < pdi[3] AND
pdi[1] < pdi[2] AND
pdi[0] < pdi[1] AND
mdi[0] > mdi[2]
```

## 🚀 Uso

### Instalación de dependencias

```bash
pip install pandas numpy scikit-learn skl2onnx ta --break-system-packages
```

### Modo 1: Estrategia Combinada (Recomendado)

Ambos indicadores deben confirmar la señal:

```bash
python train_sgradt_strategy.py \
    --csv EUR_USD_H1.csv \
    --strategy combined \
    --move_points 50 \
    --future 10 \
    --window 20 \
    --n_iter 20
```

### Modo 2: Solo Stochastic

```bash
python train_sgradt_strategy.py \
    --csv EUR_USD_H1.csv \
    --strategy stoch \
    --stoch_k 7 \
    --stoch_d 3 \
    --stoch_oversold 20 \
    --stoch_overbought 80 \
    --move_points 50 \
    --future 10
```

### Modo 3: Solo ADX

```bash
python train_sgradt_strategy.py \
    --csv EUR_USD_H1.csv \
    --strategy adx \
    --adx_period 8 \
    --adx_limit 32 \
    --move_points 50 \
    --future 10
```

## ⚙️ Parámetros

### Parámetros de Entrada/Salida

| Parámetro | Default | Descripción |
|-----------|---------|-------------|
| `--csv` | (requerido) | Archivo CSV con datos OHLC |
| `--output` | `./onnx` | Directorio de salida |
| `--window` | `20` | Barras de lookback para features |

### Parámetros de Validación

| Parámetro | Default | Descripción |
|-----------|---------|-------------|
| `--move_points` | `50.0` | Movimiento mínimo para validar señal |
| `--future` | `10` | Barras futuras para validación |

### Parámetros de Stochastic (SGRADT 5.0)

| Parámetro | Default | Descripción |
|-----------|---------|-------------|
| `--stoch_k` | `7` | Período K |
| `--stoch_d` | `3` | Período D (smoothing) |
| `--stoch_slowing` | `3` | Slowing |
| `--stoch_oversold` | `20.0` | Nivel de sobreventa |
| `--stoch_overbought` | `80.0` | Nivel de sobrecompra |

### Parámetros de ADX (SGRADT 5.0)

| Parámetro | Default | Descripción |
|-----------|---------|-------------|
| `--adx_period` | `8` | Período ADX |
| `--adx_limit` | `32.0` | Threshold de tendencia |

### Parámetros de Entrenamiento

| Parámetro | Default | Descripción |
|-----------|---------|-------------|
| `--n_iter` | `20` | Iteraciones RandomizedSearchCV |
| `--strategy` | `combined` | `stoch`, `adx` o `combined` |

## 📊 Salidas

El script genera dos archivos:

1. **Modelo ONNX**: `{nombre}_SGRADT50_{strategy}.onnx`
2. **Metadata JSON**: `{nombre}_SGRADT50_{strategy}.meta.json`

### Contenido del Metadata

```json
{
  "model_file": "EUR_USD_H1_SGRADT50_combined.onnx",
  "strategy": "combined",
  "timestamp": "2026-03-19T...",
  "window_size": 20,
  "features_per_bar": 7,
  "num_inputs": 140,
  "feature_order": [
    "feat_body",
    "feat_range",
    "feat_stoch_main",
    "feat_stoch_signal",
    "feat_adx",
    "feat_pdi",
    "feat_mdi"
  ],
  "stoch_k_period": 7,
  "stoch_d_period": 3,
  "stoch_oversold": 20.0,
  "stoch_overbought": 80.0,
  "adx_period": 8,
  "adx_limit": 32.0,
  "balanced_accuracy": 0.7234,
  "signal_counts": {
    "buy": 156,
    "sell": 142,
    "hold": 8702
  },
  "classes": {
    "0": "HOLD",
    "1": "BUY",
    "2": "SELL"
  }
}
```

## 🎓 Features del Modelo

El modelo usa 7 features por barra:

1. **feat_body**: Tamaño del cuerpo de la vela (close - open)
2. **feat_range**: Rango total (high - low)
3. **feat_stoch_main**: Línea principal Stochastic (%K)
4. **feat_stoch_signal**: Línea señal Stochastic (%D)
5. **feat_adx**: Valor ADX
6. **feat_pdi**: +DI (Directional Indicator positivo)
7. **feat_mdi**: -DI (Directional Indicator negativo)

Con un window de 20 barras: **7 × 20 = 140 inputs** al modelo ONNX.

## 📈 Clases de Predicción

| Clase | Valor | Descripción |
|-------|-------|-------------|
| HOLD | 0 | No operar |
| BUY | 1 | Señal de compra |
| SELL | 2 | Señal de venta |

## 🔍 Ejemplo de Salida

```
======================================================================
SGRADT 5.0 - Training Strategy Model
======================================================================
Archivo: EUR_USD_H1.csv
Estrategia: COMBINED
======================================================================

Datos cargados: 10000 barras

Calculando Stochastic...
Calculando ADX...
Datos válidos después de NaN: 9986 barras

Validando movimientos futuros (>50.0 puntos en 10 barras)...

======================================================================
SEÑALES DETECTADAS:
======================================================================
  BUY  (1):    156 señales ( 1.56%)
  SELL (2):    142 señales ( 1.42%)
  HOLD (0):   9688 señales (97.02%)
======================================================================

Preparando ventanas de 20 barras...
Dataset final: X shape = (9966, 140), y shape = (9966,)

======================================================================
ENTRENANDO RANDOM FOREST
======================================================================
Iteraciones: 20
Validación cruzada: TimeSeriesSplit (3 splits)
======================================================================

Fitting 3 folds for each of 20 candidates, totalling 60 fits
✓ Entrenamiento completado
  Mejor Balanced Accuracy: 0.7234
  Mejores parámetros: {'n_estimators': 300, 'min_samples_split': 5, ...}

======================================================================
EXPORTANDO MODELO ONNX
======================================================================

✓ Modelo guardado: onnx/EUR_USD_H1_SGRADT50_combined.onnx
✓ Metadata guardado: onnx/EUR_USD_H1_SGRADT50_combined.meta.json

======================================================================
RESUMEN FINAL
======================================================================
  Input shape: [1, 140]
  Features: 7 x 20 barras
  Output: 3 clases (HOLD=0, BUY=1, SELL=2)
  Accuracy: 0.7234
======================================================================

✅ PROCESO COMPLETADO CON ÉXITO
```

## 💡 Tips de Optimización

### Si hay muy pocas señales:

1. **Reduce `move_points`**: De 50 a 30 puntos
2. **Aumenta `future`**: De 10 a 20 barras
3. **Ajusta Stochastic**:
   - Sube `stoch_oversold` de 20 a 30
   - Baja `stoch_overbought` de 80 a 70
4. **Reduce `adx_limit`**: De 32 a 25
5. **Usa más datos**: Al menos 10,000 barras históricas

### Para mejor accuracy:

1. **Aumenta `n_iter`**: De 20 a 50 o 100
2. **Aumenta `window`**: De 20 a 30 o 50 barras
3. **Usa timeframe menor**: H1 en vez de H4 para más datos

### Para diferentes mercados:

- **Forex**: `move_points=50`, `future=10`
- **Crypto**: `move_points=100`, `future=15`
- **Stocks**: `move_points=2.0`, `future=5`

## 🔧 Integración con EA

El modelo ONNX generado puede ser integrado en un Expert Advisor de MetaTrader 5 usando:

```cpp
// Cargar metadata
string meta_json = // leer archivo .meta.json
int window_size = // extraer "window_size"
int num_inputs = // extraer "num_inputs"

// Preparar input array
double inputs[];
ArrayResize(inputs, num_inputs);

// Llenar con últimas 'window_size' barras
for(int i = 0; i < window_size; i++) {
    int bar = i;
    inputs[i*7 + 0] = Close[bar] - Open[bar];  // feat_body
    inputs[i*7 + 1] = High[bar] - Low[bar];    // feat_range
    inputs[i*7 + 2] = StochMain[bar];          // feat_stoch_main
    inputs[i*7 + 3] = StochSignal[bar];        // feat_stoch_signal
    inputs[i*7 + 4] = ADX[bar];                // feat_adx
    inputs[i*7 + 5] = PDI[bar];                // feat_pdi
    inputs[i*7 + 6] = MDI[bar];                // feat_mdi
}

// Ejecutar predicción ONNX
long prediction = OnnxRun(model_handle, inputs);

// Interpretar
if(prediction == 1) {
    // BUY signal
} else if(prediction == 2) {
    // SELL signal
} else {
    // HOLD
}
```

## 📝 Notas Importantes

1. **Orden de features**: Debe mantenerse exacto según `feature_order` en metadata
2. **Normalización**: El modelo espera valores raw, sin normalización
3. **Barras**: Indexación 0 = barra actual, 1 = anterior, etc.
4. **Timeframe**: Usar el mismo timeframe del CSV de entrenamiento
5. **Validación**: Las señales ya están validadas por movimiento futuro durante entrenamiento

## 📚 Referencias

- Documento de estrategia: `stoch_adx_strategy.md`
- Script original: `train_price_action_adx_points.py`
- Biblioteca de indicadores: `ta` (Technical Analysis Library in Python)
- Conversión ONNX: `skl2onnx`

---

**Versión**: 1.0  
**Fecha**: Marzo 2026  
**Compatible con**: SGRADT 5.0 Minor Timeframe Strategy
