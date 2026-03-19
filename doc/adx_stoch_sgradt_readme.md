# SGRADT 5.0 - Re-adapted Training Scripts

## 📦 Contenido del Proyecto

Este proyecto contiene scripts de entrenamiento de Machine Learning que implementan **exactamente** las condiciones de trading descritas en el documento de estrategia SGRADT 5.0.

### Archivos incluidos:

1. **`train_sgradt_strategy.py`** - Script principal de entrenamiento (NUEVO)
2. **`compare_strategies.py`** - Script de comparación Original vs SGRADT 5.0
3. **`DOCUMENTATION.md`** - Documentación completa y guía de uso
4. **`stoch_adx_strategy.md`** - Documento de estrategia original (referencia)
5. **`train_price_action_adx_points.py`** - Script original (para referencia)

## 🚀 Quick Start

### 1. Instalar dependencias

```bash
pip install pandas numpy scikit-learn skl2onnx ta --break-system-packages
```

### 2. Entrenar modelo (estrategia combinada)

```bash
python train_sgradt_strategy.py \
    --csv your_data.csv \
    --strategy combined \
    --output ./models
```

### 3. Comparar con método original

```bash
python compare_strategies.py \
    --csv your_data.csv \
    --output ./comparison
```

## 🎯 ¿Qué hace diferente este script?

### ❌ Script Original
- Crossover simple de Stochastic (K cruza D)
- Validación básica de zona oversold/overbought
- ADX solo verifica threshold
- **Resultado**: Señales genéricas, muchos falsos positivos

### ✅ Nuevo Script SGRADT 5.0
- **Stochastic**: 2 tipos de crossover + momentum fuerte (+7/-7)
- **ADX**: Pre-condición de tendencia + reversales de DI
- **Combinación**: Ambos indicadores deben confirmar
- **Resultado**: Señales precisas según estrategia real del robot

## 📊 Estrategias Disponibles

### 1. Combined (Recomendada)
Ambos indicadores (Stochastic + ADX) deben confirmar la señal.

**Uso**: Trading conservador, alta precisión.

```bash
python train_sgradt_strategy.py --csv data.csv --strategy combined
```

### 2. Stochastic Only
Solo usa señales de Stochastic Oscillator.

**Uso**: Mercados laterales, scalping.

```bash
python train_sgradt_strategy.py --csv data.csv --strategy stoch
```

### 3. ADX Only
Solo usa señales de ADX con DI+/DI-.

**Uso**: Mercados en fuerte tendencia.

```bash
python train_sgradt_strategy.py --csv data.csv --strategy adx
```

## 🔧 Parámetros Principales

| Parámetro | Default | Descripción |
|-----------|---------|-------------|
| `--csv` | (requerido) | Archivo CSV con datos OHLC |
| `--strategy` | `combined` | Estrategia: stoch, adx o combined |
| `--move_points` | `50.0` | Movimiento mínimo para validar señal |
| `--future` | `10` | Barras futuras para validación |
| `--window` | `20` | Barras de lookback |
| `--stoch_k` | `7` | Período K de Stochastic |
| `--stoch_oversold` | `20.0` | Nivel sobreventa |
| `--stoch_overbought` | `80.0` | Nivel sobrecompra |
| `--adx_period` | `8` | Período ADX |
| `--adx_limit` | `32.0` | Threshold ADX |
| `--n_iter` | `20` | Iteraciones entrenamiento |

## 📈 Ejemplo de Salida

```
======================================================================
SGRADT 5.0 - Training Strategy Model
======================================================================
Archivo: EUR_USD_H1.csv
Estrategia: COMBINED
======================================================================

SEÑALES DETECTADAS:
  BUY  (1):    156 señales ( 1.56%)
  SELL (2):    142 señales ( 1.42%)
  HOLD (0):   9688 señales (97.02%)

✓ Entrenamiento completado
  Mejor Balanced Accuracy: 0.7234

✓ Modelo guardado: models/EUR_USD_H1_SGRADT50_combined.onnx
✓ Metadata guardado: models/EUR_USD_H1_SGRADT50_combined.meta.json
```

## 🔍 Comparación de Métodos

Use el script de comparación para ver las diferencias:

```bash
python compare_strategies.py --csv your_data.csv
```

**Salida típica:**

```
ESTADÍSTICAS DE SEÑALES
Método               BUY        SELL       Total
----------------------------------------------------
Original             234         218         452
SGRADT 5.0           156         142         298
----------------------------------------------------
Diferencia           -78         -76        -154

ANÁLISIS DE COINCIDENCIA
BUY Signals:
  Ambos métodos:       112 (47.9% del original)
  Solo Original:       122
  Solo SGRADT 5.0:      44

SELL Signals:
  Ambos métodos:       103 (47.2% del original)
  Solo Original:       115
  Solo SGRADT 5.0:      39
```

**Interpretación**: SGRADT 5.0 es más selectivo, filtra ~50% de señales del método original, manteniendo las de mayor calidad.

## 📝 Formato del CSV

El archivo CSV debe contener estas columnas:

```csv
timestamp,open,high,low,close,volume
2024-01-01 00:00:00,1.1000,1.1050,1.0980,1.1020,1000
2024-01-01 01:00:00,1.1020,1.1080,1.1010,1.1065,1200
...
```

**Columnas requeridas**: `open`, `high`, `low`, `close`  
**Columnas opcionales**: `timestamp`, `volume`

## 🎓 Condiciones Implementadas

### Stochastic Oscillator

**BUY Signals:**
1. Oversold Crossover (lookback 1-2 y 2-3)
2. Strong Upward Momentum (+7 en dos barras)

**SELL Signals:**
1. Overbought Crossover (lookback 1-2 y 2-3)
2. Strong Downward Momentum (-7 en dos barras)

### Average Directional Index (ADX)

**Pre-condición (mercado en tendencia):**
- ADX > 32 (actual o anterior)
- O ADX aumenta +5 en una barra

**BUY Signals:**
1. DI+ trending up, DI- trending down
2. DI- reversal con DI+ subiendo

**SELL Signals:**
1. DI- trending up, DI+ trending down
2. DI+ reversal con DI- subiendo

Ver `DOCUMENTATION.md` para detalles completos de cada condición.

## 🔗 Integración con MetaTrader 5

El modelo ONNX generado puede integrarse directamente en un EA:

```cpp
// Preparar array de inputs (window * 7 features)
double inputs[140]; // ejemplo: window=20, features=7

// Llenar con datos actuales
for(int i = 0; i < 20; i++) {
    inputs[i*7 + 0] = Close[i] - Open[i];      // body
    inputs[i*7 + 1] = High[i] - Low[i];        // range
    inputs[i*7 + 2] = StochMain[i];            // stoch_main
    inputs[i*7 + 3] = StochSignal[i];          // stoch_signal
    inputs[i*7 + 4] = ADX[i];                  // adx
    inputs[i*7 + 5] = PDI[i];                  // +DI
    inputs[i*7 + 6] = MDI[i];                  // -DI
}

// Ejecutar modelo
long prediction = OnnxRun(model_handle, inputs);

// 0 = HOLD, 1 = BUY, 2 = SELL
```

## 💡 Tips de Optimización

### Pocas señales detectadas:
- ✓ Reduce `--move_points` (de 50 a 30)
- ✓ Aumenta `--future` (de 10 a 20)
- ✓ Ajusta umbrales Stochastic
- ✓ Reduce `--adx_limit` (de 32 a 25)

### Mejorar accuracy:
- ✓ Aumenta `--n_iter` (de 20 a 50+)
- ✓ Aumenta `--window` (de 20 a 30-50)
- ✓ Usa más datos históricos (10k+ barras)

### Por tipo de mercado:
- **Forex**: `move_points=50`, `future=10`
- **Crypto**: `move_points=100`, `future=15`
- **Stocks**: `move_points=2.0`, `future=5`

## 📚 Documentación Completa

Para información detallada, consultar:

- **`DOCUMENTATION.md`** - Guía completa de uso
- **`stoch_adx_strategy.md`** - Estrategia original SGRADT 5.0

## 🤝 Diferencias vs Script Original

| Aspecto | Original | SGRADT 5.0 |
|---------|----------|------------|
| Stochastic crossover | Simple (1 lookback) | Doble (2 lookbacks) + momentum |
| Stochastic momentum | ❌ No incluido | ✅ +7/-7 en 2 barras |
| ADX pre-condición | Solo threshold | Threshold + momentum |
| ADX señales | Solo threshold DI | Trending + Reversal |
| Features | 5 (body, range, ADX, PDI, MDI) | 7 (+ stoch_main, stoch_signal) |
| Señales típicas | ~450 en 10k barras | ~300 en 10k barras |
| Precisión | Media | Alta (más selectivo) |

## ⚠️ Requisitos

- Python 3.8+
- pandas >= 1.3.0
- numpy >= 1.21.0
- scikit-learn >= 1.0.0
- skl2onnx >= 1.10.0
- ta >= 0.10.0

## 📄 Licencia

Este proyecto implementa la estrategia SGRADT 5.0 para fines educativos y de investigación.

## 🔄 Versión

**v1.0** - Marzo 2026  
Compatible con SGRADT 5.0 Minor Timeframe Strategy

---

## 🚦 Getting Started - Paso a Paso

1. **Preparar datos**: CSV con OHLC
2. **Comparar métodos**: `python compare_strategies.py --csv data.csv`
3. **Entrenar modelo**: `python train_sgradt_strategy.py --csv data.csv --strategy combined`
4. **Revisar resultados**: Verificar accuracy y señales en output
5. **Integrar en EA**: Usar archivo .onnx en MetaTrader 5

**¡Listo para entrenar tu estrategia SGRADT 5.0!** 🚀
