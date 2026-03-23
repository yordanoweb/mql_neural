# SGRADT 5.0 - Expert Advisor con ONNX

Sistema de trading algorítmico basado en Machine Learning que utiliza 5 features técnicas para predecir señales de BUY/SELL/HOLD.

## Características Principales

- **5 Features exclusivas**: Stochastic %K, %D, ADX, +DI, -DI
- **Decisiones basadas en inferencia ONNX + filtro EMA**: El modelo decide la dirección; el EMA gate filtra entradas contra tendencia
- **Validación de movimiento futuro**: Las señales de entrenamiento se validan con movimientos reales
- **Panel de información reducido**: Información esencial sin decoraciones

## Features del Modelo

El modelo utiliza exactamente 5 indicadores técnicos:

1. **feat_stoch_main** - Stochastic %K
2. **feat_stoch_signal** - Stochastic %D  
3. **feat_adx** - Average Directional Index
4. **feat_pdi** - Plus Directional Indicator (+DI)
5. **feat_mdi** - Minus Directional Indicator (-DI)

## Requisitos

### Python
```bash
pip install pandas numpy scikit-learn skl2onnx ta
```

### MetaTrader 5
- MT5 con soporte ONNX
- Carpeta MQL5/Files/ accesible

## Uso

### 1. Entrenamiento del Modelo

El script ahora soporta procesamiento de **múltiples archivos CSV** en una sola ejecución:

```bash
# Un solo archivo
python train_sgradt_strategy.py \
    --csv EUR_USD_H1.csv \
    --output ./onnx \
    --window 20 \
    --move_points 50.0 \
    --future 10 \
    --stoch_k 7 \
    --stoch_d 3 \
    --stoch_slowing 3 \
    --adx_period 8 \
    --n_iter 20

# Múltiples archivos
python train_sgradt_strategy.py \
    --csv EUR_USD_H1.csv GBP_USD_H1.csv USD_JPY_H1.csv \
    --output ./onnx \
    --window 20 \
    --move_points 50.0 \
    --future 10 \
    --n_iter 20
```

**Parámetros clave:**
- `--csv`: Uno o más archivos CSV con columnas: time, open, high, low, close
- `--output`: Directorio de salida para archivos .onnx y .meta.json
- `--window`: Ventana de lookback (número de barras históricas)
- `--move_points`: Puntos mínimos de movimiento para validar señal
- `--future`: Barras futuras para validar el movimiento
- `--n_iter`: Iteraciones de búsqueda de hiperparámetros

**Salida por cada archivo:**
- `{symbol}_SGRADT50.onnx` - Modelo ONNX
- `{symbol}_SGRADT50.meta.json` - Metadata del modelo

**Temporizador integrado:**
El script muestra el tiempo de procesamiento para cada archivo individual y el tiempo total al finalizar todos los archivos. Ejemplo de salida:

```
######################################################################
# PROCESANDO ARCHIVO 1/3
# EUR_USD_H1.csv
######################################################################

... (procesamiento) ...

RESUMEN ARCHIVO
======================================================================
  Input shape: [1, 100]
  Features: 5 x 20 barras
  Accuracy: 0.7234
  TIEMPO: 2m 34.56s
======================================================================

######################################################################
# RESUMEN FINAL DE PROCESAMIENTO
######################################################################

Total de archivos procesados: 3
  Exitosos: 3
  Fallidos: 0

Tiempo total: 8m 15.32s

======================================================================
MODELOS GENERADOS EXITOSAMENTE:
======================================================================
  EUR_USD_H1.csv                           | Acc: 0.7234 | Tiempo: 2m 34.56s
  GBP_USD_H1.csv                           | Acc: 0.6891 | Tiempo: 3m 02.18s
  USD_JPY_H1.csv                           | Acc: 0.7512 | Tiempo: 2m 38.58s

######################################################################
# PROCESO COMPLETADO
# Tiempo total: 8m 15.32s
######################################################################
```

### 2. Configuración del EA

1. Copiar `EA_SGRADT50_ONNX.mq5` a `MQL5/Experts/`
2. Copiar archivos `.onnx` y `.meta.json` a `MQL5/Files/`
3. Compilar el EA en MetaEditor
4. Configurar parámetros en el gráfico

**Parámetros importantes del EA:**

```cpp
// Model Configuration
InpModelName        = "EUR_USD_H1_SGRADT50.onnx"
InpMinConf          = 0.55    // Confianza mínima (55%)
InpWindowSize       = 20      // Debe coincidir con training
InpFeaturesPerBar   = 5       // FIJO: 5 features

// Inference Timing
InpInferSeconds     = 15      // Inferencia cada 15 segundos (0=solo nueva barra)
InpOneTradePerBar   = true    // Limitar a 1 operación por barra

// Trading Session
InpStartHour        = 0       // Hora de inicio (0-23)
InpEndHour          = 24      // Hora de fin (0-24)

// EMA Gate
InpEMAPeriod        = 9       // BUY: Ask > EMA | SELL: Bid < EMA

// Indicator Parameters
InpStochK           = 7
InpStochD           = 3
InpStochSlowing     = 3
InpADXPeriod        = 8

// Risk Management
InpLot              = 1.0
InpStopPoints       = 50.0    // Stop Loss en POINTS
InpTakePoints       = 100.0   // Take Profit en POINTS
```

## Arquitectura del Sistema

### Flujo de Entrenamiento

1. **Carga de datos**: Lee CSV con datos OHLC
2. **Cálculo de indicadores**: Genera las 5 features
3. **Validación de movimientos**: Identifica movimientos futuros válidos
4. **Labeling**: Asigna etiquetas (0=HOLD, 1=BUY, 2=SELL)
5. **Ventanas**: Crea secuencias de `window_size` barras
6. **Entrenamiento**: Random Forest con búsqueda de hiperparámetros
7. **Exportación**: Convierte a formato ONNX

### Flujo de Trading

1. **Carga del modelo**: Lee archivo .onnx en OnInit
2. **Cálculo de indicadores**: Mantiene handles de Stochastic y ADX
3. **Preparación de features**: Construye vector de entrada [1, window×5]
4. **Inferencia ONNX**: Ejecuta modelo y obtiene predicción + probabilidades
5. **Evaluación de condiciones**:
   - Confianza >= umbral mínimo
   - Dentro del horario de trading
   - No hay posición abierta
   - No se ha operado esta barra (si activado)
   - **EMA gate**: Ask > EMA para BUY, Bid < EMA para SELL
6. **Ejecución**: Abre posición BUY o SELL según predicción

## Validación de Señales

El sistema valida que las señales generadas durante el entrenamiento resulten en movimientos reales:

- **BUY válido**: Precio sube al menos `move_points` puntos en las próximas `future` barras
- **SELL válido**: Precio baja al menos `move_points` puntos en las próximas `future` barras
- **HOLD**: Cualquier otra situación

Esto asegura que el modelo aprenda de señales que históricamente produjeron movimientos significativos.

### Implementación del forward-looking window

El máximo y mínimo futuro se calculan con:

```python
future_max = close.shift(-future).rolling(window=future).max()
future_min = close.shift(-future).rolling(window=future).min()
```

`shift(-future)` desplaza la serie `future` pasos hacia el futuro en el índice, de modo que `rolling(window=future).max()` cubre exactamente las barras `i+1` a `i+future` para cada barra `i`.

> **⚠️ Bug conocido (corregido en v5.0.1):** Usar `shift(-1)` en lugar de `shift(-future)` antes del rolling produce una ventana incorrecta que etiqueta la barra equivocada, **invirtiendo efectivamente todas las señales BUY/SELL**: el modelo aprende que indicadores de tendencia alcista corresponden a SELL y viceversa. Síntoma observable: el EA emite señales BUY persistentes con alta confianza durante tendencias bajistas claras, todas rechazadas por el EMA gate. Los modelos entrenados con la versión anterior deben **re-entrenarse desde cero**.

## Panel de Información

El EA muestra un panel reducido con:
- Estado de la sesión (activa/cerrada)
- Modo de inferencia (timer/nueva barra) y número de ejecuciones
- Valores actuales de indicadores (ADX, +DI, -DI, Stochastic K/D)
- Señal AI (BUY/SELL/HOLD) con probabilidades
- Configuración de riesgo (lote, SL, TP)
- Estado de posición actual (si existe)

## Formato de Datos CSV

El CSV debe contener estas columnas:

```
time,open,high,low,close
2024-01-01 00:00:00,1.1050,1.1055,1.1045,1.1052
2024-01-01 01:00:00,1.1052,1.1060,1.1050,1.1058
...
```

## Interpretación de Salidas

### Durante el Entrenamiento

```
SEÑALES DETECTADAS:
  BUY  (1):    523 señales ( 8.45%)
  SELL (2):    489 señales ( 7.89%)
  HOLD (0):   5188 señales (83.66%)
```

Si hay muy pocas señales (<10 por clase):
- Reduce `--move_points`
- Aumenta `--future`
- Usa más datos históricos

### Durante el Trading

```
Inference #42: BUY signal | Conf: 67.34% | Time: OK | Bar: OK | Position: NONE
Opening BUY: Price=1.10520, SL=1.10470, TP=1.10620, Lot=1.00
BUY order executed successfully
```

## Notas Importantes

1. **InpFeaturesPerBar debe ser 5**: Este valor es fijo y debe coincidir con el número de features del modelo
2. **InpWindowSize debe coincidir**: Usar el mismo valor que en el entrenamiento
3. **Parámetros de indicadores**: Deben ser idénticos entre entrenamiento y trading
4. **EMA gate activo en el EA**: El EA aplica un filtro de tendencia vía EMA sobre el precio ejecutable (Ask para BUY, Bid para SELL). Ajustar `InpEMAPeriod` según el timeframe usado
5. **Testing recomendado**: Probar primero en cuenta demo para validar el comportamiento

## Troubleshooting

**Error: Cannot load ONNX model**
- Verificar que el archivo .onnx esté en MQL5/Files/
- Verificar nombre correcto del archivo

**Error: Cannot prepare features**
- Verificar que los indicadores se hayan inicializado correctamente
- Verificar que haya suficiente historial de barras

**Muy pocas señales durante entrenamiento**
- Reducir `move_points` (ej. de 50 a 30)
- Aumentar `future` (ej. de 10 a 15)
- Asegurar suficientes datos (mínimo 5000+ barras recomendadas)

**Señales rechazadas por baja confianza**
- Reducir `InpMinConf` (ej. de 0.55 a 0.50)
- Verificar que el modelo se entrenó con datos similares al mercado actual

**Señales rechazadas por EMA gate**
- El precio ejecutable (Ask/Bid) está al lado equivocado de la EMA
- Ajustar `InpEMAPeriod` a un período más largo para un filtro más permisivo, o más corto para uno más reactivo
- Es el comportamiento esperado: el gate evita entradas contra tendencia

## Licencia

Este código es de uso libre para fines educativos y de investigación.
