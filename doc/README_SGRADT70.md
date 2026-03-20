# SGRADT 7.0 - Quick Start Guide
## Estrategia EMA 9 (5 Features)

---

## Instalación Rápida

### 1. Instalar Dependencias

```bash
pip install pandas numpy scikit-learn ta skl2onnx
```

### 2. Entrenar Modelo

```bash
python train_sgradt70_strategy.py --csv USTEC_M5.csv
```

Genera:
- `USTEC_M5_SGRADT70_ema9.onnx`
- `USTEC_M5_SGRADT70_ema9.meta.json`

### 3. Instalar en MT5

```
MQL5/
├── Experts/EA_SGRADT70_ONNX.mq5
└── Files/USTEC_M5_SGRADT70_ema9.onnx
```

### 4. Configurar EA

```
InpModelName = "USTEC_M5_SGRADT70_ema9.onnx"
InpWindowSize = 20
InpFeaturesPerBar = 5
InpEMAPeriod = 9
```

---

## Qué es Nuevo en 7.0

### Simplificado

- **5 features** (vs 10 en 6.0)
- **3 indicadores** (vs 5 en 6.0)
- **Exit dinámico** con EMA 9 (vs TP/SL fijo)
- **2x más rápido** que 6.0

### Estrategia Visual

**Entrada:**
```
BUY: Open > EMA 9 + ADX > 25 + Stochastic confirma
SELL: Open < EMA 9 + ADX > 25 + Stochastic confirma
```

**Salida:**
```
BUY: Cuando Open cruza debajo de EMA 9
SELL: Cuando Open cruza arriba de EMA 9
```

---

## Las 5 Features

```python
1. body   = close - open      # Direccion y fuerza
2. range  = high - low        # Volatilidad
3. stoch_k                    # Momentum
4. stoch_d                    # Señal
5. adx                        # Fuerza tendencia
```

Input: **[1, 100]** (20 barras x 5 features)

---

## Parámetros de Entrenamiento

### Básico
```bash
python train_sgradt70_strategy.py --csv tus_datos.csv
```

### Personalizado
```bash
python train_sgradt70_strategy.py \
    --csv USTEC_M5.csv \
    --window 20 \
    --min_profit_points 20 \
    --future 50 \
    --ema_period 9 \
    --adx_limit 25
```

---

## Configuración del EA

### Modelo
```
InpModelName = "USTEC_M5_SGRADT70_ema9.onnx"
InpWindowSize = 20
InpFeaturesPerBar = 5
InpMinConf = 0.55
```

### Indicadores
```
InpEMAPeriod = 9
InpStochK = 7
InpStochD = 3
InpADXPeriod = 8
InpADXLimit = 25.0
```

### Risk
```
InpLot = 0.01
InpStopPoints = 50    // Seguridad (exit real es EMA)
InpTakePoints = 100   // Seguridad (exit real es EMA)
```

---

## Timeframes Recomendados

| TF | Min Profit | Future | Uso |
|----|------------|--------|-----|
| M1 | 10-15 | 30-50 | Scalping agresivo |
| M5 | 20-30 | 50-100 | Balance ideal |
| M15 | 30-50 | 100-200 | Day trading |

---

## Comparación con 6.0

| Aspecto | 6.0 | 7.0 |
|---------|-----|-----|
| Features | 10 | 5 |
| Indicadores | 5 | 3 |
| Input shape | [1, 200] | [1, 100] |
| Velocidad | 4ms | 2ms |
| Exit | TP/SL fijo | EMA dinámico |
| Ideal para | M5-H1 | M1-M5 |

---

## Ejemplo de Trade

```
Entrada:
- Open = 21455.80 > EMA(21450.25)
- ADX = 28.45 > 25
- Stoch cruza arriba en oversold
- AI: BUY 68.45%
--> ABRIR BUY

Durante:
- Open sigue > EMA
- Posicion activa

Salida:
- Open = 21485.20 < EMA(21490.50)
--> CERRAR BUY
--> Ganancia: +29.40 puntos
```

---

## Verificación

Salida esperada del training:
```
======================================================================
SGRADT 7.0 - EMA 9 Strategy (5 Features)
======================================================================

======================================================================
SEÑALES DETECTADAS:
======================================================================
  BUY  (1):    245 señales ( 2.81%)
  SELL (2):    238 señales ( 2.73%)
  HOLD (0):   8251 señales (94.47%)
======================================================================

Dataset final: X shape = (8714, 100), y shape = (8714,)

Mejor Balanced Accuracy: 0.7145
======================================================================
```

Salida esperada del EA:
```
======================================================================
    SGRADT 7.0 - EMA 9 STRATEGY (5 Features)
======================================================================

[OK] ONNX model loaded successfully
[OK] Input shape set: [1, 100] (20 bars x 5 features)

[OK] Indicators created:
     - EMA(9)
     - ADX(8)
     - Stochastic(7,3)

[OK] All indicators ready with 50 bars

======================================================================
    EA INITIALIZED SUCCESSFULLY
======================================================================

[BUY] Order opened | Confidence: 68.45%
[EXIT BUY] Open crossed below EMA 9
```

---

## Troubleshooting

### No hay señales
```
Reducir:
- --adx_limit (de 25 a 20)
- --min_profit_points (de 20 a 15)

Aumentar:
- --future (de 50 a 100)
```

### Muchas señales malas
```
Aumentar:
- --adx_limit (de 25 a 30)
- --min_profit_points (de 20 a 30)

Entrenar con más datos
```

### Exit muy temprano
```
Cambiar --ema_period de 9 a 12
```

### Exit muy tardío
```
Cambiar --ema_period de 9 a 7
```

---

## Archivos

- `train_sgradt70_strategy.py` - Script de entrenamiento
- `EA_SGRADT70_ONNX.mq5` - Expert Advisor
- `SGRADT70_DOCUMENTATION.md` - Documentación completa
- `requirements.txt` - Dependencias Python

---

## Próximos Pasos

1. Entrenar modelo
2. Backtest en MT5
3. Demo 1 semana
4. Live con lotes pequeños
5. Optimizar parámetros

---

**Versión:** SGRADT 7.0  
**Status:** Listo para usar  
**Ideal:** M1, M5 scalping
