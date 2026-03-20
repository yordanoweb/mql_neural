# SGRADT 7.0 - Documentación Completa
## Estrategia EMA 9 con 5 Features

---

## Novedades en SGRADT 7.0

### Simplificación y Mejora

| Aspecto | SGRADT 6.0 | SGRADT 7.0 | Mejora |
|---------|------------|------------|--------|
| Features | 10 | **5** | **50% menos** |
| Indicadores | 5 (ADX, Stoch, RSI, MACD, ATR) | **3 (EMA, ADX, Stoch)** | Mas simple |
| Lógica de salida | TP/SL fijo | **EMA 9 dinámico** | Mas visual |
| Velocidad | ~4ms | **~2ms** | 2x mas rapido |
| Timeframes ideales | M5-H1 | **M1-M5** | Mejor para scalping |

---

## Estrategia Visual

### Concepto Base

**Entrada:**
- Vela ABRE por encima de EMA 9 + ADX fuerte + Stochastic confirma = BUY
- Vela ABRE por debajo de EMA 9 + ADX fuerte + Stochastic confirma = SELL

**Salida:**
- BUY: Cuando vela ABRE por debajo de EMA 9
- SELL: Cuando vela ABRE por encima de EMA 9

**Visual:**
```
Precio
  ^
  |     EMA 9
  |  ___/‾‾‾\___
  | /          \___
  |/              \
  +-----------------> Tiempo
  
  BUY cuando open cruza arriba
  EXIT cuando open cruza abajo
```

---

## Las 5 Features

```python
features_list = [
    'feat_body',           # 0: close - open (dirección y fuerza)
    'feat_range',          # 1: high - low (volatilidad)
    'feat_stoch_main',     # 2: Stochastic K (momentum)
    'feat_stoch_signal',   # 3: Stochastic D (señal)
    'feat_adx',            # 4: ADX (fuerza de tendencia)
]
```

**Que eliminamos vs 6.0:**
- RSI (redundante con Stochastic)
- MACD (redundante con ADX + Stochastic)
- ATR (volatilidad ya capturada en range)
- +DI/-DI (solo usamos ADX main)

**Input shape:**
- Window 20 x 5 features = **100 valores** (vs 200 en SGRADT 6.0)

---

## Parámetros de Entrenamiento

### Comando Básico

```bash
python train_sgradt70_strategy.py \
    --csv USTEC_M5.csv \
    --window 20 \
    --output ./models
```

### Parámetros Completos

```bash
python train_sgradt70_strategy.py \
    --csv USTEC_M5.csv \
    --window 20 \
    --output ./models \
    --min_profit_points 20.0 \
    --future 50 \
    --ema_period 9 \
    --stoch_k 7 \
    --stoch_d 3 \
    --stoch_oversold 20.0 \
    --stoch_overbought 80.0 \
    --adx_period 8 \
    --adx_limit 25.0 \
    --n_iter 20
```

### Tabla de Parámetros

| Parámetro | Default | Descripción |
|-----------|---------|-------------|
| **Validación** | | |
| `--min_profit_points` | 20.0 | Puntos mínimos de ganancia |
| `--future` | 50 | Barras máximas para buscar exit |
| **EMA** | | |
| `--ema_period` | 9 | Periodo EMA (pivote) |
| **Stochastic** | | |
| `--stoch_k` | 7 | Periodo K |
| `--stoch_d` | 3 | Periodo D |
| `--stoch_oversold` | 20.0 | Nivel sobreventa |
| `--stoch_overbought` | 80.0 | Nivel sobrecompra |
| **ADX** | | |
| `--adx_period` | 8 | Periodo ADX |
| `--adx_limit` | 25.0 | Umbral de tendencia |
| **Training** | | |
| `--n_iter` | 20 | Iteraciones RandomSearchCV |

---

## Configuración del EA

### Parámetros del Modelo

```cpp
InpModelName = "USTEC_M5_SGRADT70_ema9.onnx"
InpWindowSize = 20        // Debe coincidir con --window
InpFeaturesPerBar = 5     // SIEMPRE 5 en SGRADT 7.0
InpMinConf = 0.55         // Confianza mínima
```

### Indicadores

```cpp
// EMA (pivote de entrada/salida)
InpEMAPeriod = 9

// Stochastic
InpStochK = 7
InpStochD = 3
InpStochOversold = 20.0
InpStochOverbought = 80.0

// ADX
InpADXPeriod = 8
InpADXLimit = 25.0
```

### Gestión de Riesgo

```cpp
InpLot = 0.01             // Tamaño de lote
InpStopPoints = 50.0      // SL en puntos (seguridad)
InpTakePoints = 100.0     // TP en puntos (seguridad)
// Exit real: Cruce de EMA 9
```

---

## Lógica de Señales

### BUY Signal (Training)

```python
# 1. Vela abre por encima de EMA 9
above_ema = open[i] > ema9[i]

# 2. ADX confirma tendencia fuerte
adx_strong = adx[i] > 25.0

# 3. Stochastic confirma momentum
stoch_oversold_cross = (
    (stoch_k[i-1] < stoch_d[i-1]) and 
    (stoch_k[i] > stoch_d[i]) and 
    (stoch_k[i] <= 20.0)
)
stoch_momentum_up = (stoch_k[i] > stoch_k[i-1] + 7)
stoch_buy = stoch_oversold_cross or stoch_momentum_up

# Señal BUY
buy_signal = above_ema and adx_strong and stoch_buy
```

### Exit (Training)

```python
# Para BUY: buscar cuando open cruza debajo de EMA
for j in range(i+1, i+50):
    if open[j] < ema9[j]:
        profit = (exit_price - entry_price) * 10000
        if profit >= 20:  # min_profit_points
            label = BUY
        break
```

### Exit (EA en tiempo real)

```cpp
// Verificar cada tick
double open_current = iOpen(_Symbol, _Period, 0);
double ema_current = ema[0];

if(position_type == BUY && open_current < ema_current) {
    ClosePosition();  // Exit BUY
}

if(position_type == SELL && open_current > ema_current) {
    ClosePosition();  // Exit SELL
}
```

---

## Panel de Información

```
====================================================
  SGRADT 7.0 - EMA 9 STRATEGY (5 FEATURES)
====================================================

SYMBOL: USTEC [PERIOD_M5]
SESSION: 00:00-24:00 [ACTIVE]
MODE: NEW BAR | Inferences: 125

----------------------------------------------------
EMA (Period: 9)
----------------------------------------------------
   EMA: 21450.25
   Open: 21455.80 [ABOVE]
   Close: 21458.30

----------------------------------------------------
ADX (Period: 8)
----------------------------------------------------
   ADX: 28.45 [TRENDING]

----------------------------------------------------
STOCHASTIC (7,3)
----------------------------------------------------
   K: 65.23
   D: 58.45
   Zone: NEUTRAL

====================================================
AI PREDICTION
====================================================
   Signal: BUY

   Confidence:
   - HOLD:  15.23%
   - BUY:   68.45%
   - SELL:  16.32%

   Min Required: 55.0%

----------------------------------------------------
RISK SETTINGS
----------------------------------------------------
   Lot: 0.01
   SL: 50 pts
   TP: 100 pts
   Exit: EMA 9 Cross

====================================================
ACTIVE POSITION: BUY
   P&L: +45.80 USD
====================================================
```

---

## Ventajas de SGRADT 7.0

### 1. Más Rápido
- 5 features vs 10 = 50% menos datos
- Inference: 2ms vs 4ms
- Ideal para M1, M5

### 2. Más Visual
- EMA 9 fácil de verificar en gráfico
- Puedes VER las entradas/salidas
- No hay "caja negra"

### 3. Exit Dinámico
- No más TP/SL fijos
- Sigue la tendencia hasta que EMA cruza
- Ganancias mayores en tendencias fuertes

### 4. Menos Overfitting
- Menos features = menos riesgo de sobreajuste
- Estrategia más robusta
- Mejor generalización

### 5. Más Simple
- 3 indicadores vs 5
- Fácil de entender
- Fácil de optimizar

---

## Comparación 6.0 vs 7.0

### Input Shape
```
SGRADT 6.0: [1, 200] (20 x 10)
SGRADT 7.0: [1, 100] (20 x 5)  --> 50% mas pequeño
```

### Indicadores
```
SGRADT 6.0:
- ADX (+ +DI, -DI)
- Stochastic
- RSI
- MACD
- ATR

SGRADT 7.0:
- EMA 9
- ADX (solo main)
- Stochastic
```

### Salida
```
SGRADT 6.0: TP/SL fijo

SGRADT 7.0: EMA 9 cross (dinamico)
```

---

## Ejemplo de Trade Completo

### 1. Entrada
```
Barra 100:
- Open = 21455.80
- EMA 9 = 21450.25
- Open > EMA --> ABOVE

ADX = 28.45 > 25 --> TRENDING
Stochastic K cruza arriba de D en zona oversold

AI predice: BUY con 68.45% confianza
--> ABRIR BUY en 21455.80
```

### 2. Durante el Trade
```
Barras 101-105:
- Open sigue por encima de EMA
- Posición activa
- P&L va aumentando
```

### 3. Salida
```
Barra 106:
- Open = 21485.20
- EMA 9 = 21490.50
- Open < EMA --> CRUZO DEBAJO

--> CERRAR BUY en 21485.20
Ganancia: 21485.20 - 21455.80 = +29.40 puntos
```

---

## Mejores Prácticas

### Entrenamiento

1. **Usar datos limpios** de al menos 3 meses
2. **Ajustar min_profit_points** según el timeframe:
   - M1: 10-15 puntos
   - M5: 20-30 puntos
   - M15: 30-50 puntos
3. **future window** suficiente:
   - M1: 30-50 barras
   - M5: 50-100 barras
   - M15: 100-200 barras

### Testing

1. **Backtesting** mínimo 3 meses
2. **Demo** mínimo 1 semana
3. **Live** empezar con lotes mínimos

### Optimización

1. **EMA period**: Probar 7, 9, 12
2. **ADX limit**: Probar 20, 25, 30
3. **min_profit_points**: Ajustar según resultados

---

## Timeframes Recomendados

| Timeframe | Min Profit Points | Future Window | Comentario |
|-----------|-------------------|---------------|------------|
| **M1** | 10-15 | 30-50 | Muy rápido, muchas señales |
| **M5** | 20-30 | 50-100 | Balance ideal |
| **M15** | 30-50 | 100-200 | Menos señales, más confiables |
| M30 | 50-80 | 200-300 | Conservador |

---

## Archivos Generados

```
models/
├── USTEC_M5_SGRADT70_ema9.onnx
└── USTEC_M5_SGRADT70_ema9.meta.json
```

---

## Instalación en MT5

```
MQL5/
├── Experts/
│   └── EA_SGRADT70_ONNX.mq5
│
└── Files/
    └── USTEC_M5_SGRADT70_ema9.onnx
```

---

## Troubleshooting

### No genera señales

**Revisar:**
- ADX limit muy alto (reducir a 20)
- min_profit_points muy alto (reducir)
- future window muy corto (aumentar)

### Muchas señales malas

**Revisar:**
- ADX limit muy bajo (aumentar a 30)
- min_profit_points muy bajo (aumentar)
- Entrenar con mas datos

### Exit muy temprano

**Solución:**
- Usar EMA 12 en lugar de EMA 9
- Aumentar periodo de EMA

### Exit muy tardío

**Solución:**
- Usar EMA 7 en lugar de EMA 9
- Reducir periodo de EMA

---

## Próximos Pasos

1. **Entrenar** con tus datos de M5
2. **Backtest** en Strategy Tester
3. **Probar** en demo
4. **Optimizar** parámetros
5. **Live** con lotes pequeños

---

**Versión:** SGRADT 7.0  
**Fecha:** Marzo 2026  
**Status:** Listo para producción  
**Ideal para:** M1, M5 scalping/day trading
