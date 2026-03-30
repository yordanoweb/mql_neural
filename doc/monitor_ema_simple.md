## ***Resumen de Cambios**

### ✅ ***Mantenido**

- ***`--use\_h1\_candle\_gate` - Filtro de dirección de vela H1**

- ***`--start\_hour` / `--end\_hour` - Horario de trading**

- ***SL/TP fijos basados en ATR (sin trailing stop)**

### ❌ ***Eliminado**

- ***Stochastic (todas las funciones y parámetros `--stoch\_\*`)**

- ***ADX (todas las funciones y parámetros `--adx\_\*`)**

- ***Candle direction gate (dirección de vela actual)**

- ***Forecast exit (`--forecast\_horizon`)**

- ***`--entry\_points` (reemplazado)**

### 🆕 ***Nuevo**

***Table**

| **Parámetro** | **Default** | **Descripción** |
| - | - | - |
| `--entry\_confirmation\_atr` | 0.5 | Multiplicador ATR para confirmar entrada después del cruce |
| `--exit\_cross\_atr` | 1.0 | Multiplicador ATR para distancia de salida por cruce inverso |


## ***Lógica Implementada**

### ***Entrada**

***plain**

***Copy**

```
***`1. Detectar cruce de EMA en velas COMPLETADAS:`**

`   ***- BUY: close\[-3\] \< EMA\[-3\] Y close\[-2\] \> EMA\[-2\]`**

`   ***- SELL: close\[-3\] \> EMA\[-3\] Y close\[-2\] \< EMA\[-2\]`**


***`2. Confirmar distancia: |close - EMA| \>= ATR \* entry\_confirmation\_atr`**


***`3. Verificar filtro H1 (si está activo)`**


***`4. Ejecutar orden con SL/TP basados en ATR`**
```

### ***Salida**

***plain**

***Copy**

```
***`1. Detectar cruce inverso en velas COMPLETADAS`**


***`2. Confirmar alejamiento:`**

`   ***- BUY: (EMA - close) \>= ATR \* exit\_cross\_atr`**

`   ***- SELL: (close - EMA) \>= ATR \* exit\_cross\_atr`**


***`3. Cerrar posición`**
```


## ***Ejemplo de Uso**

***bash**

***Copy**

```
***`\# Configuración básica`***

***`python monitor\_ema\_simple.py --symbol EURUSD --timeframe M5 --ema\_period 9`**


***`\# Con filtro H1 y horario limitado`***

***`python monitor\_ema\_simple.py --symbol GBPUSD --timeframe M15 --use\_h1\_candle\_gate --start\_hour 8 --end\_hour 20`**


***`\# Ajustar sensibilidad de entrada/salida`***

***`python monitor\_ema\_simple.py --entry\_confirmation\_atr 0.3 --exit\_cross\_atr 0.8 --sl\_multiplier 1.5 --tp\_multiplier 2.5`**
```

***explica mejor estos dos parametros porque me son confusos en cuanto a la cantidad de puntos requeridos para usarlos: --entry\_confirmation\_atr --exit\_cross\_atr**

***Edit**

***Copy**

***Share**

***Search**

***ATR Average True Range forex trading points pips calculation**

***8 results**

***Perfecto, ahora te explico estos dos parámetros con claridad:**


## ***Explicación de los Parámetros ATR**

### ***¿Qué es el ATR?**

***El ATR (Average True Range) mide la volatilidad promedio del mercado en pips/puntos. Es decir, te dice cuánto se mueve típicamente el precio en una vela.**

***Por ejemplo:**

- ***Si ATR = 0.0016 en EUR/USD → equivale a 16 pips **

- ***Si ATR = 0.0080 en EUR/USD → equivale a 80 pips**


### ***`--entry\_confirmation\_atr` (Default: 0.5)**

***¿Qué hace? Define qué tan lejos debe estar el precio de la EMA después del cruce para confirmar la entrada.**

***Cálculo de puntos requeridos:**

***plain**

***Copy**

```
***`Puntos requeridos = ATR × entry\_confirmation\_atr`**
```

***Table**

| **Ejemplo EUR/USD** | **ATR** | **Multiplicador** | **Puntos requeridos** |
| - | - | - | - |
| Mercado calmado | 10 pips | 0.5 | **5 pips** |
| Mercado normal | 20 pips | 0.5 | **10 pips** |
| Mercado volátil | 40 pips | 0.5 | **20 pips** |

***¿Por qué es útil? Evita entradas falsas cuando el precio "roza" la EMA pero no tiene momentum. Requiere que el precio se aleje lo suficiente para confirmar que el cruce es válido.**


### ***`--exit\_cross\_atr` (Default: 1.0)**

***¿Qué hace? Define qué tan lejos debe estar el precio de la EMA en dirección contraria para confirmar la salida por cruce inverso.**

***Cálculo de puntos requeridos:**

***plain**

***Copy**

```
***`Puntos requeridos = ATR × exit\_cross\_atr`**
```

***Table**

| **Ejemplo EUR/USD** | **ATR** | **Multiplicador** | **Puntos requeridos** |
| - | - | - | - |
| Mercado calmado | 10 pips | 1.0 | **10 pips** |
| Mercado normal | 20 pips | 1.0 | **20 pips** |
| Mercado volátil | 40 pips | 1.0 | **40 pips** |

***¿Por qué es útil? Evita salidas prematuras por pequeños retrocesos. Requiere que el precio cruce la EMA Y se aleje significativamente antes de cerrar la posición.**


## ***Visualización**

***plain**

***Copy**

```
***`ENTRADA BUY:`**

***`Precio cruza EMA hacia arriba → Espera cierre de vela`**

`↓`

***`¿Distancia desde EMA ≥ ATR × 0.5? `**

`   ***SÍ → Entra`**

`   ***NO → Espera`**


***`SALIDA BUY (cruce inverso):`**

***`Precio cruza EMA hacia abajo → Espera cierre de vela  `**

`↓`

***`¿Distancia debajo de EMA ≥ ATR × 1.0?`**

`   ***SÍ → Cierra posición`**

`   ***NO → Mantiene posición`**
```


## ***Valores recomendados según tu estilo**

***Table**

| **Estilo** | **`--entry\_confirmation\_atr`** | **`--exit\_cross\_atr`** | **Razón** |
| - | - | - | - |
| **Conservador** | 0.3 - 0.5 | 0.8 - 1.0 | Entra solo con momentum fuerte, sale ante señales débiles |
| **Moderado** | 0.5 - 0.8 | 1.0 - 1.5 | Balance entre confirmación y oportunidad |
| **Agresivo** | 0.2 - 0.3 | 1.5 - 2.0 | Entra rápido, tolera más drawdown antes de salir |


## ***Ejemplo práctico**

***bash**

***Copy**

```
***`\# EUR/USD en M5 con ATR típico de 15 pips`***

***`python monitor\_ema\_simple.py \\`**

`  ***--symbol EURUSD \\`**

`  ***--timeframe M5 \\`**

`  ***--entry\_confirmation\_atr 0.5 \\   *\# Requiere 7.5 pips de confirmación`***

`  ***--exit\_cross\_atr 1.0 \\            *\# Requiere 15 pips para salir`***

`  ***--sl\_multiplier 2.0 \\`**

`  ***--tp\_multiplier 3.0`**
```

***La clave es que ambos parámetros se adaptan automáticamente a la volatilidad del mercado: en días volátiles requieren más puntos, en días calmados requieren menos.**

