# 🧠 1. Definición formal del problema

## Hipótesis operativa

> “Dado el estado del mercado antes de las 20:00 (RJ), existe una probabilidad de que entre 20:00–21:00 ocurra un movimiento direccional explotable.”


# 🧩 2. Estructura general del pipeline

Dividido en 3 etapas:

```
\[Estado previo del mercado\]

            ↓

\[Modelo 1: ¿habrá movimiento?\]

            ↓ (si = 1)

\[Modelo 2: dirección\]

            ↓

\[Decisión final\]
```


# 🔹 3. Definición exacta de la muestra (esto es clave)

Cada sample NO es una vela.

👉 Cada sample es:

> “el estado del mercado justo antes de la ventana objetivo”


## Cómo se construye un sample

Para cada día:

- Tomas el último punto antes de las 20:00 

- Construyes un window hacia atrás (ej: 20 velas) 

- Eso es el input 


## Resultado

Pasas de:

- miles de samples correlacionados 

a:

- **1 sample por día (independiente)** 

👉 Esto reduce muchísimo el ruido.


# 🎯 4. Target bien definido (núcleo del diseño)

## Ventana objetivo fija

Ejemplo:

- Inicio: 20:00 

- Fin: 21:00 


## Dentro de esa ventana calculas:

- `max\_up` 

- `max\_down` 


## 4.1 Modelo 1 → “¿Hay oportunidad?”

```
target\_opportunity =

    1 si max(|move|) ≥ threshold

    0 si no
```

👉 Aquí NO hay dirección.


## 4.2 Modelo 2 → Dirección

Solo si hubo oportunidad:

```
target\_direction =

    1 si max\_up \> max\_down

    0 si max\_down \> max\_up
```


## 🔴 Casos ambiguos (muy importante)

Si:

```
max\_up ≈ max\_down
```

👉 se eliminan del dataset


## Resultado

- Dataset limpio 

- Señal coherente 

- Menos ruido 


# ⏱️ 5. Alineación temporal (esto corrige tu problema actual)

Ahora todo está anclado a:

> “lo que pasa específicamente en esa hora”

No:

> “lo que pasa en cualquier momento del futuro”


# 🧱 6. Features (estructura, no lista)

Ahora sí tienen sentido:

## Tipo A — Estado previo

- momentum reciente 

- volatilidad 

- compresión/expansión 

## Tipo B — Contexto temporal

- hora (codificada cíclicamente) 

- distancia a la ventana 

- sesión (Asia / London / NY) 

## Tipo C — Régimen

- tendencia vs rango 

- fuerza de tendencia 


👉 Diferencia clave:

Antes:

- features → intentan adivinar todo 

Ahora:

- features → describen el estado previo a un evento fijo 


# 📊 7. Dataset final

## Modelo 1

- X: estado previo 

- y: 0 / 1 (hay movimiento o no) 


## Modelo 2

- X: mismo estado previo 

- y: dirección (solo casos con movimiento) 


# 🔁 8. Flujo en inferencia

En tiempo real:

```
antes de 20:00 →

    Modelo 1 →

        si 0 → no trade

        si 1 →

            Modelo 2 →

                long o short
```


# 🧠 9. Qué cambia estructuralmente

### Antes:

- muestras mezcladas 

- targets difusos 

- tiempo implícito 


### Ahora:

- muestras alineadas por evento 

- targets limpios 

- tiempo explícito 


# ⚠️ 10. Insight importante

Este diseño transforma el problema de:

> predicción continua del mercado

a:

> **predicción de eventos discretos recurrentes**


# 🧩 Resumen final

Este diseño:

- convierte tu hipótesis en algo **aprendible** 

- elimina ambigüedad del target 

- alinea datos con el evento real 

- separa “si ocurre” de “cómo ocurre” 


## En una línea

> Dejas de predecir “el mercado en general” y pasas a predecir “qué va a pasar en esa hora específica, dado el contexto previo”.

# 🧠 1. Qué significa “movimiento claro”

No es solo “se movió X%”.

Es:

> **movimiento direccional dominante, limpio y explotable dentro de la ventana**


# 🔴 2. Variables base (dentro de la ventana 20:00–21:00)

Para cada ventana calculas:

```
max\_up   = (máximo\_high - open\_ventana) / open\_ventana

max\_down = (open\_ventana - mínimo\_low) / open\_ventana
```

Y opcionalmente:

```
close\_move = (close\_ventana - open\_ventana) / open\_ventana
```


# 🎯 3. Definición de “movimiento claro”

Se basa en **3 condiciones simultáneas**:


## ✔️ (1) Magnitud suficiente

```
max(max\_up, max\_down) ≥ threshold
```

Ejemplo típico en oro:

- 0.3% – 0.8% (dependiendo timeframe) 


## ✔️ (2) Dominancia direccional

```
dominancia = |max\_up - max\_down|
```

Debe cumplirse:

```
dominancia ≥ α \* threshold
```

Ejemplo:

- α = 0.5 

👉 Evita casos donde sube y baja casi igual.


## ✔️ (3) Coherencia de cierre (opcional pero potente)

Para LONG:

```
close\_ventana \> open\_ventana
```

Para SHORT:

```
close\_ventana \< open\_ventana
```

👉 Filtra movimientos caóticos con reversión.


# 🔵 4. Clasificación final

## LONG (1)

```
max\_up ≥ threshold

AND max\_up \> max\_down

AND dominancia suficiente

AND (opcional) cierre confirma
```


## SHORT (2)

```
max\_down ≥ threshold

AND max\_down \> max\_up

AND dominancia suficiente

AND (opcional) cierre confirma
```


## NO TRADE (0)

Todo lo demás:

- no alcanza magnitud 

- o no hay dominancia 

- o es caótico 


# ⚠️ 5. Casos eliminados (CRÍTICO)

Estos NO deben entrar al entrenamiento:

```
max\_up ≥ threshold

AND max\_down ≥ threshold

AND dominancia baja
```

👉 Estos son los que destruyen la “confianza” del modelo.


# 🧪 6. Intuición de mercado (por qué esto funciona)

En oro, en ventanas como esa:

- hay momentos de **expansión direccional** 

- y momentos de **liquidez / barrido en ambos lados** 

Tu definición separa:

| **Tipo de movimiento** | **Qué haces** |
| :-: | :-: |
| limpio direccional | lo usas |
| lateral / barrido | lo descartas |
| débil | lo ignoras |


# 📊 7. Resultado en el dataset

Después de aplicar esto:

- menos samples 

- pero mucho más consistentes 

👉 Esto aumenta:

- separabilidad 

- estabilidad 

- “confianza” del output 


# 📌 8. Resumen en una línea

> “Movimiento claro” = magnitud suficiente + dirección dominante + (opcional) confirmación de cierre, excluyendo casos bidireccionales.

# 🧠 1. Contexto real del oro en esa hora

En ese horario:

- liquidez media-baja (post NY, pre Asia activa) 

- movimientos suelen ser: 

  - **expansiones cortas pero limpias** 

  - o **barridos en ambos lados (fake moves)** 

👉 Esto implica:

- thresholds **ni muy bajos (ruido)** 

- ni muy altos (pierdes setups) 


# 🎯 2. Threshold de magnitud (CLAVE)

## ✔️ Recomendación base

```
threshold = 0.004 → 0.006  (0.4% – 0.6%)
```


## 🔍 Traducción práctica en oro

Si XAUUSD está en 2000:

- 0.4% ≈ $8 

- 0.6% ≈ $12 

👉 Ese rango captura:

- movimientos reales de sesión 

- evita micro-ruido 


## 🔴 Ajuste fino

- Si ves demasiados trades → sube a 0.5–0.6% 

- Si ves pocos → baja a 0.3–0.4% 


# ⚖️ 3. Dominancia direccional

## ✔️ Definición

```
dominancia = |max\_up - max\_down|
```


## ✔️ Umbral recomendado

```
dominancia ≥ 0.5 \* threshold
```


## 🔍 Interpretación

Si threshold = 0.5%:

- dominancia mínima = 0.25% 

👉 Esto elimina:

- velas tipo “spike up + spike down” 

- sesiones de liquidez sucia 


# 🔵 4. Confirmación de cierre

## ✔️ Recomendación: USARLA

Para LONG:

```
close \> open + 0.2 \* threshold
```

Para SHORT:

```
close \< open - 0.2 \* threshold
```


## 🔍 Por qué no solo close \> open

Porque:

- el oro hace muchos fakeouts 

- necesitas “convicción”, no solo dirección 


# ⚠️ 5. Filtro anti-caos (muy importante)

Eliminar completamente si:

```
max\_up ≥ threshold

AND max\_down ≥ threshold

AND dominancia \< 0.5 \* threshold
```

👉 Este es el filtro que más mejora la calidad del dataset.


# 📊 6. Resultado esperado en datos

Con estos valores:

- ~60–75% → NO TRADE 

- ~12–20% → LONG 

- ~12–20% → SHORT 

👉 Esto es sano.

Si ves:

- demasiados LONG/SHORT → threshold bajo 

- casi todo NO TRADE → threshold alto 


# 🧠 7. Insight clave (importante)

En oro, en esa hora:

> No ganas por predecir dirección siempre  
ganas por **filtrar cuándo vale la pena participar**


# 📌 8. Configuración final recomendada

```
threshold = 0.5% (0.005)

dominancia ≥ 0.5 \* threshold

confirmación cierre ≥ 0.2 \* threshold

eliminar casos bidireccionales fuertes
```


# 🧩 9. En una línea

> Estás definiendo “movimiento claro” como: expansión suficiente + dirección dominante + continuidad, excluyendo liquidez caótica.

