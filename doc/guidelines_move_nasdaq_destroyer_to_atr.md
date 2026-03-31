# 🎯 Diseño del Sistema Basado en ATR

## Features a Calcular:

```python
# Features existentes (normalizadas por ATR):
feat_body = (close - open) / atr
feat_range = (high - low) / atr
feat_rsi = rsi / 100.0
```

## El ATR ya normaliza por volatilidad del mercado

Target Labeling:

```python
# Calcular ATR
atr = ta.volatility.AverageTrueRange(high, low, close, window=atr_period)

for i in range(len(df) - future):
    entry = df['close'].iloc[i]
    current_atr = atr.iloc[i]
    
    # Máximos movimientos en las próximas 'future' velas
    max_upside = (future_high - entry)
    max_downside = (entry - future_low)
    
    # Normalizar por ATR
    max_upside_atr = max_upside / current_atr
    max_downside_atr = max_downside / current_atr
    
    # Etiquetar
    if max_upside_atr >= min_profit_atr and max_upside_atr > max_downside_atr:
        label = 1  # BUY
    elif max_downside_atr >= min_profit_atr and max_downside_atr > max_upside_atr:
        label = -1  # SELL
    else:
        label = 0  # NEUTRAL
```

Parámetros:

```python
--atr_period 14          # Periodo de ATR (default: 14)
--min_profit_atr 1.5     # Mínimo movimiento en múltiplos de ATR (default: 1.5)
```

## 🤔 Decisión Importante: Normalizar Features por ATR

```python
df['atr'] = ta.volatility.AverageTrueRange(...).average_true_range()

df['feat_body'] = (df['close'] - df['open']) / df['atr']
df['feat_range'] = (df['high'] - df['low']) / df['atr']
df['feat_rsi'] = rsi / 100.0  # RSI ya está normalizado [0-100]
```

Ventajas:

✅ Features universales para cualquier símbolo
✅ Body de 0.5 ATR significa lo mismo en NASDAQ que en EUR/USD
✅ El modelo aprende patrones de volatilidad relativa

## Normalizar features por ATR hace el modelo verdaderamente universal:

```python
# NASDAQ @ 15000, ATR=50
body = 25 puntos → 25/50 = 0.5 ATR

# EUR/USD @ 1.0500, ATR=0.0008
body = 0.0004 → 0.0004/0.0008 = 0.5 ATR

# ¡El modelo ve el MISMO patrón!
```

## 📝 Estructura Final del Script

```python
# Parámetros:
--atr_period 14              # Periodo para calcular ATR
--min_profit_atr 1.5         # Target: 1.5× ATR mínimo
--rsi_period 14              # Periodo RSI (independiente)
--window 20                  # Ventana de features
--future 5                   # Velas futuras para label
```

## Ya NO necesitamos:

- --pip_unit
- --min_profit_points

