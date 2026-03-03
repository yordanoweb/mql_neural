import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import sys
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

# --- CONFIGURACIÓN ---
symbol = sys.argv[1] if len(sys.argv) > 1 else "EURUSD"
print(f"--- ENTRENAMIENTO RÁPIDO PARA: {symbol} ---")

def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

# 1. OBTENCIÓN DE DATOS
if not mt5.initialize():
    quit()

candles_count = 50000 
rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M15, 0, candles_count)
mt5.shutdown()

df = pd.DataFrame(rates)
pip_unit = 0.01 if "JPY" in symbol else 0.0001

df['feat_body'] = (df['close'] - df['open']) / pip_unit
df['feat_range'] = (df['high'] - df['low']) / pip_unit
df['feat_rsi'] = calculate_rsi(df['close'], 14) / 100.0
df['target'] = (df['close'].shift(-1) > df['close']).astype(int)
df.dropna(inplace=True)

# 2. PREPARAR VENTANAS (60 entradas)
window = 20
X, y = [], []
features = ['feat_body', 'feat_range', 'feat_rsi']

for i in range(window, len(df) - 1):
    window_data = df[features].iloc[i-window:i].values.flatten()
    X.append(window_data)
    y.append(df['target'].iloc[i])

X = np.array(X).astype(np.float32)
y = np.array(y)

# 3. OPTIMIZACIÓN RÁPIDA (Solo 10 iteraciones)
print("Buscando configuración eficiente (Random Search)...")
param_dist = {
    'n_estimators': [100, 150, 200],
    'max_depth': [5, 8, 12],
    'min_samples_leaf': [1, 5]
}

# TimeSeriesSplit con 2 pliegues para velocidad
tscv = TimeSeriesSplit(n_splits=2)

search = RandomizedSearchCV(
    RandomForestClassifier(random_state=42),
    param_distributions=param_dist,
    n_iter=5, # Solo prueba 5 combinaciones al azar (muy rápido)
    cv=tscv,
    scoring='accuracy',
    n_jobs=-1
)

search.fit(X, y)
model = search.best_estimator_
print(f"Mejor configuración: {search.best_params_}")

# 4. EXPORTAR SIEMPRE COMO model_m15.onnx
output_filename = "model_m15.onnx"
initial_type = [('float_input', FloatTensorType([None, 60]))]
onx = convert_sklearn(model, initial_types=initial_type, options={type(model): {'zipmap': False}})

with open(output_filename, "wb") as f:
    f.write(onx.SerializeToString())

print(f"--- PROCESO COMPLETADO EN SEGUNDOS ---")
