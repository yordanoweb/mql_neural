import pandas as pd
import numpy as np
import sys
import os
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
from indicators import calculate_rsi  as rsi

# --- CONFIGURACIÓN ---
if len(sys.argv) < 2:
    print("Usage: python train_onnx_from_csv.py <csv_file>")
    print("Example: python train_onnx_from_csv.py eurusd_m15_2024.csv")
    sys.exit(1)

csv_file = sys.argv[1]
if not os.path.exists(csv_file):
    print(f"Error: File '{csv_file}' not found")
    sys.exit(1)

# Generate output filename: same basename as CSV but with .onnx extension
output_filename = Path(csv_file).stem + ".onnx"
print(f"--- ENTRENAMIENTO RÁPIDO ---")
print(f"Cargando tasas desde: {csv_file}")
print(f"ONNX de salida será: {output_filename}")

def calculate_rsi(series, period=14):
    """Calculate RSI using indicators module, compatible with pandas Series"""
    rsi_list = rsi(series.values.tolist(), period)
    return pd.Series(rsi_list, index=series.index)

# 1. CARGA DE DATOS DESDE CSV
df = pd.read_csv(csv_file)
print(f"Registros cargados: {len(df)}")

# Infer pip unit from data (optional, or set based on symbol detection if available)
# If symbol info is not available, we'll use a reasonable default
pip_unit = 0.0001  # Default for most pairs; could be refined if symbol is known

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

# 4. EXPORTAR CON NOMBRE BASADO EN CSV
initial_type = [('float_input', FloatTensorType([None, 60]))]
onx = convert_sklearn(model, initial_types=initial_type, options={type(model): {'zipmap': False}})

with open(output_filename, "wb") as f:
    f.write(onx.SerializeToString())

print(f"Modelo guardado en: {output_filename}")
print(f"--- PROCESO COMPLETADO ---")
