import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

if not mt5.initialize():
    quit()

# Aumentamos a 5000 barras para que el modelo tenga más ejemplos reales
rates = mt5.copy_rates_from_pos("EURUSD", mt5.TIMEFRAME_H1, 0, 5000)
mt5.shutdown()
df = pd.DataFrame(rates)

# --- NORMALIZACIÓN A PIPS ---
# Suponiendo 5 decimales (0.00010 = 1 pip)
pip_unit = 0.0001 

df['feat_body'] = (df['close'] - df['open']) / pip_unit
df['feat_range'] = (df['high'] - df['low']) / pip_unit
df['feat_rsi'] = calculate_rsi(df['close'], 14) / 100.0

# TARGET CORRECTO: ¿La vela que SIGUE a mi ventana subirá?
df['target'] = (df['close'].shift(-1) > df['close']).astype(int)
df.dropna(inplace=True)

window = 10
X, y = [], []
features = ['feat_body', 'feat_range', 'feat_rsi']

for i in range(window, len(df) - 1): # -1 para asegurar que existe el shift(-1)
    # X: Velas de i-window hasta i-1 (10 velas de pasado)
    window_data = df[features].iloc[i-window:i].values.flatten()
    X.append(window_data)
    
    # y: El target de la vela i (el futuro inmediato de esa ventana)
    y.append(df['target'].iloc[i]) # <--- CORREGIDO: i en lugar de i-1

X = np.array(X).astype(np.float32)
y = np.array(y)

# Entrenamos con un modelo un poco más robusto
model = RandomForestClassifier(n_estimators=100, max_depth=7, random_state=42)
model.fit(X, y)

# Exportar
initial_type = [('float_input', FloatTensorType([None, 30]))]
onx = convert_sklearn(model, initial_types=initial_type, options={type(model): {'zipmap': False}})

with open("model_multi.onnx", "wb") as f:
    f.write(onx.SerializeToString())

print("¡Modelo exportado con lógica de futuro real!")
