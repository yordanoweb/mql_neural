"""
Entrenamiento de modelo ONNX con 3 features financieras según especificación:
- body: ln(close/open)               (retorno logístico normalizado)
- range: (high - low) / open         (rango normalizado, volatilidad intrabarra)
- volume_ratio: tick_volume / SMA(tick_volume, period)  (volumen relativo)

Uso:
    python train_onnx_model.py --input_csv data.csv --output_dir ./models --volume_sma_period 20 --window 20 --future 5 --n_iter 10
"""

import argparse
import sys
import os
import logging
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
import onnx
import onnxruntime as ort

# Configuración de logs colorizados
try:
    from colorama import init, Fore, Style
    init(autoreset=True)
except ImportError:
    # Fallback si colorama no está instalado
    class Fore:
        RED = '\033[91m'
        GREEN = '\033[92m'
        YELLOW = '\033[93m'
        BLUE = '\033[94m'
        MAGENTA = '\033[95m'
        CYAN = '\033[96m'
        RESET = '\033[0m'
    Style = type('Style', (), {'BRIGHT': '\033[1m', 'RESET_ALL': '\033[0m'})()

def setup_logger():
    logger = logging.getLogger('ONNX_Trainer')
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        ch.setFormatter(formatter)
        logger.addHandler(ch)
    return logger

logger = setup_logger()

def colorize(text, color=Fore.WHITE, bright=False):
    brightness = Style.BRIGHT if bright else ''
    return f"{brightness}{color}{text}{Style.RESET_ALL}"

def log_info(msg, color=Fore.CYAN, bright=False):
    logger.info(colorize(msg, color))

def log_success(msg):
    logger.info(colorize(msg, Fore.GREEN, bright=True))

def log_warning(msg):
    logger.warning(colorize(msg, Fore.YELLOW))

def log_error(msg):
    logger.error(colorize(msg, Fore.RED, bright=True))

def load_data(csv_path):
    """Carga CSV y ordena por timestamp si existe"""
    log_info(f"Cargando datos desde {csv_path}...")
    df = pd.read_csv(csv_path)
    
    # Identificar columna de tiempo
    time_col = None
    for col in ['timestamp', 'date', 'datetime', 'time']:
        if col in df.columns:
            time_col = col
            break
    if time_col:
        df[time_col] = pd.to_datetime(df[time_col])
        df = df.sort_values(time_col).reset_index(drop=True)
        log_info(f"Ordenado por {time_col}")
    else:
        log_warning("No se encontró columna de tiempo. Se asume orden secuencial correcto.")
    
    # Verificar columnas necesarias
    required = ['open', 'high', 'low', 'close', 'tick_volume']
    missing = [c for c in required if c not in df.columns]
    if missing:
        log_error(f"Faltan columnas requeridas: {missing}")
        sys.exit(1)
    
    log_success(f"Datos cargados: {len(df)} filas, columnas: {list(df.columns)}")
    return df

def compute_features(df, volume_sma_period):
    """
    Calcula las tres features:
    - body = ln(close/open)
    - range = (high - low) / open
    - volume_ratio = tick_volume / SMA(tick_volume, volume_sma_period)
    """
    log_info("Calculando features (body, range, volume_ratio)...")
    
    # Body: retorno logístico normalizado
    df['body'] = np.log(df['close'] / df['open'])
    
    # Range: rango normalizado por apertura
    df['range'] = (df['high'] - df['low']) / df['open']
    
    # Volume ratio: tick_volume / media móvil simple
    volume_sma = df['tick_volume'].rolling(window=volume_sma_period, min_periods=1).mean()
    df['volume_ratio'] = df['tick_volume'] / volume_sma
    
    # Eliminar filas con NaN generados (principalmente al inicio de la SMA)
    initial_len = len(df)
    df = df.dropna().reset_index(drop=True)
    log_info(f"Features calculadas. Filas después de dropna: {len(df)} (eliminadas {initial_len - len(df)})")
    
    return df

def create_sequences(df, window, future, feature_cols):
    """
    Crea secuencias X (window x features) y etiquetas y (binaria: 1 si precio futuro > close actual)
    """
    log_info(f"Creando secuencias con ventana={window}, horizonte={future}...")
    data = df[feature_cols].values
    close_prices = df['close'].values
    
    X, y = [], []
    for i in range(len(data) - window - future + 1):
        X.append(data[i:i+window, :])   # forma (window, n_features)
        future_close = close_prices[i + window + future - 1]
        current_close = close_prices[i + window - 1]
        y.append(1 if future_close > current_close else 0)
    
    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int32)
    
    log_success(f"Secuencias generadas: X shape {X.shape}, y shape {y.shape}")
    log_info(f"Distribución de clases: 0={np.sum(y==0)}, 1={np.sum(y==1)} (ratio: {np.mean(y):.3f})")
    return X, y

def train_model(X, y, n_iter):
    """Entrena RandomForest con RandomizedSearchCV y devuelve el mejor modelo y el scaler"""
    log_info("Iniciando búsqueda aleatoria de hiperparámetros...")
    
    # Dividir en entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    log_info(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")
    
    nsamples, nwindow, nfeat = X_train.shape
    # Aplanar directamente: cada muestra es un vector de longitud nwindow * nfeat
    X_train_flat = X_train.reshape(nsamples, -1)  # (nsamples, nwindow * nfeat)
    
    # Escalar las muestras aplanadas
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_flat)
    
    # Preparar test
    nsamples_test, _, _ = X_test.shape
    X_test_flat = X_test.reshape(nsamples_test, -1)
    X_test_scaled = scaler.transform(X_test_flat)
    
    # Modelo y búsqueda
    rf = RandomForestClassifier(random_state=42, n_jobs=-1)
    param_dist = {
        'n_estimators': [50, 100, 200],
        'max_depth': [5, 10, 15, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2', None]
    }
    
    random_search = RandomizedSearchCV(
        rf, param_distributions=param_dist,
        n_iter=n_iter, cv=3, scoring='accuracy',
        random_state=42, n_jobs=4, verbose=7
    )
    
    log_info(f"Ejecutando RandomizedSearchCV con {n_iter} iteraciones...")
    random_search.fit(X_train_scaled, y_train)
    
    best_model = random_search.best_estimator_
    best_params = random_search.best_params_
    best_cv_score = random_search.best_score_
    
    log_success(f"Mejores parámetros: {best_params}")
    log_success(f"Precisión media en validación cruzada: {best_cv_score:.4f}")
    
    # Evaluación en test
    y_pred = best_model.predict(X_test_scaled)
    test_acc = accuracy_score(y_test, y_pred)
    log_success(f"Precisión en conjunto de prueba: {test_acc:.4f}")
    log_info("Reporte de clasificación en test:")
    print(colorize(classification_report(y_test, y_pred), Fore.MAGENTA))
    
    input_dim = (X_train_flat.shape[1],)   # (60,)
    return best_model, scaler, input_dim

def export_to_onnx(model, scaler, input_dim, output_path):
    """Exporta pipeline (scaler + modelo) a ONNX"""
    log_info(f"Exportando modelo a ONNX en {output_path}...")
    
    from sklearn.pipeline import Pipeline
    pipeline = Pipeline([
        ('scaler', scaler),
        ('classifier', model)
    ])
    
    initial_type = [('float_input', FloatTensorType([None, input_dim[0]]))]
    onnx_model = convert_sklearn(pipeline, initial_types=initial_type, target_opset=12)
    
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    with open(output_path, 'wb') as f:
        f.write(onnx_model.SerializeToString())
    
    # Verificación
    onnx.checker.check_model(onnx_model)
    log_success("Modelo ONNX validado correctamente.")
    
    # Prueba de inferencia
    sess = ort.InferenceSession(output_path)
    dummy_input = np.random.randn(1, input_dim[0]).astype(np.float32)
    outputs = sess.run(None, {'float_input': dummy_input})
    log_info(f"Prueba de inferencia ONNX exitosa. Salida shape: {outputs[0].shape}")
    
    return output_path

def main():
    parser = argparse.ArgumentParser(description="Entrena modelo ONNX con features: body, range, volume_ratio")
    parser.add_argument("--input_csv", type=str, required=True, help="Ruta al archivo CSV con columnas: open, high, low, close, tick_volume")
    parser.add_argument("--output_dir", type=str, default=".", help="Directorio donde guardar el modelo ONNX")
    parser.add_argument("--volume_sma_period", type=int, default=20, help="Periodo de la media móvil simple para el volumen (default: 20)")
    parser.add_argument("--window", type=int, default=20, help="Ventana de tiempo (número de barras) para cada muestra")
    parser.add_argument("--future", type=int, default=5, help="Número de barras hacia adelante para etiquetar el target")
    parser.add_argument("--n_iter", type=int, default=5, help="Número de iteraciones para RandomizedSearchCV")
    args = parser.parse_args()
    
    log_info("=== INICIO DEL ENTRENAMIENTO ===", Fore.MAGENTA, bright=True)
    log_info(f"Argumentos: {vars(args)}")
    
    # 1. Cargar datos
    df = load_data(args.input_csv)
    
    # 2. Calcular features
    df = compute_features(df, args.volume_sma_period)
    
    # 3. Crear secuencias
    feature_cols = ['body', 'range', 'volume_ratio']
    X, y = create_sequences(df, args.window, args.future, feature_cols)
    
    if len(X) == 0:
        log_error("No hay suficientes datos para crear secuencias. Aumenta el tamaño del CSV o reduce window/future.")
        sys.exit(1)
    
    # 4. Entrenar modelo
    model, scaler, input_dim = train_model(X, y, args.n_iter)
    
    # 5. Exportar a ONNX
    output_filename = f"onnx_model_w{args.window}_f{args.future}_volSMA{args.volume_sma_period}.onnx"
    output_path = os.path.join(args.output_dir, output_filename)
    export_to_onnx(model, scaler, input_dim, output_path)
    
    log_success(f"Modelo guardado en: {output_path}")
    log_info("=== ENTRENAMIENTO COMPLETADO ===", Fore.MAGENTA, bright=True)

if __name__ == "__main__":
    main()