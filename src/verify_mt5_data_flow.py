import MetaTrader5 as mt5
import numpy as np

def test_data_extraction():
    # 1. Intentar conexión (usa path si es necesario, pero si ya te funcionó antes, empty está bien)
    if not mt5.initialize():
        print(f"Error de inicialización: {mt5.last_error()}")
        return

    print("--- Conexión establecida ---")

    # 2. Verificar si el símbolo está disponible en el Market Watch
    symbol = "EURUSD" # Cambia esto si usas otro (ej. "GBPUSD")
    symbol_info = mt5.symbol_info(symbol)
    
    if symbol_info is None:
        print(f"Símbolo {symbol} no encontrado. Intentando seleccionarlo...")
        mt5.symbol_select(symbol, True)
        symbol_info = mt5.symbol_info(symbol)

    if not symbol_info.visible:
        print(f"Error: El símbolo {symbol} no es visible en el Market Watch de Wine.")
        mt5.shutdown()
        return

    # 3. EXTRAER DATOS (La prueba de fuego del flujo de memoria)
    # Pedimos las últimas 10 velas de 1 Hora
    rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_H1, 0, 10)

    if rates is None or len(rates) == 0:
        print(f"Error al extraer datos: {mt5.last_error()}")
    else:
        # Convertir a un structured array de NumPy para verificar integridad
        data_array = np.array(rates)
        print(f"\n¡Éxito! Datos recibidos para {symbol}:")
        print(f"Cantidad de velas: {len(data_array)}")
        print(f"Estructura del primer registro (Open, High, Low, Close):")
        print(data_array) # Indices de OHLC en el struct de MT5
        
        # Verificamos que no sean ceros (problema común de sincronización en Wine)
        if data_array['close'].any():
            print("\nIntegridad de datos: OK (Los precios no son nulos)")

    mt5.shutdown()

if __name__ == "__main__":
    test_data_extraction()
