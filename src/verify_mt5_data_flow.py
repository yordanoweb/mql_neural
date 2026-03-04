import MetaTrader5 as mt5
import numpy as np

def test_data_extraction():
    # 1. Attempt connection (use path if needed; if it worked before, empty is fine)
    if not mt5.initialize():
        print(f"Initialization error: {mt5.last_error()}")
        return

    print("--- Connection established ---")

    # 2. Check if the symbol is available in Market Watch
    symbol = "EURUSD" # Change this if you use another (e.g. "GBPUSD")
    symbol_info = mt5.symbol_info(symbol)
    
    if symbol_info is None:
        print(f"Symbol {symbol} not found. Attempting to select it...")
        mt5.symbol_select(symbol, True)
        symbol_info = mt5.symbol_info(symbol)

    if not symbol_info.visible:
        print(f"Error: Symbol {symbol} is not visible in Wine's Market Watch.")
        mt5.shutdown()
        return

    # 3. EXTRACT DATA (The acid test for memory flow)
    # Request the last 10 one-hour candles
    rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_H1, 0, 10)

    if rates is None or len(rates) == 0:
        print(f"Error extracting data: {mt5.last_error()}")
    else:
        # Convert to a NumPy structured array to verify integrity
        data_array = np.array(rates)
        print(f"\nSuccess! Data received for {symbol}:")
        print(f"Number of candles: {len(data_array)}")
        print(f"Structure of first record (Open, High, Low, Close):")
        print(data_array) # OHLC indices in the MT5 struct
        
        # Check they are not zeros (common sync issue in Wine)
        if data_array['close'].any():
            print("\nData integrity: OK (Prices are non-zero)")

    mt5.shutdown()

if __name__ == "__main__":
    test_data_extraction()
