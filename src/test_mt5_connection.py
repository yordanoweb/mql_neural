# This script has been moved to the python_scripts directory.
import MetaTrader5 as mt5

# IMPORTANT: Use the Windows path inside your prefix
path = "C:\\Program Files\\MetaTrader 5\\terminal64.exe"

if not mt5.initialize(path=path):
    print("Connection failed. Error:", mt5.last_error())
else:
    print("Connection successful!")
    print("Version:", mt5.version())
    mt5.shutdown()
