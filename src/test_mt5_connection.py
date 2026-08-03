import MetaTrader5 as mt5

# IMPORTANTE: Usa la ruta de Windows dentro de tu prefijo
path = "C:\\Program Files\\MetaTrader 5\\terminal64.exe"

if not mt5.initialize(path=path):
    print("Fallo al conectar. Error:", mt5.last_error())
else:
    print("¡Conexión exitosa!")
    print("Versión:", mt5.version())
    mt5.shutdown()
