import os
import time
import collections
import MetaTrader5 as mt5
import pandas as pd
import ta               # pip install ta

# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------
SYMBOL     = os.getenv("MT5_SYMBOL", "EURUSD")
EMA_SHORT  = 6
EMA_LONG   = 9
MAX_BARS   = max(EMA_SHORT, EMA_LONG) * 3
INTERVAL   = 5.0                     # seconds between processing steps
WINDOW    = 5                         # número de cambios a acumular
# Se muestra la tendencia aunque sea mínima; si prefieres filtrar, pon un valor >0
THRESHOLD = 0.0

# ----------------------------------------------------------------------
# Initialise MT5
# ----------------------------------------------------------------------
if not mt5.initialize():
    print("Failed to initialize MT5, error code =", mt5.last_error())
    quit()

print(f"MT5 initialized – processing {SYMBOL} every {INTERVAL}s")

# ----------------------------------------------------------------------
# Store recent mid‑prices
# ----------------------------------------------------------------------
price_history = collections.deque(maxlen=MAX_BARS)

def compute_emas():
    series = pd.Series(list(price_history))
    ema6 = ta.trend.EMAIndicator(series, window=EMA_SHORT).ema_indicator().iloc[-1]
    ema9 = ta.trend.EMAIndicator(series, window=EMA_LONG).ema_indicator().iloc[-1]
    return ema6, ema9

# ----------------------------------------------------------------------
# Tracking variables
# ----------------------------------------------------------------------
prev_delta = None                                 # Δ del ciclo anterior
delta_changes = collections.deque(maxlen=WINDOW) # últimos 5 cambios

# ----------------------------------------------------------------------
# Main loop – act only each INTERVAL seconds
# ----------------------------------------------------------------------
last_tick_time = None
last_process   = time.time() - INTERVAL   # force immediate first run

try:
    while True:
        tick = mt5.symbol_info_tick(SYMBOL)

        if tick is None:
            time.sleep(0.1)
            continue

        if tick.time != last_tick_time:
            last_tick_time = tick.time

        now = time.time()
        if now - last_process >= INTERVAL:
            last_process = now

            # Mid‑price from the latest tick
            price = (tick.bid + tick.ask) / 2.0
            price_history.append(price)

            ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(tick.time))

            if len(price_history) >= EMA_LONG:
                ema6, ema9 = compute_emas()
                relation = (
                    "above EMA‑9"
                    if price > ema9
                    else "below EMA‑9"
                    if price < ema9
                    else "exactly at EMA‑9"
                )

                # ---- distancia al EMA‑9 y cambio respecto al ciclo anterior ----
                delta = price - ema9
                if prev_delta is not None:
                    change = delta - prev_delta
                    delta_changes.append(change)   # acumular el cambio
                else:
                    change = None
                prev_delta = delta

                # ---- determinar tendencia cuando tengamos WINDOW cambios ----
                direction_msg = ""
                if len(delta_changes) == WINDOW:
                    avg_change = sum(delta_changes) / WINDOW
                    # Siempre mostramos la tendencia (THRESHOLD = 0)
                    direction = "↘ acercándose" if avg_change < 0 else "↗ alejándose"
                    direction_msg = f" → tendencia media: {abs(avg_change):.5f} {direction}"
                    # Vaciar para la siguiente ventana de 5 cambios
                    delta_changes.clear()

                # ---- salida ----
                print(
                    f"[{ts}] {SYMBOL} price={price:.5f} EMA6={ema6:.5f} EMA9={ema9:.5f} "
                    f"→ price is {relation} | Δ={delta:.5f}{direction_msg}"
                )
                print()   # espacio para dar sensación de progreso
            else:
                print(f"[{ts}] {SYMBOL} price={price:.5f} (building EMA history…)\n")

        time.sleep(0.05)

except KeyboardInterrupt:
    print("\nScript stopped by user.")

mt5.shutdown()
