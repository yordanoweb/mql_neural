import pandas as pd
import vectorbt as vbt
import argparse


def run_backtest(parquet_path, window, buffer, atr_mult):
    # 1. Cargar datos
    df = pd.read_parquet(parquet_path)

    # limpiar columnas innecesarias
    df = df.drop(columns=["time"], errors="ignore")

    close = df["close"]
    high = df["high"]
    low = df["low"]

    # 2. Breakout levels
    rolling_high = high.rolling(window).max().shift(1)
    rolling_low = low.rolling(window).min().shift(1)
    trailing_low = low.rolling(10).min().shift(1)

    # 2.1 Filtro de volatilidad (ATR)
    atr = vbt.ATR.run(high, low, close, window=14)
    atr_mean = atr.atr.rolling(50).mean()
    vol_filter = atr.atr > atr_mean * 1.2

    momentum = close.pct_change(2)

    # 2.2 ATR para trailing stop
    atr_values = atr.atr

    # 3. Señales (🔥 ahora con buffer)
    ema200 = vbt.MA.run(close, 200).ma
    ema_slope = ema200.diff()

    trend_filter = (close > ema200 * 1.001) & (ema_slope > 0)

    entries = (close > rolling_high * (1 + buffer)) & trend_filter & vol_filter

    # duración máxima en velas (ej: 96 = 1 día en M15)
    max_bars = 96

    # contador de barras en posición
    bar_count = entries.cumsum() - entries.cumsum().where(~entries).ffill().fillna(0)

    # salida por tiempo
    time_exit = bar_count >= max_bars

    if atr_mult > 0:
        sl_stop = atr_mult * atr_values / close
        exits = (close < trailing_low) | time_exit
    else:
        sl_stop = None
        exits = (close < trailing_low) | time_exit

    # 4. Backtest
    pf = vbt.Portfolio.from_signals(
        close,
        entries,
        exits,
        sl_stop=sl_stop,
        init_cash=10000,
        fees=0.0002  # 0.02%
    )

    # 5. Resultados
    print("\n=== RESULTADOS ===")
    print(f"Archivo: {parquet_path}")
    print(f"Breakout window: {window}")
    print(f"Buffer: {buffer}")
    print("Filtro ATR: activado")
    print(f"ATR Mult: {atr_mult}")
    print(pf.stats())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Backtest Breakout desde Parquet")

    parser.add_argument("--data", required=True, help="Ruta al archivo Parquet")
    parser.add_argument("--window", type=int, default=20, help="Ventana breakout (ej: 20)")
    parser.add_argument("--buffer", type=float, default=0.0, help="Buffer de ruptura (ej: 0.001 = 0.1%)")
    parser.add_argument("--atr_mult", type=float, default=0.0, help="Multiplicador ATR para trailing stop (0 = desactivado)")

    args = parser.parse_args()

    run_backtest(
        parquet_path=args.data,
        window=args.window,
        buffer=args.buffer,
        atr_mult=args.atr_mult
    )
