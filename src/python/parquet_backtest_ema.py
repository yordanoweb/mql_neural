import pandas as pd
import vectorbt as vbt
import argparse


def run_backtest(parquet_path, fast_window, slow_window, rsi_window, rsi_threshold):
    # 1. Cargar datos
    df = pd.read_parquet(parquet_path)

    # limpiar columnas innecesarias
    df = df.drop(columns=["time"], errors="ignore")

    close = df["close"]

    # 2. Indicadores
    fast_ema = vbt.MA.run(close, window=fast_window)
    slow_ema = vbt.MA.run(close, window=slow_window)

    # 🔥 NUEVO: RSI
    rsi = vbt.RSI.run(close, window=rsi_window)

    # 3. Señales (EMA + RSI)
    entries = (
        fast_ema.ma_crossed_above(slow_ema)
        & (rsi.rsi > rsi_threshold)
    )

    exits = fast_ema.ma_crossed_below(slow_ema)

    # 4. Backtest
    pf = vbt.Portfolio.from_signals(
        close,
        entries,
        exits,
        init_cash=10000
    )

    # 5. Resultados
    print("\n=== RESULTADOS ===")
    print(f"Archivo: {parquet_path}")
    print(f"EMA rápida: {fast_window} | EMA lenta: {slow_window}")
    print(f"RSI window: {rsi_window} | threshold: {rsi_threshold}")
    print(pf.stats())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Backtest EMA + RSI desde Parquet")

    parser.add_argument("--data", required=True, help="Ruta al archivo Parquet")
    parser.add_argument("--fast", type=int, default=10, help="Periodo EMA rápida")
    parser.add_argument("--slow", type=int, default=30, help="Periodo EMA lenta")
    parser.add_argument("--rsi_window", type=int, default=14, help="Periodo RSI")
    parser.add_argument("--rsi_threshold", type=float, default=55, help="Filtro RSI (ej: >55)")

    args = parser.parse_args()

    run_backtest(
        parquet_path=args.data,
        fast_window=args.fast,
        slow_window=args.slow,
        rsi_window=args.rsi_window,
        rsi_threshold=args.rsi_threshold
    )
