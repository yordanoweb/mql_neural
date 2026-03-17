import pandas as pd
import argparse
import os

def csv_to_parquet(
    input_path,
    output_path,
    timeframe="5min",
    tz="UTC",
    compression="zstd"
):
    # 1. Leer CSV
    df = pd.read_csv(input_path)

    # 2. Normalizar nombres de columnas
    df.columns = [c.lower() for c in df.columns]

    expected_cols = ["time", "open", "high", "low", "close", "tick_volume"]
    if not all(col in df.columns for col in expected_cols):
        raise ValueError(f"El CSV debe contener columnas: {expected_cols}")

    # 3. Convertir timestamp → datetime
    # intenta autodetectar si está en segundos o milisegundos
    # parsing universal (funciona en ambos casos)
    df["timestamp"] = pd.to_datetime(df["time"], errors="coerce")

    # 4. Index temporal
    df.set_index("timestamp", inplace=True)

    # 5. Ordenar
    df.sort_index(inplace=True)

    # 6. Convertir tipos numéricos
    for col in ["open", "high", "low", "close", "tick_volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # 7. Eliminar duplicados
    df = df[~df.index.duplicated(keep="first")]

    # 8. Normalizar frecuencia
    df = df.asfreq(timeframe)

    # 9. Rellenar gaps pequeños
    df = df.ffill()

    # 10. Asegurar timezone
    if df.index.tz is None:
        df.index = df.index.tz_localize(tz)
    else:
        df.index = df.index.tz_convert(tz)

    # 11. Crear carpeta si no existe
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # 12. Guardar Parquet
    df.to_parquet(
        output_path,
        engine="pyarrow",
        compression=compression
    )

    print(f"✅ Parquet guardado en: {output_path}")
    print(f"Filas: {len(df)} | Columnas: {list(df.columns)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convertir CSV de velas a Parquet")

    parser.add_argument("--input", required=True, help="Ruta al CSV de entrada")
    parser.add_argument("--output", required=True, help="Ruta del Parquet de salida")
    parser.add_argument("--timeframe", default="5min", help="Frecuencia (ej: 5min)")
    parser.add_argument("--tz", default="UTC", help="Timezone (ej: UTC)")
    parser.add_argument("--compression", default="zstd", help="Compresión: zstd, snappy")

    args = parser.parse_args()

    csv_to_parquet(
        input_path=args.input,
        output_path=args.output,
        timeframe=args.timeframe,
        tz=args.tz,
        compression=args.compression
    )
