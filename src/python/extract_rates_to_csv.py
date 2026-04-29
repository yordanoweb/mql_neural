"""
Extract OHLC rates from MetaTrader 5 to CSV format.

Timeframe and date range are supplied via command-line parameters.
No processing, transformations, or calculations are applied—only raw rates are extracted.

Usage:
    python extract_rates_to_csv.py SYMBOL TIMEFRAME DAYS_BACK
    
Example:
    python extract_rates_to_csv.py EURUSD M15 300
"""

import sys
import MetaTrader5 as mt5
import pandas as pd
from datetime import datetime, timedelta
import os


def parse_args():
    """Parse command-line arguments."""
    if len(sys.argv) < 4:
        print(__doc__)
        print("\nArguments:")
        print("  SYMBOL       - Trading pair (e.g., EURUSD, GBPUSD, USDJPY)")
        print("  TIMEFRAME    - Candle timeframe (M1, M5, M15, M30, H1, H4, D1, W1, MN1)")
        print("  DAYS_BACK    - Number of days to fetch backwards from current day")
        sys.exit(1)

    symbol = sys.argv[1].upper()
    timeframe_str = sys.argv[2].upper()
    
    try:
        days_back = int(sys.argv[3])
    except ValueError:
        raise ValueError(f"Invalid DAYS_BACK: {sys.argv[3]}. Must be an integer.")
    
    if days_back <= 0:
        raise ValueError(f"DAYS_BACK must be positive, got {days_back}")

    return symbol, timeframe_str, days_back


def get_timeframe(timeframe_str):
    """
    Map timeframe string to MT5 constant.
    
    Valid values: M1, M5, M15, M30, H1, H4, D1, W1, MN1
    """
    timeframe_map = {
        'M1': mt5.TIMEFRAME_M1,
        'M5': mt5.TIMEFRAME_M5,
        'M15': mt5.TIMEFRAME_M15,
        'M30': mt5.TIMEFRAME_M30,
        'H1': mt5.TIMEFRAME_H1,
        'H4': mt5.TIMEFRAME_H4,
        'D1': mt5.TIMEFRAME_D1,
        'W1': mt5.TIMEFRAME_W1,
        'MN1': mt5.TIMEFRAME_MN1,
    }
    
    if timeframe_str not in timeframe_map:
        raise ValueError(f"Invalid timeframe: {timeframe_str}. Valid values: {', '.join(timeframe_map.keys())}")
    
    return timeframe_map[timeframe_str]


def extract_rates(symbol, timeframe, timeframe_str, days_back):
    """
    Extract rates from MetaTrader 5 and save to split CSV files.
    
    Args:
        symbol: Trading pair symbol (e.g., 'EURUSD')
        timeframe: MT5 timeframe constant (e.g., mt5.TIMEFRAME_M15)
        timeframe_str: Timeframe string (e.g., 'M15')
        days_back: Number of days to fetch backwards from current day
    """
    
    # Calculate date range
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days_back)
    
    # Initialize MetaTrader 5
    if not mt5.initialize():
        raise ConnectionError("Failed to initialize MetaTrader 5. Ensure MT5 is running.")
    
    try:
        print(f"Connecting to MetaTrader 5...")
        print(f"Fetching {symbol} {timeframe_str} rates from {start_date.date()} to {end_date.date()} ({days_back} days)...")
        
        # Get rates from the specified date range
        rates = mt5.copy_rates_range(symbol, timeframe, start_date, end_date)
        
        if rates is None or len(rates) == 0:
            raise ValueError(f"No rates found for {symbol} in the specified range.")
        
        print(f"Retrieved {len(rates)} candles.")
        
        # Convert to DataFrame
        df = pd.DataFrame(rates)
        
        # Rename columns for clarity
        # The mt5.copy_rates_range returns structured array with:
        # time, open, high, low, close, tick_volume, spread, real_volume
        df.columns = ['time', 'open', 'high', 'low', 'close', 'tick_volume', 'spread', 'real_volume']
        
        # Convert time from unix timestamp to datetime
        df['time'] = pd.to_datetime(df['time'], unit='s')
        
        # Split data into two parts
        split_index = len(df) // 2
        df_part1 = df.iloc[:split_index]
        df_part2 = df.iloc[split_index:]
        
        # Generate output filenames in csv/ directory
        csv_dir = "csv"
        os.makedirs(csv_dir, exist_ok=True)
        base_filename = f"{symbol}_{timeframe_str}"
        part1_file = os.path.join(csv_dir, f"{base_filename}_part1.csv")
        part2_file = os.path.join(csv_dir, f"{base_filename}_part2.csv")
        
        # Save to CSV files
        df_part1.to_csv(part1_file, index=False)
        df_part2.to_csv(part2_file, index=False)
        
        print(f"Part 1 saved to {part1_file} ({len(df_part1)} candles)")
        print(f"Part 2 saved to {part2_file} ({len(df_part2)} candles)")
        print(f"Total file size: {(os.path.getsize(part1_file) + os.path.getsize(part2_file)) / 1024:.2f} KB")
        
    finally:
        # Always shutdown MT5 connection
        mt5.shutdown()


def main():
    """Main entry point."""
    try:
        symbol, timeframe_str, days_back = parse_args()
        
        # Parse arguments
        timeframe = get_timeframe(timeframe_str)
        
        # Extract rates
        extract_rates(symbol, timeframe, timeframe_str, days_back)
        
        print("\n✓ Extraction completed successfully!")
        
    except Exception as e:
        print(f"\n✗ Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
