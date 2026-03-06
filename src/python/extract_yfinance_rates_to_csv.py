"""
Extract OHLC rates from MetaTrader 5 to CSV format.

Timeframe and date range are supplied via command-line parameters.
No processing, transformations, or calculations are applied—only raw rates are extracted.

Usage:
    python extract_rates_to_csv.py EURUSD M15 2024-01-01 2024-12-31 rates.csv
    python extract_rates_to_csv.py GBPUSD H1 2024-06-01 2024-06-30 gbp_rates.csv
"""

import sys
import yfinance as yf
import pandas as pd
from datetime import datetime
import os


def parse_args():
    """Parse command-line arguments."""
    if len(sys.argv) < 5:
        print(__doc__)
        print("\nArguments:")
        print("  SYMBOL       - Trading pair (e.g., EURUSD, GBPUSD, USDJPY)")
        print("  TIMEFRAME    - Candle timeframe (M1, M5, M15, M30, H1, H4, D1, W1, MN1)")
        print("  START_DATE   - Start date in YYYY-MM-DD format")
        print("  END_DATE     - End date in YYYY-MM-DD format")
        print("  OUTPUT_FILE  - Output CSV filename (optional, default: rates.csv)")
        sys.exit(1)

    symbol = sys.argv[1].upper()
    timeframe_str = sys.argv[2].upper()
    start_date_str = sys.argv[3]
    end_date_str = sys.argv[4]
    output_file = sys.argv[5] if len(sys.argv) > 5 else "rates.csv"

    return symbol, timeframe_str, start_date_str, end_date_str, output_file


def get_timeframe(timeframe_str):
    """
    Map timeframe string to MT5 constant.
    
    Valid values: M1, M5, M15, M30, H1, H4, D1, W1, MN1
    """
    timeframe_map = {
        'M1': '1m',
        'M5': '5m',
        'M15': '15m',
        'M30': '30m',
        'H1': '1h',
        'H4': '4h',
        'D1': '1d',
    }
    
    if timeframe_str not in timeframe_map:
        raise ValueError(f"Invalid timeframe: {timeframe_str}. Valid values: {', '.join(timeframe_map.keys())}")
    
    return timeframe_map[timeframe_str]


def parse_date(date_str):
    """Parse date string in YYYY-MM-DD format to datetime object."""
    try:
        return datetime.strptime(date_str, "%Y-%m-%d")
    except ValueError:
        raise ValueError(f"Invalid date format: {date_str}. Use YYYY-MM-DD")


def extract_rates(symbol, timeframe, start_date, end_date, output_file):
    """
    Extract rates from MetaTrader 5 and save to CSV.
    
    Args:
        symbol: Trading pair symbol (e.g., 'EURUSD')
        timeframe: MT5 timeframe constant (e.g., mt5.TIMEFRAME_M15)
        start_date: datetime object for start date
        end_date: datetime object for end date
        output_file: Path to output CSV file
    """
    
    try:
        print(f"Connecting to YAHOO Finance service...")
        print(f"Fetching {symbol} {timeframe} rates from {start_date.date()} to {end_date.date()}...")
        
        # Get rates from the specified date range
        rates = yf.download(symbol, start=start_date, 
                            end=end_date, interval=timeframe,
                            multi_level_index=False)
        
        if rates is None or len(rates) == 0:
            raise ValueError(f"No rates found for {symbol} in the specified range.")
        
        print(f"Retrieved {len(rates)} candles.")
        
        # Reset index to make Date a column (it's currently the index)
        df = rates.reset_index()

        # Rename columns for clarity and standardization
        # yfinance returns: Date, Open, High, Low, Close, Adj Close, Volume
        df = df.rename(columns={
            'Date': 'time',
            'Open': 'open',
            'High': 'high',
            'Low': 'low',
            'Close': 'close',
            'Volume': 'volume'
        })

        # Save to CSV
        df.to_csv(output_file, index=False)
        print(f"Rates saved to {output_file}")
        print(f"File size: {os.path.getsize(output_file) / 1024:.2f} KB")

    except Exception as e:
        raise Exception(f"Failed to extract rates from YAHOO service: {e}")


def main():
    """Main entry point."""
    try:
        symbol, timeframe_str, start_date_str, end_date_str, output_file = parse_args()
        
        # Parse arguments
        timeframe = get_timeframe(timeframe_str)
        start_date = parse_date(start_date_str)
        end_date = parse_date(end_date_str)
        
        if start_date > end_date:
            raise ValueError("Start date must be before end date.")
        
        # Extract rates
        extract_rates(symbol, timeframe, start_date, end_date, output_file)
        
        print("\n✓ Extraction completed successfully!")
        
    except Exception as e:
        print(f"\n✗ Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
