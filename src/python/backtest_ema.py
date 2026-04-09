from backtesting import Backtest, Strategy
from backtesting.lib import FractionalBacktest
import talib
import pandas as pd
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Dual EMA Crossover Strategy - Long Only')
    parser.add_argument('--csv', type=str, default="ndx100_rates_h1_2020.csv", help='CSV file path')
    parser.add_argument('--ema-fast', type=int, default=20, help='Fast EMA period')
    parser.add_argument('--trade-start-hour', type=int, default=14, help='Start hour for trading (0-23)')
    parser.add_argument('--trade-end-hour', type=int, default=22, help='End hour for trading (0-23)')
    parser.add_argument('--cooldown', type=int, default=120, help='Cooldown period in minutes')
    parser.add_argument('--exit_percent', type=int, default=1, help='Price increase/decrease to close')
    parser.add_argument('--debug', action='store_true', default=False, help='Debug mode')
    return parser.parse_args()

args = parse_args()
CSV_FILE = args.csv

class SgradtStrategy(Strategy):
    ema_fast = args.ema_fast
    trade_start_hour = args.trade_start_hour
    trade_end_hour = args.trade_end_hour
    cooldown_mins = args.cooldown
    debug = args.debug
    entry_price = 0.0
    exit_percent = args.exit_percent / 100
    
    def init(self):
        close = self.data.Close
        self.ema_fast_line = self.I(talib.EMA, close, self.ema_fast)
        self.entry_pattern = []
        self.last_exit_time = None
        self.entry_direction = None  # Track if we entered long (1) or short (-1)
    
    def next(self):
        current_time = self.data.index[-1]
        
        # Record trend direction based on EMA slope
        ema_rising = self.ema_fast_line[-1] > self.ema_fast_line[-2]
        ema_falling = self.ema_fast_line[-1] < self.ema_fast_line[-2]
        
        if ema_rising:
            self.entry_pattern.append(1)
        elif ema_falling:
            self.entry_pattern.append(-1)
        
        # EXIT LOGIC: Close on FIRST opposing candle after entry
        if self.position:

            if self.position.is_long:
                # Check if price has increased from last entry
                price_inc_percent = self.data.Open[-1] > (self.entry_price + (self.entry_price * self.exit_percent))
                if price_inc_percent:
                    self.log(f"Price increase {self.exit_percent}% | {self.data.Open[-1]} -> {self.entry_price}")
                # Long position: exit immediately when conditions are met
                if price_inc_percent or (self.entry_direction == 1 and ema_falling):
                    self.log(f"Long exit (first reversal): EMA fell at {current_time}")
                    self.position.close()
                    self.entry_direction = None
 
            elif self.position.is_short:
                # Check if price has decreased from last entry
                price_dec_percent = self.data.Open[-1] < (self.entry_price - (self.entry_price * self.exit_percent))
                if price_dec_percent:
                    self.log(f"Price decrease {self.exit_percent}% | {self.data.Open[-1]} -> {self.entry_price}")
                # Short position: exit immediately when conditions are met
                elif price_dec_percent or (self.entry_direction == -1 and ema_rising):
                    self.log(f"Short exit (first reversal): EMA rose at {current_time}")
                    self.position.close()
                    self.entry_direction = None

        # DO NOT TRADE OUTSIDE TRADING HOURS
        if (current_time.hour < self.trade_start_hour or # pyright: ignore 
            current_time.hour >= self.trade_end_hour):   # pyright: ignore
            return
        
        # ENTRY LOGIC: Require 3 consecutive candles in same direction
        if not self.position:
            # Long entry: 3 consecutive rising EMAs
            if len(self.entry_pattern) >= 3 and self.entry_pattern[-3:] == [1, 1, 1]:
                self.log(f"Long entry at {current_time}: {self.entry_pattern[-10:]}")
                self.buy()
                self.entry_price = self.data.Open[-1]
                self.entry_direction = 1  # Mark as long entry
            
            # Short entry: 3 consecutive falling EMAs
            elif len(self.entry_pattern) >= 3 and self.entry_pattern[-3:] == [-1, -1, -1]:
                self.log(f"Short entry at {current_time}: {self.entry_pattern[-10:]}")
                self.sell()
                self.entry_price = self.data.Open[-1]
                self.entry_direction = -1  # Mark as short entry

    def log(self, o):
        if self.debug:
            print(o)


if __name__ == "__main__":
    # Load data
    df = pd.read_csv(CSV_FILE, parse_dates=True, index_col="time")
    df['Open'] = df['open']
    df['High'] = df['high']
    df['Low'] = df['low']
    df['Close'] = df['close']
    df['Volume'] = df['tick_volume']

    bt = FractionalBacktest(df, SgradtStrategy, cash=10000, commission=0.0)

    run_stats = bt.run()
    print(run_stats)

    bt.plot(resample=False)
