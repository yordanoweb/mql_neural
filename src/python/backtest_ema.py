from backtesting import Backtest, Strategy
from backtesting.lib import FractionalBacktest, crossover
import talib
import pandas as pd
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Dual EMA Crossover Strategy - Long Only')
    parser.add_argument('--csv', type=str, default="ndx100_rates_h1_2020.csv", help='CSV file path')
    parser.add_argument('--ema-fast', type=int, default=20, help='Fast EMA period')
    parser.add_argument('--trade-start-hour', type=int, default=9, help='Start hour for trading (0-23)')
    parser.add_argument('--trade-end-hour', type=int, default=11, help='End hour for trading (0-23)')
    parser.add_argument('--cooldown', type=int, default=120, help='Cooldown period in minutes')
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
    
    def init(self):
        close = self.data.Close
        self.ema_fast_line = self.I(talib.EMA, close, self.ema_fast)
        self.entry_pattern = []
        self.long_entry_pattern  = [-1, -1, -1, -1, -1,  1,  1]
        self.short_entry_pattern = [ 1,  1,  1,  1,  1, -1, -1]
        self.last_exit_time = None
    
    def next(self):
        # DO NOT TRADE OUTSIDE TRADING HOURS
        start_hour = self.trade_start_hour
        end_hour = self.trade_end_hour
        if self.data.index[-1].hour < start_hour or self.data.index[-1].hour >= end_hour:
            return

        if self.ema_fast_line[-2] < self.ema_fast_line[-3]:
            if self.debug:
                print(f"EMA falling: True at {self.data.index[-1]}")
            self.entry_pattern.append(-1)
        if self.ema_fast_line[-2] > self.ema_fast_line[-3]:
            if self.debug:
                print(f"EMA rising: True at {self.data.index[-1]}")
            self.entry_pattern.append(1)

        current_pattern = []
        l = len(self.long_entry_pattern)
        if len(self.entry_pattern) > l:
            current_pattern = self.entry_pattern[-l:]
        else:
            return

        # Never have more than 45 bars in the entry pattern
        if len(self.entry_pattern) > 45:
            self.entry_pattern = self.entry_pattern[-45:]

        if not self.position:
            # COOLDOWN CHECK
            if self.last_exit_time is not None:
                minutes_passed = (self.data.index[-1] - self.last_exit_time).total_seconds() / 60
                if minutes_passed < self.cooldown_mins:
                    return

            # LONG ENTRY
            if current_pattern == self.long_entry_pattern:
                if args.debug:
                    print(f"LONG ENTRY at {self.data.index[-1]}")
                    print(f"Bigger context: {self.entry_pattern[-30:]}")
                self.buy()
            # SHORT ENTRY
            if current_pattern == self.short_entry_pattern:
                if args.debug:
                    print(f"SHORT ENTRY at {self.data.index[-1]}")
                    print(f"Bigger context: {self.entry_pattern[-30:]}")
                self.sell()
        else:
            # LONG EXIT
            if self.position.is_long:
                if current_pattern == self.short_entry_pattern:
                    if args.debug:
                        print(f"LONG EXIT at {self.data.index[-1]}")
                        print(f"Bigger context: {self.entry_pattern[-30:]}")
                    self.position.close()
                    self.last_exit_time = self.data.index[-1]
            # SHORT EXIT
            if self.position.is_short:
                if current_pattern == self.long_entry_pattern:
                    if args.debug:
                        print(f"SHORT EXIT at {self.data.index[-1]}")
                        print(f"Bigger context: {self.entry_pattern[-30:]}")
                    self.position.close()
                    self.last_exit_time = self.data.index[-1]

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