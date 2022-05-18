import traceback

import numpy as np
import pandas as pd
import os
import datetime
from tqdm import tqdm
from sklearn.linear_model import LinearRegression

import warnings
warnings.simplefilter("ignore")

import data_feeder
settings = data_feeder.get_settings()
DF = data_feeder.DataFeeder(settings)


class PearsonBot:
    def __init__(self, settings):
        try:
            self.timeframe = settings['timeframe']
            self.min_linreg = settings['min_linreg']
            self.df_timeframe = pd.DataFrame()
            self.counter = 0
            self.m = np.nan
            self.c = np.nan
            self.x = settings['x'] if settings['x'] else 1
            self.tp = settings['tp']
            self.sl = settings['sl']
            self.process = True
            self.std = np.nan
            self.in_position = False
            self.signal = 'none'
            self.df_master = pd.DataFrame()
        except Exception as e:
            print(f"Please enter correct settings\n\t{str(e)}\n")
            traceback.print_exc()
            exit()

    def flush_counter(self):
        self.counter = 0

    def flush_df(self):
        self.df_timeframe = pd.DataFrame()

    def flush_all(self):
        self.flush_df()
        self.flush_counter()

    def calc_hl2(self) -> None:
        self.df_timeframe['hl2'] = ((self.df_timeframe['High'] - self.df_timeframe['Low']) / 2) + self.df_timeframe[
            'Low']

    def calc_coeffs(self):
        self.df_timeframe = self.df_timeframe.reset_index().drop(columns='index').reset_index()
        lr = LinearRegression()
        lr.fit(self.df_timeframe[['index']], self.df_timeframe['hl2'])
        self.m = lr.coef_[0]
        self.c = lr.intercept_

    def lin_reg_fn(self, x):
        return (self.m * x) + self.c

    def std_channel_up(self, x, n=1):
        return (self.m * x) + (self.c + (n * self.std))

    def std_channel_down(self, x, n=1):
        return (self.m * x) + (self.c - (n * self.std))

    def calc_lr(self):
        self.calc_coeffs()
        self.df_timeframe['lin_reg'] = self.df_timeframe['index'].apply(self.lin_reg_fn)

    def on_tick(self, tick: pd.Series):
        if self.counter <= self.timeframe:
            self.counter += 1
            temp_df = pd.DataFrame([tick], columns=['Datetime', 'Open', 'High', 'Low', 'Close'])
            self.df_timeframe = pd.concat([self.df_timeframe, temp_df])

            if not np.isnan(self.m) and not np.isnan(self.c):

                if self.in_position and self.signal == 'sell':  # Monitor for tp and sl
                    if tick.Close <= self.prev.Close - self.tp:   # Take profit -> close position
                        print(f"{tick.Datetime} Position closed, TAKE PROFIT hit (short) -> {tick.Close}")
                        self.in_position = False
                        self.signal = 'none'
                    elif (self.prev.Close - self.tp) < tick.Close < (self.prev.Close + self.sl): # Keep monitoring, do nothing
                        pass
                    elif tick.Close >= self.prev.Close + self.sl: # Stop loss -> close position
                        print(f"{tick.Datetime} Position closed, STOP LOSS hit (short) -> {tick.Close}")
                        self.in_position = False
                        self.signal = 'none'

                if self.in_position and self.signal == 'buy':   # Monitor for tp and sl
                    if tick.Close <= self.prev.Close - self.sl:   # Stop loss -> close position
                        print(f"{tick.Datetime} Position closed, STOP LOSS hit (long) -> {tick.Close}")
                        self.in_position = False
                        self.signal = 'none'
                    elif (self.prev.Close - self.sl) < tick.Close < (self.prev.Close + self.tp): # Keep monitoring, do nothing
                        pass
                    elif tick.Close >= self.prev.Close + self.tp: # Take profit -> close position
                        print(f"{tick.Datetime} Position closed, TAKE PROFIT hit (long) -> {tick.Close}")
                        self.in_position = False
                        self.signal = 'none'

                if not self.in_position and self.signal == 'buy':   # Open Long position
                    print(f"{tick.Datetime} Position opened -> LONG @ {tick.Close}")
                    self.in_position = True
                    # self.signal = 'none'

                if not self.in_position and self.signal == 'sell':  # Open Short position
                    print(f"{tick.Datetime} Position opened -> SHORT @ {tick.Close}")
                    self.in_position = True
                    # self.signal = 'none'

                if self.status == 'middle':
                    if tick.Close >= self.std_channel_up(tick.Close, self.x):    # Sell signal
                        self.signal = 'sell'
                    elif tick.Close <= self.std_channel_down(tick.Close, self.x):   # Buy signal
                        self.signal = 'buy'

                if tick.Close > self.std_channel_up(tick.Close, self.x):
                    self.status = 'up'
                elif tick.Close < self.std_channel_down(tick.Close, self.x):
                    self.status = 'down'
                else:
                    self.status = 'middle'

        else:
            self.calc_hl2()
            self.std = self.df_timeframe['hl2'].std()
            self.calc_lr()
            if (self.df_timeframe['lin_reg'].max() - self.df_timeframe['lin_reg'].min()) > self.min_linreg: # Tighter channels
                self.std = self.std / 2
                # print("Tighter channels")
            self.df_timeframe['1_std_up'] = self.df_timeframe['index'].apply(lambda x: self.std_channel_up(x, self.x))
            self.df_timeframe['1_std_down'] = self.df_timeframe['index'].apply(lambda x: self.std_channel_down(x, self.x))
            self.df_master = pd.concat([self.df_master, self.df_timeframe])
            self.flush_all()
            self.counter += 1
            temp_df = pd.DataFrame([tick], columns=['Datetime', 'Open', 'High', 'Low', 'Close'])
            self.df_timeframe = pd.concat([self.df_timeframe, temp_df])

            if self.in_position:
                if self.signal == "buy":
                    print(f"{tick.Datetime} LONG Position closed @ mkt {tick.Close} -> END OF PERIOD")
                    self.in_position = False
                    self.signal = 'none'
                elif self.signal == "sell":
                    print(f"{tick.Datetime} SHORT Position closed @ mkt {tick.Close} -> END OF PERIOD")
                    self.in_position = False
                    self.signal = 'none'

            self.signal = 'none'

            if tick.Close > self.std_channel_up(tick.Close, self.x):
                self.status = 'up'
            elif tick.Close < self.std_channel_down(tick.Close, self.x):
                self.status = 'down'
            else:
                self.status = 'middle'

            # if not self.in_position and self.signal == 'buy':  # Buy position
            #     print(f"{tick.Datetime} Buy @ mkt\topen: {tick.Open}")
            #     self.in_position = True
            # elif not self.in_position and not self.signal == 'buy':  # Do nothing
            #     pass
            # if self.in_position and self.signal == 'sell':  # Sell position
            #     print(f"{tick.Datetime} Sell @ mkt\tclose: {tick.Close}")
            #     self.in_position = False
            # elif self.in_position and not self.signal == 'sell':  # Do nothing
            #     pass
            #
            # if self.status == 'middle':
            #     if tick.Close >= self.std_channel_up(tick.Close):  # Sell signal
            #         self.signal = 'sell'
            #     elif tick.Close <= self.std_channel_down(tick.Close):  # Buy signal
            #         self.signal = 'buy'
            #
            # if tick.Close > self.std_channel_up(tick.Close):
            #     self.status = 'up'
            # elif tick.Close < self.std_channel_down(tick.Close):
            #     self.status = 'down'
            # else:
            #     self.status = 'middle'

    def main(self):
        curr_tick = pd.Series()
        for idx, series in tqdm(enumerate(DF.old_get_tick())):
            if idx == 0:
                self.prev = series
            else:
                self.on_tick(series)
                self.prev = series

        if self.in_position:
            print(f"{curr_tick.Datetime} Backtesting positions closed @ mkt {curr_tick.Close} ## END")
            self.in_position = False

        self.df_master.to_csv("master.csv", index=False)


if __name__ == "__main__":
    pb = PearsonBot(settings)
    pb.main()
