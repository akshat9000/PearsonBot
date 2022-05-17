import numpy as np
import pandas as pd
import os
import datetime
from tqdm import tqdm
from sklearn.linear_model import LinearRegression

import warnings
warnings.simplefilter("ignore")

import data_feeder


class PearsonBot:
    def __init__(self, timeframe: int = 30, min_linreg: int = 3):
        self.timeframe = timeframe
        self.min_linreg = min_linreg
        self.df_timeframe = pd.DataFrame()
        self.counter = 0
        self.m = np.nan
        self.c = np.nan
        self.process = True
        self.std = np.nan
        self.in_position = False
        self.signal = 'none'
        self.status = 'none'
        self.df_master = pd.DataFrame()

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
                if not self.in_position and self.signal == 'buy':   # Buy position
                    print(f"{tick.Datetime} Buy @ mkt\topen: {tick.Open}")
                    self.in_position = True
                elif not self.in_position and not self.signal == 'buy': # Do nothing
                    pass
                if self.in_position and self.signal == 'sell':  # Sell position
                    print(f"{tick.Datetime} Sell @ mkt\tclose: {tick.Close}")
                    self.in_position = False
                elif self.in_position and not self.signal == 'sell':    # Do nothing
                    pass

                if self.status == 'middle':
                    if tick.Close >= self.std_channel_up(tick.Close):    # Sell signal
                        self.signal = 'sell'
                    elif tick.Close <= self.std_channel_down(tick.Close):   # Buy signal
                        self.signal = 'buy'

                if tick.Close > self.std_channel_up(tick.Close):
                    self.status = 'up'
                elif tick.Close < self.std_channel_down(tick.Close):
                    self.status = 'down'
                else:
                    self.status = 'middle'

        else:
            self.calc_hl2()
            self.std = self.df_timeframe['hl2'].std()
            self.calc_lr()
            if (self.df_timeframe['lin_reg'].max() - self.df_timeframe['lin_reg'].min()) > self.min_linreg: # Tighter channels
                self.std = self.std / 2
                print("Tighter channels")
            self.df_timeframe['1_std_up'] = self.df_timeframe['index'].apply(lambda x: self.std_channel_up(x, 1))
            self.df_timeframe['2_std_up'] = self.df_timeframe['index'].apply(lambda x: self.std_channel_up(x, 2))
            self.df_timeframe['1_std_down'] = self.df_timeframe['index'].apply(lambda x: self.std_channel_down(x, 1))
            self.df_timeframe['2_std_down'] = self.df_timeframe['index'].apply(lambda x: self.std_channel_down(x, 2))
            self.df_master = pd.concat([self.df_master, self.df_timeframe])
            self.flush_all()
            self.counter += 1
            temp_df = pd.DataFrame([tick], columns=['Datetime', 'Open', 'High', 'Low', 'Close'])
            self.df_timeframe = pd.concat([self.df_timeframe, temp_df])

            self.signal = 'none'

            if tick.Close > self.std_channel_up(tick.Close):
                self.status = 'up'
            elif tick.Close < self.std_channel_down(tick.Close):
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
        for series in tqdm(data_feeder.get_ticks()):
            curr_tick = series
            # print(series.Datetime)
            self.on_tick(series)

        if self.in_position:
            print(f"{curr_tick.Datetime} Sell @ mkt\tclose: {curr_tick.Close} ## EOP")
            self.in_position = False

        self.df_master.to_csv("master.csv", index=False)


if __name__ == "__main__":
    pb = PearsonBot()
    pb.main()
