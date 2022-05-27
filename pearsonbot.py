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
            self.start_time = datetime.datetime.strptime(settings['start_time'], "%H:%M:%S").time()
            self.end_time = datetime.datetime.strptime(settings['end_time'], "%H:%M:%S").time()
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
            self.signal_next = 'none'
            self.df_master = pd.DataFrame()
            self.entry = np.nan
            self.entry_df = pd.DataFrame()
            self.exit = np.nan
            self.exit_df = pd.DataFrame()
            self.all_trades = pd.DataFrame(columns=['Datetime', 'Open', 'High', 'Low', 'Close', 'side', 'reg_line_val', 'status'])
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

    def sell_signal(self, tick):
        index = 30 + self.counter
        value = self.std_channel_up(index, self.x)
        if self.prev.High < value <= tick.High:
            return True
        else:
            return False

    def buy_signal(self, tick):
        index = 30 + self.counter
        if self.prev.Low > self.std_channel_down(index, self.x) >= tick.Low:
            return True
        else:
            return False

    def do_tick(self, tick):
        # if not self.in_position and self.sell_signal(tick) and self.signal_next == "none":  # Open SHORT position on next candle
        #     self.signal_next = "sell"
        # elif not self.in_position and not self.sell_signal(tick) and self.signal_next == "none":  # Do nothing
        #     pass
        #
        # if not self.in_position and self.buy_signal(tick) and self.signal_next == "none":  # Open LONG position on next candle
        #     self.signal_next = "buy"
        # elif not self.in_position and not self.buy_signal(tick) and self.signal_next == "none":  # Do nothing
        #     pass

        if not self.in_position and self.sell_signal(tick) and self.signal_next == "none":  # Open SHORT position on tick.Open
            self.signal_next = "sell"
            index = 30 + self.counter
            value = self.std_channel_up(index, self.x)
            value = round(value * 4) / 4
            print(f"{tick.Datetime} SHORT position opened @ mkt open -> {value}")
            temp = pd.DataFrame([tick], columns=['Datetime', 'Open', 'High', 'Low', 'Close'])
            temp['side'] = "short"
            temp['entry_price'] = value
            temp['entry_price'] = [value]
            self.entry_df = pd.concat([self.entry_df, temp])
            self.all_trades = pd.concat([self.all_trades, temp])
            self.entry = value
            self.in_position = True
        elif self.signal_next == "sell" and self.in_position:  # Monitor for sl, tp
            take_profit = self.entry - self.tp
            take_profit = round(take_profit*4)/4
            if tick.Low <= take_profit:  # Take profit -> close position
                temp = pd.DataFrame([tick], columns=['Datetime', 'Open', 'High', 'Low', 'Close'])
                index = 30 + self.counter
                value = self.std_channel_up(index, self.x)
                temp['side'] = "short"
                temp['exit_price'] = take_profit
                temp['exit_price'] = [take_profit]
                temp['original_status'] = "win"
                self.exit_df = pd.concat([self.exit_df, temp])
                self.all_trades = pd.concat([self.all_trades, temp])
                self.exit = take_profit
                print(f"{tick.Datetime} SHORT position closed @ mkt close -> {take_profit}\tTAKE PROFIT\n\tentry price: {self.entry}\texit price: {self.exit}")
                self.in_position = False
                self.signal_next = "none"
            elif self.entry - self.tp < tick.Close < self.entry + self.sl:  # Do nothing
                pass
            elif tick.Close >= self.entry + self.sl:  # Stop loss -> close position
                stop_loss = self.entry + self.sl
                stop_loss = round(stop_loss*4)/4
                temp = pd.DataFrame([tick], columns=['Datetime', 'Open', 'High', 'Low', 'Close'])
                index = 30 + self.counter
                value = self.std_channel_up(index, self.x)
                temp['side'] = "short"
                temp['exit_price'] = stop_loss
                temp['exit_price'] = [stop_loss]
                temp['original_status'] = "lose"
                self.exit_df = pd.concat([self.exit_df, temp])
                self.all_trades = pd.concat([self.all_trades, temp])
                self.exit = stop_loss
                print(f"{tick.Datetime} SHORT position closed @ mkt close -> {stop_loss}\tSTOP LOSS\n\tentry price: {self.entry}\texit price: {self.exit}")
                self.in_position = False
                self.signal_next = "none"
        elif not self.in_position and self.buy_signal(tick) and self.signal_next == "none":  # Open LONG position on tick.Open
            index = 30 + self.counter
            value = self.std_channel_up(index, self.x)
            value = round(value*4)/4
            self.signal_next = "buy"
            print(f"{tick.Datetime} LONG position opened @ mkt open -> {value}")
            temp = pd.DataFrame([tick], columns=['Datetime', 'Open', 'High', 'Low', 'Close'])
            temp['side'] = "long"
            temp['entry_price'] = value
            temp['entry_price'] = [value]
            self.entry_df = pd.concat([self.entry_df, temp])
            self.all_trades = pd.concat([self.all_trades, temp])
            self.entry = value
            self.in_position = True
        elif self.signal_next == "buy" and self.in_position:  # Monitor for sl, top
            take_profit = self.entry + self.tp
            take_profit = round(take_profit*4)/4
            if tick.High >= take_profit:  # Take profit -> close position
                temp = pd.DataFrame([tick], columns=['Datetime', 'Open', 'High', 'Low', 'Close'])
                index = 30 + self.counter
                value = self.std_channel_up(index, self.x)
                temp['side'] = "long"
                temp['exit_price'] = take_profit
                temp['exit_price'] = [take_profit]
                temp['original_status'] = "win"
                self.exit_df = pd.concat([self.exit_df, temp])
                self.all_trades = pd.concat([self.all_trades, temp])
                self.exit = take_profit
                print(f"{tick.Datetime} LONG position closed @ mkt close -> {take_profit}\tTAKE PROFIT\n\tentry price: {self.entry}\texit price: {self.exit}")
                self.in_position = False
                self.signal_next = "none"
            elif self.entry + self.tp < tick.Close < self.entry - self.sl:  # Do nothing
                pass
            elif tick.Low <= self.entry - self.sl:  # Stop loss -> close position
                stop_loss = self.entry - self.sl
                stop_loss = round(stop_loss*4)/4
                temp = pd.DataFrame([tick], columns=['Datetime', 'Open', 'High', 'Low', 'Close'])
                index = 30 + self.counter
                value = self.std_channel_up(index, self.x)
                temp['side'] = "long"
                temp['exit_price'] = stop_loss
                temp['exit_price'] = [stop_loss]
                temp['original_status'] = "lose"
                self.exit_df = pd.concat([self.exit_df, temp])
                self.all_trades = pd.concat([self.all_trades, temp])
                self.exit = stop_loss
                print(f"{tick.Datetime} LONG position closed @ mkt close -> {stop_loss}\tSTOP LOSS\n\tentry price: {self.entry}\texit price: {self.exit}")
                self.in_position = False
                self.signal_next = "none"

    def on_tick(self, tick: pd.Series):
        if self.counter <= self.timeframe:
            self.counter += 1
            index = self.counter + 30
            temp_df = pd.DataFrame([tick], columns=['Datetime', 'Open', 'High', 'Low', 'Close'])

            try:
                std_up = self.std_channel_up(index, self.x)
                std_down = self.std_channel_down(index, self.x)
                temp_df['cont_up'] = std_up
                temp_df['cont_up'] = [std_up]
                temp_df['cont_down'] = std_down
                temp_df['cont_down'] = [std_down]
            except Exception as ex:
                print(str(ex))
                temp_df['cont_up'] = 0
                temp_df['cont_up'] = [0]
                temp_df['cont_down'] = 0
                temp_df['cont_down'] = [0]

            self.df_timeframe = pd.concat([self.df_timeframe, temp_df])

            if not np.isnan(self.m) and not np.isnan(self.c):
                self.do_tick(tick)

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
            index = self.counter + 30

            try:
                std_up = self.std_channel_up(index, self.x)
                std_down = self.std_channel_down(index, self.x)
                temp_df['cont_up'] = std_up
                temp_df['cont_up'] = [std_up]
                temp_df['cont_down'] = std_down
                temp_df['cont_down'] = [std_down]
            except Exception as ex:
                print(str(ex))
                temp_df['cont_up'] = 0
                temp_df['cont_up'] = [0]
                temp_df['cont_down'] = 0
                temp_df['cont_down'] = [0]

            self.df_timeframe = pd.concat([self.df_timeframe, temp_df])

            if not np.isnan(self.m) and not np.isnan(self.c):
                self.do_tick(tick)

    def main(self):
        curr_tick = pd.Series()
        for idx, series in enumerate(DF.old_get_tick()):
            curr_time = series.Datetime.time()
            if idx == 0:
                self.prev = series
            else:
                if self.start_time <= curr_time <= self.end_time:
                    self.on_tick(series)
                    self.prev = series
                else:
                    if self.in_position:
                        print(f"{self.prev.Datetime} Backtesting positions closed @ mkt {self.prev.Close} ## END")
                        self.in_position = False
                        self.signal_next = "none"
                        temp = pd.DataFrame([self.prev], columns=['Datetime', 'Open', 'High', 'Low', 'Close'])
                        index = 30 + self.counter
                        value = self.std_channel_up(index, self.x)
                        value = round(value*4)/4
                        temp['side'] = "prev"
                        temp['exit_price'] = value
                        temp['exit_price'] = [value]
                        temp['original_status'] = "END local"
                        self.exit_df = pd.concat([self.exit_df, temp])
                        self.all_trades = pd.concat([self.all_trades, temp])


        if self.in_position:
            print(f"{self.prev.Datetime} Backtesting positions closed @ mkt {self.prev.Close} ## END")
            self.in_position = False
            temp = pd.DataFrame([self.prev], columns=['Datetime', 'Open', 'High', 'Low', 'Close'])
            index = 30 + self.counter
            value = self.std_channel_up(index, self.x)
            value = round(value*4)/4
            temp['side'] = "prev"
            temp['exit_price'] = value
            temp['exit_price'] = [value]
            temp['original_status'] = "END"
            self.exit_df = pd.concat([self.exit_df, temp])
            self.all_trades = pd.concat([self.all_trades, temp])

        self.all_trades.to_csv("all_trades.csv", index=False)
        self.df_master.to_csv("master.csv", index=False)
        self.entry_df.to_csv("entry_df.csv", index=False)
        self.exit_df.to_csv("exit_df.csv", index=False)


if __name__ == "__main__":
    pb = PearsonBot(settings)
    pb.main()
