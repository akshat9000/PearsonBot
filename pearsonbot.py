import traceback

import numpy as np
import pandas as pd
import os
import datetime
import helpers
from sklearn.linear_model import LinearRegression

import warnings

warnings.simplefilter("ignore")

import data_feeder

settings = data_feeder.get_settings()
DF = data_feeder.DataFeeder(settings)

ohlc_dict = {
    "Open": "first",
    "High": "max",
    "Low": "min",
    "Close": "last"
}


class PearsonBot:
    def __init__(self, settings):
        try:
            self.entry_timer = np.nan
            self.amtperpoint = settings['amtperpoint']
            self.tick_value = settings['tickvalue']
            self.start_time = datetime.datetime.strptime(settings['start_time'], "%H:%M:%S").time()
            self.end_time = datetime.datetime.strptime(settings['end_time'], "%H:%M:%S").time()
            self.timeframe = settings['timeframe']  # * 60 # in minutes
            self.min_linreg = settings['min_linreg']
            self.df_timeframe_sec = pd.DataFrame()
            self.df_timeframe_min = pd.DataFrame()
            self.status = "none"
            self.counter = 0
            self.m = np.nan
            self.c = np.nan
            self.x = settings['x'] if settings['x'] else 1
            self.tp = settings['tp'] #/ settings['amtperpoint']
            self.tp_original = settings['tp']
            self.sl = settings['sl'] #/ settings['amtperpoint']
            self.sl_original = settings['sl']
            self.process = True
            self.std = np.nan
            self.in_position = False
            self.signal_next = 'none'
            self.df_master = pd.DataFrame()
            self.entry = np.nan
            self.entry_df = pd.DataFrame()
            self.exit = np.nan
            self.exit_df = pd.DataFrame()
            self.trade_taken = False
            self.all_trades = pd.DataFrame(
                columns=['Datetime', 'Open', 'High', 'Low', 'Close', 'side'])
        except Exception as e:
            print(f"Please enter correct settings\n\t{str(e)}\n")
            traceback.print_exc()
            exit()

    def flush_trade_taken(self):
        self.trade_taken = False

    def flush_counter(self):
        self.counter = 0

    def flush_df(self):
        self.df_timeframe_sec = pd.DataFrame()
        self.df_timeframe_min = pd.DataFrame()

    def flush_all(self):
        self.flush_df()
        self.flush_counter()
        self.flush_trade_taken()

    def save_minutely(self):
        df_master = self.df_master.set_index("Datetime").resample("1T", closed="left", label="left").agg(ohlc_dict)
        df_master.reset_index(inplace=True)
        df_master.dropna(subset=['Open'], inplace=True)
        return df_master

    def calc_hl2(self) -> None:
        self.df_timeframe_min['hl2'] = ((self.df_timeframe_min['High'] + self.df_timeframe_min['Low']) / 2)

    def calc_coeffs(self):
        self.df_timeframe_min = self.df_timeframe_min.reset_index().drop(columns='index').reset_index()
        lr = LinearRegression()
        lr.fit(self.df_timeframe_min[['index']], self.df_timeframe_min['hl2'])
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
        self.df_timeframe_min['lin_reg'] = self.df_timeframe_min['index'].apply(self.lin_reg_fn)

    def sell_signal(self, tick):
        index = int(30 + (self.counter / 60))
        value = self.std_channel_up(index, self.x)
        # if self.prev.High < value <= tick.High:
        if value <= tick.High:
            return True
        else:
            return False

    def buy_signal(self, tick):
        index = int(30 + (self.counter / 60))
        # if self.prev.Low > self.std_channel_down(index, self.x) >= tick.Low:
        value = self.std_channel_down(index, self.x)
        if value >= tick.Low:
            return True
        else:
            return False

    def set_status(self, tick, up_std, down_std, hl2):
        if tick.Low > up_std:
            self.status = "up"
        elif tick.High < down_std:
            self.status = "down"
        # else:
        #     self.status = "middle"
        elif tick.High <= up_std and tick.Low >= down_std:
            self.status = "middle"
        elif tick.High > up_std > tick.Low > down_std:
            if hl2 > up_std:
                self.status = "hup"
            else:
                self.status = "middle"
        elif up_std > tick.High > down_std > tick.Low:
            if hl2 > down_std:
                self.status = "middle"
            else:
                self.status = "hdown"

    def monitor_short(self, tick, take_profit, stop_loss):
        if tick.Low <= take_profit:  # Take profit -> close position
            temp = pd.DataFrame([tick], columns=['Datetime', 'Open', 'High', 'Low', 'Close'])
            temp['side'] = "short"
            temp['exit_price'] = take_profit
            temp['exit_price'] = [take_profit]
            temp['original_status'] = "win"
            self.exit_df = pd.concat([self.exit_df, temp])
            self.all_trades = pd.concat([self.all_trades, temp])
            self.exit = take_profit
            print(
                f"{tick.Datetime} SHORT position closed @ mkt close -> {take_profit}\tTAKE PROFIT\n\tentry price: {self.entry}\texit price: {self.exit}")
            self.in_position = False
            self.signal_next = "none"
        elif tick.High >= stop_loss:  # Stop loss -> close position
            temp = pd.DataFrame([tick], columns=['Datetime', 'Open', 'High', 'Low', 'Close'])
            temp['side'] = "short"
            temp['exit_price'] = stop_loss
            temp['exit_price'] = [stop_loss]
            temp['original_status'] = "lose"
            self.exit_df = pd.concat([self.exit_df, temp])
            self.all_trades = pd.concat([self.all_trades, temp])
            self.exit = stop_loss
            print(
                f"{tick.Datetime} SHORT position closed @ mkt close -> {stop_loss}\tSTOP LOSS\n\tentry price: {self.entry}\texit price: {self.exit}")
            self.in_position = False
            self.signal_next = "none"
        elif tick.Low > take_profit and tick.High < stop_loss:  # Do nothing
            pass

    def monitor_long(self, tick, take_profit, stop_loss):
        if tick.High >= take_profit:  # Take profit -> close position
            temp = pd.DataFrame([tick], columns=['Datetime', 'Open', 'High', 'Low', 'Close'])
            temp['side'] = "long"
            temp['exit_price'] = take_profit
            temp['exit_price'] = [take_profit]
            temp['original_status'] = "win"
            self.exit_df = pd.concat([self.exit_df, temp])
            self.all_trades = pd.concat([self.all_trades, temp])
            self.exit = take_profit
            print(
                f"{tick.Datetime} LONG position closed @ mkt close -> {take_profit}\tTAKE PROFIT\n\tentry price: {self.entry}\texit price: {self.exit}")
            self.in_position = False
            self.signal_next = "none"
        elif tick.Low <= stop_loss:  # Stop loss -> close position
            temp = pd.DataFrame([tick], columns=['Datetime', 'Open', 'High', 'Low', 'Close'])
            temp['side'] = "long"
            temp['exit_price'] = stop_loss
            temp['exit_price'] = [stop_loss]
            temp['original_status'] = "lose"
            self.exit_df = pd.concat([self.exit_df, temp])
            self.all_trades = pd.concat([self.all_trades, temp])
            self.exit = stop_loss
            print(
                f"{tick.Datetime} LONG position closed @ mkt close -> {stop_loss}\tSTOP LOSS\n\tentry price: {self.entry}\texit price: {self.exit}")
            self.in_position = False
            self.signal_next = "none"
        elif tick.High < take_profit and tick.Low > stop_loss:  # Do nothing
            pass

    def do_tick(self, tick):
        index = int(30 + (self.counter / 60))
        up_std = self.std_channel_up(index, self.x)
        down_std = self.std_channel_down(index, self.x)
        hl2 = ((tick.High - tick.Low) / 2) + tick.Low

        self.set_status(tick, up_std, down_std, hl2)

        if not self.in_position and self.sell_signal(tick) and self.signal_next == "none" and (self.status == "middle" or self.status == "hup"):
            # Open SHORT position on tick.Open
            self.trade_taken = True
            self.signal_next = "sell"
            value = up_std
            value = round(value * self.tick_value) / self.tick_value
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
            stop_loss = self.entry + self.sl
            self.monitor_short(tick, take_profit, stop_loss)
            # if tick.Low <= take_profit:  # Take profit -> close position
            #     temp = pd.DataFrame([tick], columns=['Datetime', 'Open', 'High', 'Low', 'Close'])
            #     temp['side'] = "short"
            #     temp['exit_price'] = take_profit
            #     temp['exit_price'] = [take_profit]
            #     temp['original_status'] = "win"
            #     self.exit_df = pd.concat([self.exit_df, temp])
            #     self.all_trades = pd.concat([self.all_trades, temp])
            #     self.exit = take_profit
            #     print(f"{tick.Datetime} SHORT position closed @ mkt close -> {take_profit}\tTAKE PROFIT\n\tentry price: {self.entry}\texit price: {self.exit}")
            #     self.in_position = False
            #     self.signal_next = "none"
            # elif tick.Low > take_profit and tick.High < stop_loss:  # Do nothing
            #     pass
            # elif tick.High >= stop_loss:  # Stop loss -> close position
            #     temp = pd.DataFrame([tick], columns=['Datetime', 'Open', 'High', 'Low', 'Close'])
            #     temp['side'] = "short"
            #     temp['exit_price'] = stop_loss
            #     temp['exit_price'] = [stop_loss]
            #     temp['original_status'] = "lose"
            #     self.exit_df = pd.concat([self.exit_df, temp])
            #     self.all_trades = pd.concat([self.all_trades, temp])
            #     self.exit = stop_loss
            #     print(f"{tick.Datetime} SHORT position closed @ mkt close -> {stop_loss}\tSTOP LOSS\n\tentry price: {self.entry}\texit price: {self.exit}")
            #     self.in_position = False
            #     self.signal_next = "none"
        elif not self.in_position and self.buy_signal(tick) and self.signal_next == "none" and (self.status == "middle" or self.status == "hdown"):
            # Open LONG position on tick.Open
            self.trade_taken = True
            value = down_std
            value = round(value * self.tick_value) / self.tick_value
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
            stop_loss = self.entry - self.sl
            self.monitor_long(tick, take_profit, stop_loss)
            # if tick.High >= take_profit:  # Take profit -> close position
            #     temp = pd.DataFrame([tick], columns=['Datetime', 'Open', 'High', 'Low', 'Close'])
            #     temp['side'] = "long"
            #     temp['exit_price'] = take_profit
            #     temp['exit_price'] = [take_profit]
            #     temp['original_status'] = "win"
            #     self.exit_df = pd.concat([self.exit_df, temp])
            #     self.all_trades = pd.concat([self.all_trades, temp])
            #     self.exit = take_profit
            #     print(f"{tick.Datetime} LONG position closed @ mkt close -> {take_profit}\tTAKE PROFIT\n\tentry price: {self.entry}\texit price: {self.exit}")
            #     self.in_position = False
            #     self.signal_next = "none"
            # elif tick.High < take_profit and tick.Low > stop_loss:  # Do nothing
            #     pass
            # elif tick.Low <= stop_loss:  # Stop loss -> close position
            #     temp = pd.DataFrame([tick], columns=['Datetime', 'Open', 'High', 'Low', 'Close'])
            #     temp['side'] = "long"
            #     temp['exit_price'] = stop_loss
            #     temp['exit_price'] = [stop_loss]
            #     temp['original_status'] = "lose"
            #     self.exit_df = pd.concat([self.exit_df, temp])
            #     self.all_trades = pd.concat([self.all_trades, temp])
            #     self.exit = stop_loss
            #     print(f"{tick.Datetime} LONG position closed @ mkt close -> {stop_loss}\tSTOP LOSS\n\tentry price: {self.entry}\texit price: {self.exit}")
            #     self.in_position = False
            #     self.signal_next = "none"

    def on_tick(self, tick: pd.Series):
        # if self.counter < self.timeframe:
        try:
            if np.isnan(self.entry_timer):
                self.entry_timer = tick
        except Exception:
            pass
        check_val = self.entry_timer.Datetime + datetime.timedelta(0, 60 * self.timeframe)
        if tick.Datetime < check_val:
            self.counter += 1
            index = int(30 + (self.counter / 60))
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

            self.df_timeframe_sec = pd.concat([self.df_timeframe_sec, temp_df])

            if not np.isnan(self.m) and not np.isnan(self.c):
                if not self.trade_taken:
                    self.do_tick(tick)
                else:
                    if self.in_position:
                        # first monitor short
                        if self.signal_next == "sell":
                            take_profit = self.entry - self.tp
                            stop_loss = self.entry + self.sl
                            self.monitor_short(tick, take_profit, stop_loss)

                        # then monitor long
                        elif self.signal_next == "buy":
                            take_profit = self.entry + self.tp
                            stop_loss = self.entry - self.sl
                            self.monitor_long(tick, take_profit, stop_loss)

        else:
            self.df_timeframe_min = helpers.convert_second_to_minute(self.df_timeframe_sec)
            self.entry_timer = tick
            self.calc_hl2()
            max = self.df_timeframe_min['High'].max()
            min = self.df_timeframe_min['Low'].min()
            self.std = (max - min) #/ 2
            # self.std = self.df_timeframe_sec['hl2'].std()
            self.calc_lr()
            # print(self.df_timeframe_sec['lin_reg'].max() - self.df_timeframe_sec['lin_reg'].min())
            if (self.df_timeframe_min['lin_reg'].max() - self.df_timeframe_min['lin_reg'].min()) > self.min_linreg:  # Tighter channels
                self.std = self.std / 2
            self.df_timeframe_min['1_std_up'] = self.df_timeframe_min['index'].apply(lambda x: self.std_channel_up(x, self.x))
            self.df_timeframe_min['1_std_down'] = self.df_timeframe_min['index'].apply(
                lambda x: self.std_channel_down(x, self.x))
            self.df_master = pd.concat([self.df_master, self.df_timeframe_min])
            self.flush_all()
            self.counter += 1
            temp_df = pd.DataFrame([tick], columns=['Datetime', 'Open', 'High', 'Low', 'Close'])
            index = int(30 + (self.counter / 60)) #-1

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

            self.df_timeframe_sec = pd.concat([self.df_timeframe_sec, temp_df])

            if not np.isnan(self.m) and not np.isnan(self.c):
                self.do_tick(tick)

    def main(self):
        curr_tick = pd.Series()
        for idx, series in enumerate(DF.old_get_tick()):
            # print("*"*10, f"running for: {series.Datetime}", "*"*10)
            curr_time = series.Datetime.time()
            if idx == 0:
                self.prev = series
                # self.entry_timer = series
            if self.start_time <= curr_time <= self.end_time:
                if curr_time >= datetime.time(6, 31, 59):
                    # print("IN")
                    pass
                self.on_tick(series)
                self.prev = series
            else:
                if self.in_position:
                    print(f"{self.prev.Datetime} Backtesting positions closed @ mkt {self.prev.Close} ## END")
                    self.in_position = False
                    self.signal_next = "none"
                    temp = pd.DataFrame([self.prev], columns=['Datetime', 'Open', 'High', 'Low', 'Close'])
                    index = int(30 + (self.counter / 60))
                    value = self.std_channel_up(index, self.x)
                    value = round(value * self.tick_value) / self.tick_value
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
            index = int(30 + (self.counter / 60))
            value = self.std_channel_up(index, self.x)
            value = round(value * self.tick_value) / self.tick_value
            temp['side'] = "prev"
            temp['exit_price'] = value
            temp['exit_price'] = [value]
            temp['original_status'] = "END"
            self.exit_df = pd.concat([self.exit_df, temp])
            self.all_trades = pd.concat([self.all_trades, temp])

        self.all_trades.to_csv(f"outputs/all_trades/all_trades_std_{self.x}_tp_{self.tp_original}_sl_{self.sl_original}.csv", index=False)

        # self.df_master.to_csv(f"outputs/master/master_std_{self.x}_tp_{self.tp_original}_sl_{self.sl_original}.csv", index=False)
        # df_master = self.save_minutely()
        # df_master.to_csv(f"outputs/master/master_std_{self.x}_tp_{self.tp_original}_sl_{self.sl_original}.csv", index=False)
        self.df_master.to_csv(f"outputs/master/master_std_{self.x}_tp_{self.tp_original}_sl_{self.sl_original}.csv", index=False)

        self.entry_df.to_csv(f"outputs/entry/entry_df_std_{self.x}_tp_{self.tp_original}_sl_{self.sl_original}.csv", index=False)
        self.exit_df.to_csv(f"outputs/exit/exit_df_std_{self.x}_tp_{self.tp_original}_sl_{self.sl_original}.csv", index=False)

        entries = pd.read_csv(os.path.join(os.getcwd(), "outputs", "entry", f"entry_df_std_{self.x}_tp_{self.tp_original}_sl_{self.sl_original}.csv"), usecols=['Datetime', 'entry_price', 'side'])
        entries.rename(columns={'Datetime': 'datetime_entry'}, inplace=True)
        exits = pd.read_csv(os.path.join(os.getcwd(), "outputs", "exit", f"exit_df_std_{self.x}_tp_{self.tp_original}_sl_{self.sl_original}.csv"), usecols=['Datetime', 'exit_price'])
        exits.rename(columns={'Datetime': 'datetime_exit'}, inplace=True)

        temp_df = pd.concat([entries, exits], axis=1)[['datetime_entry', 'datetime_exit', 'entry_price', 'exit_price', 'side']]
        temp_df['pnl'] = temp_df['exit_price'] - temp_df['entry_price']
        temp_df['pnl'] = np.where(temp_df['side'] == "long", 1, -1) * temp_df['pnl']
        temp_df['pnl%'] = ((temp_df['exit_price'] / temp_df['entry_price']) - 1)
        temp_df['pnl%'] = np.where(temp_df['side'] == "long", 1, -1) * temp_df['pnl%']
        temp_df['rtns'] = temp_df['pnl'] * self.amtperpoint
        temp_df['cuml pnl%'] = np.cumsum(temp_df['pnl%'])
        temp_df['cuml rtns'] = np.cumsum(temp_df['rtns'])
        temp_df['status'] = np.where(temp_df['pnl%'] > 0, "win", "lose")
        temp_df.to_csv(f"outputs/pnl/pnl_df_std_{self.x}_tp_{self.tp_original}_sl_{self.sl_original}.csv", index=False)


if __name__ == "__main__":
    pb = PearsonBot(settings)
    pb.main()
