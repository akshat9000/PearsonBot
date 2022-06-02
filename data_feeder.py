import os

import numpy as np
import pandas as pd
from tqdm import tqdm
import json
import time
import datetime

import warnings
warnings.simplefilter("ignore")


def get_settings():
    with open("settings.json", 'r') as f:
        settings = json.load(f)
    return settings


class DataFeeder:
    def __init__(self, settings):
        self.data_list = settings['data_list']

    def print_settings(self):
        print(self.path)
        print(self.data_list)

    def read_data(self, data: str):
        temp = pd.read_csv(self.path+f"/{data}", sep=",", usecols=['Date', 'Time', 'Open', 'High', 'Low', 'Close'])
        temp['Datetime'] = pd.to_datetime(temp['Date'] + " " + temp['Time'])
        return temp[['Datetime', 'Open', 'High', 'Low', 'Close']].reset_index().drop(columns='index')

    def get_tick(self):
        for data in self.data_list:
            df = self.read_data(data)
            for i in range(len(df)):
                yield df.iloc[i]

    def old_get_tick(self):
        try:
            # print(f"\nIn get_old_tick function...\n")
            all_data = pd.DataFrame()
            for data in self.data_list:
                print(os.getcwd())
                temp = pd.read_csv(os.path.join(os.getcwd(), "data", f"{data}"), sep=",", usecols=['Date', 'Time', 'Open', 'High', 'Low', 'Close'])
                temp['Datetime'] = temp['Date'] + " " + temp['Time']
                temp = temp[['Datetime', 'Open', 'Low', 'High', 'Close']]
                all_data = pd.concat([all_data, temp])
            # time.sleep(2)
            # print(f"\n\nConverting to datetime...\n\n")
            # time.sleep(2)
            for i in range(len(all_data)):
                series = all_data.iloc[i]
                dt_str = str(series.Datetime)
                series.Datetime = datetime.datetime.strptime(dt_str, "%m/%d/%Y %H:%M:%S")
                yield series
        except FileNotFoundError:
            print(f"Resampled file not found in data directory, please create data before running")


# def read_data():
#     df = pd.read_csv("resampled.csv", usecols=['Datetime','Open','High','Low','Close'])
#     return df
#
#
# def get_ticks():
#     df = read_data()
#     df = df.reset_index().drop(columns='index')
#     for i in range(len(df)):
#         # yield pd.DataFrame([df.iloc[i]], columns=['Datetime','Open','High','Low','Close'])
#         yield df.iloc[i]


if __name__ == "__main__":
    settings = get_settings()
    # st = settings['start_time']
    # time = datetime.datetime.strptime(st, "%H:%M:%S").time()
    # print(type(time))
    DF = DataFeeder(settings)
    for idx, series in tqdm(enumerate(DF.old_get_tick())):
        print(series)
        break