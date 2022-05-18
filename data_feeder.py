import numpy as np
import pandas as pd
from tqdm import tqdm
import json

import warnings
warnings.simplefilter("ignore")


def get_settings():
    with open("settings.json", 'r') as f:
        settings = json.load(f)
    return settings


class DataFeeder:
    def __init__(self, settings):
        self.path = settings['data_path']
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
        temp = pd.read_csv("data/resampled.csv", usecols=['Datetime', 'Open', 'High', 'Low', 'Close'])
        temp['Datetime'] = pd.to_datetime(temp['Datetime'])
        for i in range(len(temp)):
            yield temp.iloc[i]


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
    DF = DataFeeder(settings)
    for idx, series in tqdm(enumerate(DF.old_get_tick())):
        if idx == 0:
            print(idx)
            print("prev")
        else:
            print(idx)
            break