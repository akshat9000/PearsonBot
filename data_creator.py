import os.path

import numpy as np
import pandas as pd
import data_feeder

import warnings
warnings.simplefilter("ignore")

from tqdm import tqdm
import datetime

# data_path = "data/estest.txt"   # Provide the path to the txt file here

ohlc_dict = {
    "Open": "first",
    "High": "max",
    "Low": "min",
    "Close": "last"
}

def min1(x):
    x - datetime.timedelta(0, 1)
    return x


def main(data_list: list):
    """
    Reads the secondly tick data txt file and converts it into minutely bars
    Outputs a csv named 'resampled' in the data folder
    """
    df = pd.DataFrame()

    for data in tqdm(data_list):
        file = os.path.join(os.getcwd(), "data", data)
        print(f"Reading file: {file}")
        temp = pd.read_csv(file, sep=',', usecols=['Date', 'Time', 'Open', 'High', 'Low', 'Close'])
        df = pd.concat([df, temp])
    df['Datetime'] = df['Date'] + ' ' + df['Time']
    df = df[['Datetime', 'Open', 'High', 'Low', 'Close']]
    df.drop_duplicates(subset=['Datetime'], keep="first", inplace=True)
    print(f"\nConverting to Datetime values...\nThis might take a few minutes...")
    df['Datetime'] = pd.to_datetime(df['Datetime'])
    df.Datetime = df.Datetime - pd.Timedelta(1)
    # df.sort_values(by='Datetime', ascending=True, inplace=True)
    # df['Datetime'] = df['Datetime'].apply(lambda x: min1(x))
    print(f"\nResampling...")
    df_resample_raw = df.set_index("Datetime").resample("1T", closed="left", label="left").agg(ohlc_dict)
    df_resample_raw.reset_index(inplace=True)
    df_resample = df_resample_raw.dropna(subset=['Open'])
    df_resample.to_csv("data/resampled.csv", index=False)
    print(f"\nDone! New file called 'resampled.csv' created in the data folder\nThis file will be used to feed the tickwise data to the backtester\n")


if __name__ == "__main__":
    settings = data_feeder.get_settings()
    print(settings)
    print(settings['tickvalue'])
    # main(settings['data_list'])