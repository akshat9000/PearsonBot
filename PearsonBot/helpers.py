import numpy as np
import pandas as pd
import datetime
from tqdm import tqdm

import warnings
warnings.simplefilter("ignore")

ohlc_dict = {
    "Open": "first",
    "High": "max",
    "Low": "min",
    "Close": "last",
    "cont_up": "mean",
    "cont_down": "mean",
}


def convert_second_to_minute(df: pd.DataFrame) -> pd.DataFrame:
    # print(f"\n\n\tConverting seconds into minutes\n\n")
    df.reset_index(inplace=True)
    df.to_csv(f"check_times.csv", index=False)
    # df['Datetime'] = df['Datetime'] - pd.Timedelta(1)
    df.set_index("Datetime", inplace=True)
    df_sampled = df.resample("1min").apply(ohlc_dict)
    df_sampled = df_sampled.reset_index()
    # df_sampled = df_sampled.dropna(subset=['Open'])
    return df_sampled