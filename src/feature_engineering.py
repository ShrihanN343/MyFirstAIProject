# meant for pre-processing data, giving model technical data such as trends, moving data, etc (used by industry professionals)

import pandas as pd 
import numpy as np 

def create_features(df):
    # create time series features from a data frame of price data
    df = df.copy
    # keeping raw data, df means data frame
    for i in range (1,31):
        df[f'lag_{i}'] = df['Close'].shift(i)
    # allows model to understand changes-of-price trends across months 
    for window in [5,10,20]:
        df[f'sma_{window}'] = df['Close'].rolling(window=window).mean()
        # simple moving avg = sma, ema = exp moving avg, sma analyzes broader trends, ema is more sensitive in shorter time spans
        df[f'ema_{window}'] = df['Close'].ewm(span = window, adjust = False).mean()
    
    df['volatility_20'] = np.log(df['Close']/df['Close'].shift(1)).rolling(window = 20).std()
    # creates rolling metric of the last 20 days to reflect the overall volatility of stock price regardless of + or -

    df['dayofweek'] = df.index.dayofweek
    df['month'] = df.index.month
    df['year'] = df.index.year

    df.dropna(inplace=True)
    # original data frame is diretly modified, just modifying df object, no new data 
    return df