

import pandas as pd 
import numpy as np 

def create_features(df):
    df = df.copy() 
    df['Close'] = pd.to_numeric(df['Close'], errors = 'coerce')
    for i in range (1,31):
        df[f'lag_{i}'] = df['Close'].shift(i) 
        """
        Represents closing prices of previous i days, and are crucial for models to learn from past price movement. 
        A 30 day window is chosen to capture approx. one month of trade history
        Using this method, it's reasonable to assume our model would have the highest precision for a 1 month forecast
        """
    
    for window in [5,10,20]: 
        df[f'sma_{window}'] = df['Close'].rolling(window=window).mean() 
       
        df[f'ema_{window}'] = df['Close'].ewm(span = window, adjust = False).mean() 

    
    
    df['volatility_20'] = np.log(df['Close']/df['Close'].shift(1)).rolling(window = 20).std() 


    df['dayofweek'] = df.index.dayofweek # Monday = 0, Sunday = 6
    df['month'] = df.index.month
    df['year'] = df.index.year


    df.dropna(inplace=True)
    return df