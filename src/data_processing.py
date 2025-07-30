import pandas as pd 
import numpy as np

def split_data(df: pd.DataFrame, test_size: float = 0.2):
    split_index = int(len(df)*(1-test_size)) # split_index: cutoff between training data and testing data
    train_df = df.iloc[:split_index] #  creating training df 
    test_df = df.iloc[split_index:] # creating testing df
    return train_df,test_df 

def create_sequences(data: np.ndarray, sequence_length: int): # taking any form of data and sequencializing it into a template our model can understand
    x=[]
    y=[]
    for i in range(len(data)-sequence_length):
        x.append(data[i:(i+sequence_length)])
        y.append(data[i+sequence_length])
    return np.array(x), np.array(y)