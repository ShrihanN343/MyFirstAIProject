# Step 1: Read and print the data from yfinance
# Need to take user input through HTML screen (currently hard-coded)

import yfinance as yf
import pandas as pd 
import os # "Operating system" file management, file creation, executing scripts, etc

def download_price_data(ticker, start, end):
    os.makedirs("data/raw", exist_ok=True) # makedirs: makes a new folder/directory specified by what is in the string
    df=yf.download(ticker, start=start, end=end, auto_adjust = True) # df is NOT a keyword, but it is standard naming conventions for a Pandas data frame
    if df.empty: # checking for invalid ticker (nothing was returned from the yfinance package)
        raise ValueError(f"No data found for ticker {ticker} from {start} to {end}.")

    # df.index.name="Date" 
    df.reset_index(inplace=True) # yfinance returns date as an index - resetting the index makes date a regular column which is easier to work with in Pandas
    output_path=f"data/raw/{ticker}_{start}_{end}.csv" # output_path is specifying where we're creating and storing data for the stock
    df.to_csv(output_path, index=False) # removes index from df
    print(f"Data for {ticker} save to {output_path}") 
    return df

if __name__ == "__main__":
    download_price_data(ticker ="XOM",start="2010-06-20", end="2025-06-20") # downloading data
