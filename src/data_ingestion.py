import yfinance as yf
import pandas as pd 
import os 

def download_price_data(ticker, start, end):
    os.makedirs("data/raw", exist_ok=True)
    df=yf.download(ticker, start=start, end=end)
    if df.empty:
        raise ValueError(f"No data found for ticker {ticker} from {start} to {end}.")

    df.index.name="Date"
    output_path=f"data/raw/{ticker}_{start}_{end}.csv"
    df.to_csv(output_path)
    print(f"Data for {ticker} save to {output_path}")
    return df

if __name__ == "__main__":
    download_price_data(ticker ="TSLA",start="2020-06-20", end="2025-06-20")