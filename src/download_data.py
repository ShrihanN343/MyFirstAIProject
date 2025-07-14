import datetime 
from data_ingestion import download_price_data
def main():
    tickers = ['AAPL', 'NVDA', 'TSLA', 'MSFT', 'AMZN', 'TSLA', 'JPM', 'XOM', 'UNH', 'TSM']
    end_date = datetime.date.today()
    start_date = end_date - datetime.timedelta(days=10*365)
    start_str = start_date.strftime('%Y-%m-%d')
    end_str = end_date.strftime('%Y-%m-%d')
    for ticker in tickers:
        try:
            download_price_data(ticker=ticker, start=start_str, end=end_str)
        except Excpetion as e:
            print(f"Failed to download data for {ticker}: {e}")

if __name__ == "__main__":
    main()