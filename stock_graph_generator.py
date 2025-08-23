# STOCK_PRICE_GRAPH.py

import pandas as pd
import matplotlib.pyplot as plt
import os
import glob # Import glob to find files

def generate_stock_price_graph(ticker_input: str):
    """
    Loads a stock data CSV file and generates a graph of its weekly 
    closing price over time to show long-term trends.

    Args:
        ticker_input (str): The stock ticker symbol or filename prefix (e.g., 'AAPL').
    """
    # Standardize the ticker name for clean labels and searching
    ticker = ticker_input.split('_')[0]

    print(f"\n--- Generating Weekly History Price Graph for {ticker} ---")
    
    # --- 1. Find and Load Data ---
    # Use glob to find the file that starts with the ticker symbol
    search_pattern = os.path.join('data', 'raw', f'{ticker}*.csv')
    file_list = glob.glob(search_pattern)

    if not file_list:
        print(f"Error: No CSV file found for ticker '{ticker}' in the 'data/raw/' directory.")
        return
    
    # Use the first file found that matches the pattern
    file_path = file_list[0]
    print(f"Found data file: {file_path}")
    
    try:
        # Read the CSV, using the 'Date' column as the index and parsing it as dates
        df = pd.read_csv(file_path, index_col='Date', parse_dates=True)
        
        # --- FIX: Ensure the 'Close' column is numeric ---
        # This forces the column to be numbers. If any value can't be converted, 
        # it becomes NaN (Not a Number).
        df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
        
        # Drop any rows where the 'Close' price could not be converted to a number
        df.dropna(subset=['Close'], inplace=True)

        # Sort the DataFrame by the date index to ensure correct plotting
        df.sort_index(inplace=True)

        # Resample the daily data to weekly data to clarify the long-term trend
        df_weekly = df['Close'].resample('W').last()

    except Exception as e:
        print(f"An error occurred while reading or processing the file: {e}")
        return

    # --- 2. Generate the Plot ---
    print("Generating plot...")
    plt.figure(figsize=(14, 7))
    
    # Plot the resampled weekly data
    plt.plot(df_weekly.index, df_weekly, color='dodgerblue', label=f'{ticker} Weekly Close Price')
    
    # Add titles and labels for clarity
    plt.title(f"Historical Weekly Closing Price for {ticker}")
    plt.xlabel('Date')
    plt.ylabel('Price (USD)')
    plt.legend()
    plt.grid(True)
    
    # --- 3. Save the Plot ---
    # Ensure the image directory exists
    os.makedirs("images", exist_ok=True)
    
    # Update the filename to reflect the weekly view
    image_path = os.path.join("images", f"{ticker}_weekly_history.png")
    plt.savefig(image_path)
    print(f"Graph saved to: {image_path}")
    
    # The plot will no longer pop up automatically
    # plt.show()

if __name__ == "__main__":
    # Prompt the user to enter the stock's ticker symbol
    ticker_input = input("Please enter the stock's ticker symbol (e.g., AAPL):\n")
    
    if ticker_input:
        generate_stock_price_graph(ticker_input)
    else:
        print("No ticker entered. Exiting.")
