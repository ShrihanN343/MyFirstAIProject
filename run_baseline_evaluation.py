# run_baseline_evaluation.py

import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import os

# Import project-specific functions and the specific model class
from src.feature_engineering import create_features
from src.models.baseline import NaiveLastValue # Import the NaiveLastValue class

def run_baseline_evaluation(filename: str, test_size: float = 0.2, lag_period: int = 21):
    """
    Runs a consistent evaluation pipeline for the NaiveLastValue baseline model.

    Args:
        filename (str): The name of the CSV file in data/raw/ (e.g., 'AAPL').
        test_size (float): The proportion of the dataset to use for testing.
        lag_period (int): The number of days to lag for the naive forecast. Defaults to 21 (~1 month).
    """
    print(f"\n--- Running Baseline (Lag-{lag_period}) Evaluation for {filename} ---")
    
    # --- 1. Load Data and Create Features ---
    try:
        df_raw = pd.read_csv(f'data/raw/{filename}.csv', index_col='Date', parse_dates=True)
    except FileNotFoundError:
        print(f"Error: File not found at 'data/raw/{filename}.csv'")
        return

    # The NaiveLastValue model uses the specified lag feature
    df_features = create_features(df_raw)
    
    # Drop rows with NaN values that were created by the feature engineering process.
    # This is crucial to ensure the lag features are available for all data points.
    df_features = df_features.dropna()

    # --- 2. Define Features (X) and Target (y) ---
    # The feature set X contains all potential features
    X = df_features.drop('Close', axis=1)
    y = df_features['Close']

    # --- 3. Split Data into Training and Testing sets ---
    # Manual split to create X/y train/test sets
    split_index = int(len(df_features) * (1 - test_size))
    X_train, X_test = X[:split_index], X[split_index:]
    y_train, y_test = y[:split_index], y[split_index:]

    # --- 4. Instantiate and "Train" the Model ---
    # Create an instance of the NaiveLastValue class with the specified lag period.
    print(f"Instantiating baseline model with a {lag_period}-day lag...")
    model = NaiveLastValue(lag_period=lag_period)
    model.fit(X_train, y_train) # This step is just for structural consistency
    
    # --- 5. Get Forecasts ---
    # The predict method will return the specified lag column from X_test
    print("Making predictions on the test set...")
    predictions = model.predict(X_test)
    
    # The actual values for comparison are from our test set
    y_test_actual = y_test.values

    # --- 6. Evaluate the Model ---
    rmse = np.sqrt(mean_squared_error(y_test_actual, predictions))
    print(f"Test Set RMSE: {rmse:.4f}")

    # --- 7. Plot the results ---
    # Ensure the image directory exists
    os.makedirs("images", exist_ok=True)
    
    plt.figure(figsize=(14, 7))
    plt.plot(y_test.index, y_test_actual, color='blue', label='Actual Price')
    plt.plot(y_test.index, predictions, color='red', label=f'Predicted Price (Lag-{lag_period} Baseline)')
    plt.title(f"{filename} - Baseline (Lag-{lag_period}) Model Prediction")
    plt.xlabel('Date')
    plt.ylabel('Price (USD)')
    plt.legend()
    plt.grid(True)
    
    # Save the plot
    image_path = os.path.join("images", f"{filename}_baseline_lag{lag_period}_{rmse:.4f}.png")
    plt.savefig(image_path)
    print(f"Prediction plot saved to: {image_path}")
    # plt.show() # Commented out to allow running for multiple files without interruption

if __name__ == "__main__":
    filename_input = input("Please enter filename (e.g., AAPL):\n")
    if filename_input:
        # We will use a 21-day lag to represent one trading month
        run_baseline_evaluation(filename_input, lag_period=21)
