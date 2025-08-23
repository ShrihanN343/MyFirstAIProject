# run_classical_evaluation.py

import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import os

# Import project-specific functions and the specific model class
from src.feature_engineering import create_features
from src.models.classical import RandomForestModel # Import the RandomForestModel class

def run_classical_evaluation(filename: str, test_size: float = 0.2):
    """
    Runs a consistent evaluation pipeline for the classical (Random Forest) model.

    Args:
        filename (str): The name of the CSV file in data/raw/ (e.g., 'AAPL').
        test_size (float): The proportion of the dataset to use for testing.
    """
    print(f"\n--- Running Classical (Random Forest) Evaluation for {filename} ---")
    
    # --- 1. Load Data and Create Features ---
    try:
        df_raw = pd.read_csv(f'data/raw/{filename}.csv', index_col='Date', parse_dates=True)
    except FileNotFoundError:
        print(f"Error: File not found at 'data/raw/{filename}.csv'")
        return

    # Classical models like Random Forest require features to work effectively
    df_features = create_features(df_raw)
    
    # --- 2. Define Features (X) and Target (y) ---
    X = df_features.drop('Close', axis=1)
    y = df_features['Close']

    # --- 3. Split Data into Training and Testing sets ---
    # Manual split to create X/y train/test sets
    split_index = int(len(df_features) * (1 - test_size))
    X_train, X_test = X[:split_index], X[split_index:]
    y_train, y_test = y[:split_index], y[split_index:]

    # --- 4. Instantiate and Train the Model ---
    # Create an instance of the RandomForestModel class
    print("Instantiating and training Random Forest model...")
    model = RandomForestModel() # You can pass parameters here, e.g., n_estimators=200
    model.fit(X_train, y_train)
    
    # --- 5. Get Forecasts ---
    # Explicitly call the .predict() method on the model instance
    print("Making predictions on the test set...")
    predictions = model.predict(X_test)
    
    # The actual values for comparison are from our test set
    y_test_actual = y_test.values

    # --- 6. Evaluate the Model ---
    # This will now work correctly as 'predictions' is a numpy array of numbers
    rmse = np.sqrt(mean_squared_error(y_test_actual, predictions))
    print(f"Test Set RMSE: {rmse:.4f}")

    # --- 7. Plot the results ---
    # Ensure the image directory exists
    os.makedirs("images", exist_ok=True)
    
    plt.figure(figsize=(14, 7))
    plt.plot(y_test.index, y_test_actual, color='blue', label='Actual Price')
    plt.plot(y_test.index, predictions, color='green', label='Predicted Price (Random Forest)')
    plt.title(f"{filename} - Classical (Random Forest) Model Prediction")
    plt.xlabel('Date')
    plt.ylabel('Price (USD)')
    plt.legend()
    plt.grid(True)
    
    # Save the plot
    image_path = os.path.join("images", f"{filename}_classical_{rmse:.4f}.png")
    plt.savefig(image_path)
    print(f"Prediction plot saved to: {image_path}")
    # plt.show() # Commented out to allow running for multiple files without interruption

if __name__ == "__main__":
    filename_input = input("Please enter filename (e.g., AAPL):\n")
    if filename_input:
        run_classical_evaluation(filename_input)
