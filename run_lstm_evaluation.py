import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

from src.feature_engineering import create_features
from src.data_processing import split_data, create_sequences
from src.models.lstm import LSTMTrainer

def run_lstm_evaluation(filename, sequence_length: int = 90, test_size: float = 0.2):
    try:
        df_raw = pd.read_csv(f'data/raw/{filename}.csv', index_col = 'Date', parse_dates = True)
    
    except FileNotFoundError:
        print("Error: File not found")
        
        return 
    df_features = create_features(df_raw)
    
    # only use closing price for our lstm model
    data = df_features[['Close']].copy()
    
    # splitting and scale the data
    train_df, test_df = split_data(data, test_size = test_size)
    
    scalar = MinMaxScaler(feature_range = (0,1))
    train_scaled = scalar.fit_transform(train_df)
    test_scaled = scalar.transform(test_df)
   
    # create sequences
    x_train, y_train = create_sequences(train_scaled, sequence_length)
    x_test, y_test = create_sequences(test_scaled, sequence_length) 
   
    # reshape y_train and y_test to be 2-D arrays
    y_train = y_train.reshape(-1, 1)
    y_test = y_test.reshape(-1,1)
    
    # training lstm model
    lstm_trainer = LSTMTrainer(
        input_size = 1, 
        hidden_size = 50, 
        num_layers = 2,
        output_size = 1,
        epochs = 25,
        batch_size = 32, 
        learning_rate = 0.01,
        random_state = 42
    )
    lstm_trainer.fit(x_train, y_train)
    
    # evaluate the model
    predictions_scaled = lstm_trainer.predict(x_test)
    
    # this is where the inverse transform the predictions and values to get the true price and scale 
    predictions = scalar.inverse_transform(predictions_scaled.reshape(-1, 1))
    y_test_actual = scalar.inverse_transform(y_test)
    
    rmse = np.sqrt(mean_squared_error(y_test_actual, predictions))
    print(f"\n--- LSTM Evaluation Complete ---")
    # print(f"Ticker: {ticker}")
    print(f"Test Set RMSE: {rmse:.4f}")

    # 6. Plot the results
    plt.figure(figsize=(14, 7))
    plt.plot(test_df.index[sequence_length:], y_test_actual, color='blue', label='Actual Price')
    plt.plot(test_df.index[sequence_length:], predictions, color='red', label='Predicted Price')
    plt.title("Stock Price Prediction - LSTM")
    plt.xlabel('Date')
    plt.ylabel('Price (USD)')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    # You can change the ticker to evaluate other stocks
    run_lstm_evaluation(input("Please enter filename:\n"))
    
    
    