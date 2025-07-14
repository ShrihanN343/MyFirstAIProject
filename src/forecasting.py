import pandas as pd
import numpy as np
import json 
import os 
from src.feature_engineering import create_features
from src.models.classical import RidgeModel
import datetime

def generate_forecast(ticker, days_to_forecast=30, RMSE=0.0165):
    # Step 1: load and prepare data 

    end_date = datetime.date.today()
    start_date = end_date - datetime.timedelta(days=10*365)
    start_str = start_date.strftime('%Y-%m-%d')
    end_str = end_date.strftime('%Y-%m-%d')
    file_path=f"data/raw/{ticker}_{start_str}_{end_str}.csv"
    try:
        df_raw=pd.read_csv(file_path, index_col='Date', parse_dates=True, header=0)
    except FileNotFoundError:
        print(f"Data not found for ticker {ticker} at {file_path}")
        return None
    df_features=create_features(df_raw)
    x=df_features.drop(columns=['Close'])
    y=df_features['Close']

    # Step 2: train the RidgeModel 
        # .fit trains the model

    top_model=RidgeModel()
    top_model.fit(x,y)

    # Step 3: generate iterative forecasts

    # Start with the most recent data point to generate features for the next step

    last_row=df_features.iloc[[-1]]

    forecasted_values = []
    current_features = last_row.drop(columns=['Close'])
    for _ in range(days_to_forecast):
        # predict the next step
        next_prediction = top_model.predict(current_features)[0]
        forecasted_values.append(next_prediction)

        # update features for next prediction

        new_row = current_features.iloc[0].to_dict

        # integrating shift lags - closing prices of previous i days and is crucial for models to learn for past price movements

        for i in range(29,0,-1):
            new_row[f'lag_{i+1}'] = new_row[f'lag_{i}']

        new_row['lag_1'] = next_prediction  # the new prediction becomes the most recent lag
        current_feature = pd.DataFrame([new_row], index=[current_features.index[0]+pd.Timedelta(days=1)])

    # step 4: save artifacts 

    forecast_dates = pd.date_range(start=df_features.index[-1]+pd.Timedelta(days=1), periods=days_to_forecast)

    # calculate prediction interval

    margin_of_error=1.96*RMSE
    forecast_df=pd.DataFrame({
        'Forecast': forecasted_values, 
        'Lower_bound_95': np.array(forecasted_values)-margin_of_error,
        'Upper_bound_95': np.array(forecasted_values)+margin_of_error
    }, index=forecast_dates)

    os.makedirs('results/forecasts', exist_ok=True)
    forecasts_output_path = 'results/forecasts/final_forecast.csv'
    forecast_df.to_csv(forecasts_output_path)
    
    print(f"Final forecasts saved to {forecasts_output_path}")

    # Save the final model metrics 

    os.makedirs('results/metrics', exist_ok=True)
    # exist_ok=True: doesnt make a new file if the one specified is there

    metrics_output_path = 'results/metrics/final_metrics.json'
    with open(metrics_output_path, 'w') as file: # calling the open method to write and create file at metrics_output_path
        json.dump({"model": "LSTMTrainer"}, file)
    print(f"Final metric saved to {metrics_output_path}")
    
if __name__ == "__main__":
    generate_forecast("TSLA", "")
    
