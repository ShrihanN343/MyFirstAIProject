# purpose: testing model, training the model

import pandas as pd 
import numpy as np
from sklearn.metrics import mean_squared_error
from src.data_ingestion import download_price_data
from src.feature_engineering import create_features
from src.models.lstm import LSTMTrainer
from src.models.baseline import NaiveLastValue
from src.models.classical import RidgeModel, RandomForestModel
from sklearn.model_selection import TimeSeriesSplit

def evaluate_models(filename):
    # 1. Load and prepare data
    print("Loading and preparing data...")
    try:
        df_raw=pd.read_csv(f'data/raw/{filename}.csv', index_col='Date',parse_dates=True)
        for col in ['Close','High','Low','Open','Volume']:
            if col in df_raw.columns:
                df_raw[col]=pd.to_numeric(df_raw[col], errors='coerce')
        df_raw.dropna(inplace=True)
    except FileNotFoundError:
        print("File not found. Please run data ingestion")
        return 
    df_raw.columns=['Close', 'High', 'Low', 'Open', 'Volume'] # define columns in our data frame
    df_features=create_features(df_raw) # calling function create_features

    x=df_features.drop(columns=['Close']) # Get rid of the close column from our x-axis 
    y=df_features['Close'] # y is labels for data at closing time, used for training and evaluating model
    print("Data preparation complete")
    # 2. define model 
    print("Defining models")
    models={
        "LSTM": LSTMTrainer(input_size=1, epochs=20), # input_size is the size of model, epoch is the number of cycles that our model is trained for 
        "Naive Last Value": NaiveLastValue(),
        "Random Forest": RandomForestModel(n_estimators=100,max_depth=10, random_state=42),
        "Ridge": RidgeModel(alpha=1.0)
    }

    # 3. time series cross validation
    # for times series data, standard kfold cross validation is not appropriate bc it shuffles data, 
    # breaking the temporal order and causing data leakage (training on future data to predict the past)
    # time series split ensures that the validation set always comes after the training set
    tscv = TimeSeriesSplit(n_splits=5)
    results = {}

    print("Starting model evaluation...")
    for model_name, model in models.items():
        fold_RMSES = [] # list to store RMSE for each fold
        # each fold is a window for splitting the data between training and test set
        # fold represents a single round of training and testing
        # a fold is one complete train-test cycle on a chronologically-ordered slice of time series data

        # The tscv.split(x) generates a pair of train/test indices for each fold
        for fold, (train_index, test_index) in enumerate(tscv.split(x)):
            # split the data into training and testing sets for the current fold
            x_train,x_test = x.iloc[train_index], x.iloc[test_index]
            y_train,y_test = y.iloc[train_index], y.iloc[test_index]

            # fit the model on the training data
            model.fit(x_train,y_train)

            # make predictions on test data 
            predictions=model.predict(x_test)


            """
            Evaluate the predictions using Root Mean Squared Error (RMSE)
            RMSE isa good metric for regression tasks as it penalizes large error more heavily
            and is in the same unit as the target variable (price).
            Lower RMSE = better
            """

            RMSE=np.sqrt(mean_squared_error(y_test, predictions))
            fold_RMSES.append(RMSE)
            print(f" Fold {fold+1}/{tscv.n_splits} RMSE: {RMSE:.4f}")

        # calculate the avg RMSE across all folds for the current model
        avg_RMSE=np.mean(fold_RMSES)
        results[model_name]=avg_RMSE
        print(f" avg RMSE for {model_name}: {avg_RMSE:.4f}")

    # 4. print final results

    print("evaluation complete: model rankings")
    for model_name, avg_RMSE in sorted(results.items(),key=lambda item: item[1]):
        print(f"{model_name}: avg RMSE = {avg_RMSE:.4f}")

if __name__ == "__main__":
    evaluate_models("AAPL_2015-06-20_2025-06-20")