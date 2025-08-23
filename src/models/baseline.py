# Dummy model which returns yesterday's closing value

import numpy as np

class NaiveLastValue:
    """
    A simple baseline model that forecasts the value from a specified prior time step.
    """
    def __init__(self, lag_period=1):
        """
        Initializes the model with a specific lag period.

        Args:
            lag_period (int): The number of time steps to look back for the forecast. 
                              Defaults to 1 (previous day's value).
        """
        self.lag_period = lag_period
        if self.lag_period < 1:
            raise ValueError("Lag period must be at least 1.")

    def fit(self, X, y):
        """This model is stateless, so fit does nothing."""
        pass

    def predict(self, X):
        """
        For each instance in X, predict the value from the specified lag feature.
        """
        # Construct the column name for the desired lag
        lag_column = f'lag_{self.lag_period}'
        
        # Check if the required lag feature exists in the DataFrame
        if lag_column not in X.columns:
            raise ValueError(
                f"'{lag_column}' not found in features. "
                f"Ensure feature_engineering.py creates lags up to at least {self.lag_period} days."
            )
            
        return X[lag_column]


