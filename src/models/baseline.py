# Dummy model which returns yesterday's closing value

class NaiveLastValue:
    def fit(self, x, y):
        pass
    def predict(self, x):
        return x['lag_1']