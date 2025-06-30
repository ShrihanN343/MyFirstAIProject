from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor

class RidgeModel:
    def __init__(self, alpha=1.0):
        self.model = Ridge(alpha=alpha)

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

class RandomForestModel:
    def __init__(self, n_estimators=100, max_depth=10, random_state=42):
        self.model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
            n_jobs=-1  # Use all available cores
        )

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)
