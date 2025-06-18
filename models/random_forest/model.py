from sklearn.ensemble import RandomForestRegressor
from models.base_model import BaseModel
import joblib

class RandomForestModel(BaseModel):
    def __init__(self, **kwargs):
        self.model = RandomForestRegressor(**kwargs)

    def train(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

    def save(self, path):
        joblib.dump(self.model, path)

    def load(self, path):
        self.model = joblib.load(path)
