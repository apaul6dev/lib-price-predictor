from catboost import CatBoostRegressor
from models.base_model import BaseModel

class CatBoostModel(BaseModel):
    def __init__(self, **kwargs):
        self.model = CatBoostRegressor(**kwargs)

    def train(self, X, y):
        self.model.fit(X, y, verbose=False)

    def predict(self, X):
        return self.model.predict(X)

    def save(self, path):
        self.model.save_model(path)

    def load(self, path):
        self.model.load_model(path)
