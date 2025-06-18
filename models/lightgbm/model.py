import lightgbm as lgb
from models.base_model import BaseModel

class LightGBMModel(BaseModel):
    def __init__(self, **kwargs):
        self.model = lgb.LGBMRegressor(**kwargs)

    def train(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

    def save(self, path):
        self.model.booster_.save_model(path)

    def load(self, path):
        self.model = lgb.Booster(model_file=path)
