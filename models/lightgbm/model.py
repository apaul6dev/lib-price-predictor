import lightgbm as lgb
import pandas as pd
import numpy as np
from models.base_model import BaseModel


class LightGBMModel(BaseModel):
    def __init__(self, **kwargs):
        self.model = lgb.LGBMRegressor(**kwargs)
        self.is_trained = False

    def train(self, X, y):
        if pd.isna(y).any() or pd.isna(X).any().any():
            raise ValueError("❌ Los datos de entrenamiento contienen valores NaN.")
        self.model.fit(X, y)
        self.is_trained = True

    def predict(self, X):
        if not self.is_trained:
            raise RuntimeError("❌ El modelo no ha sido entrenado. Llama a `train()` primero o carga un modelo existente.")
        if pd.isna(X).any().any():
            raise ValueError("❌ Los datos de predicción contienen valores NaN.")
        return self.model.predict(X)

    def save(self, path):
        if not self.is_trained:
            raise RuntimeError("❌ No puedes guardar un modelo que no ha sido entrenado.")
        self.model.booster_.save_model(path)

    def load(self, path):
        self.model = lgb.Booster(model_file=path)
        self.is_trained = True
