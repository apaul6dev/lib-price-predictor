import xgboost as xgb
import pandas as pd
from AIlibrary.models.base_model import BaseModel


class XGBoostModel(BaseModel):
    def __init__(self, **kwargs):
        self.model = xgb.XGBRegressor(**kwargs)
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
        self.model.save_model(path)

    def load(self, path):
        self.model.load_model(path)
        self.is_trained = True
