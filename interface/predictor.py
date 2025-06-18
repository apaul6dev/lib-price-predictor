from models.random_forest.model import RandomForestModel
from models.xgboost.model import XGBoostModel
from models.lightgbm.model import LightGBMModel
from models.catboost.model import CatBoostModel

MODEL_DISPATCHER = {
    "random_forest": RandomForestModel,
    "xgboost": XGBoostModel,
    "lightgbm": LightGBMModel,
    "catboost": CatBoostModel,
}

def test_model(model_name: str, X_train, y_train, X_test, y_test):
    model_class = MODEL_DISPATCHER.get(model_name)
    if not model_class:
        raise ValueError(f"Modelo '{model_name}' no soportado.")

    model = model_class()
    model.train(X_train, y_train)
    preds = model.predict(X_test)
    return preds
