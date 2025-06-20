from vehicle_price_predictor.models.random_forest.model import RandomForestModel
from vehicle_price_predictor.models.xgboost.model import XGBoostModel
from vehicle_price_predictor.models.lightgbm.model import LightGBMModel
from vehicle_price_predictor.models.catboost.model import CatBoostModel

MODEL_DISPATCHER = {
    "random_forest": RandomForestModel,
    "xgboost": XGBoostModel,
    "lightgbm": LightGBMModel,
    "catboost": CatBoostModel,
}
