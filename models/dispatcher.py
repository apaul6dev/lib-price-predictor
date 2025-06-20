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
