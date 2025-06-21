from AIlibrary.models.random_forest.model import RandomForestModel
from AIlibrary.models.xgboost.model import XGBoostModel
from AIlibrary.models.lightgbm.model import LightGBMModel
from AIlibrary.models.catboost.model import CatBoostModel

MODEL_DISPATCHER = {
    "random_forest": RandomForestModel,
    "xgboost": XGBoostModel,
    "lightgbm": LightGBMModel,
    "catboost": CatBoostModel,
}
