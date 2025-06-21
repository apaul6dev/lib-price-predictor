import os
from joblib import dump, load

MODEL_EXTENSION = {
    "catboost": ".cbm",
    "lightgbm": ".txt",
    "random_forest": ".joblib",
    "xgboost": ".json"
}

def save_model(model_name: str, model) -> str:
    """
    Guarda el modelo en disco y devuelve su path.
    """
    ext = MODEL_EXTENSION.get(model_name, ".model")
    os.makedirs("models_storage", exist_ok=True)
    model_path = os.path.join("models_storage", f"{model_name}_model{ext}")

    if hasattr(model, "save"):
        model.save(model_path)
    else:
        dump(model, model_path)

    return model_path


def load_model(model_name: str):
    """
    Carga un modelo previamente guardado desde disco.
    """
    ext = MODEL_EXTENSION.get(model_name, ".model")
    model_path = os.path.join("models_storage", f"{model_name}_model{ext}")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"No se encontró el modelo en: {model_path}")

    from vehicle_price_predictor.models.dispatcher import MODEL_DISPATCHER
    model_class = MODEL_DISPATCHER.get(model_name)

    if not model_class:
        raise ValueError(f"Modelo '{model_name}' no está registrado.")

    model_instance = model_class()
    if hasattr(model_instance, "load"):
        model_instance.load(model_path)
        return model_instance
    else:
        return load(model_path)
