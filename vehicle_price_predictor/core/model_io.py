import os

# Opcionalmente puedes usar joblib o los métodos nativos de cada modelo
from joblib import dump, load

MODEL_EXTENSION = {
    "catboost": ".cbm",
    "lightgbm": ".txt",
    "random_forest": ".joblib",
    "xgboost": ".json"
}

def save_model(model_name: str, model) -> str:
    """
    Guarda el modelo en disco según su nombre y devuelve el path.

    :param model_name: Nombre del modelo (clave en dispatcher)
    :param model: Instancia del modelo ya entrenado
    :return: Ruta del archivo guardado
    """
    ext = MODEL_EXTENSION.get(model_name, ".model")
    os.makedirs("models_storage", exist_ok=True)
    model_path = os.path.join("models_storage", f"{model_name}_model{ext}")

    # Método save propio o fallback a joblib
    if hasattr(model, "save"):
        model.save(model_path)
    else:
        dump(model, model_path)

    return model_path


def load_model(model_name: str):
    """
    Carga un modelo previamente guardado según su nombre.

    :param model_name: Nombre del modelo
    :return: Modelo cargado
    """
    ext = MODEL_EXTENSION.get(model_name, ".model")
    model_path = os.path.join("models_storage", f"{model_name}_model{ext}")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"No se encontró el modelo en: {model_path}")

    # Método load propio o fallback a joblib
    if model_name in ["catboost", "xgboost", "lightgbm"]:
        from vehicle_price_predictor.models.dispatcher import MODEL_DISPATCHER
        model = MODEL_DISPATCHER[model_name]()
        model.load(model_path)
        return model
    else:
        return load(model_path)
