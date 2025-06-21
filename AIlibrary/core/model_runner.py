import pandas as pd
import os
from AIlibrary.core.evaluator import evaluate
from AIlibrary.models.dispatcher import MODEL_DISPATCHER

def test_model(model_name: str, X_train, y_train, X_test, y_test):
    """
    Crea, entrena y evalúa un modelo según el nombre especificado usando el dispatcher.

    :param model_name: Nombre del modelo (debe estar en MODEL_DISPATCHER)
    :param X_train: Datos de entrenamiento (features)
    :param y_train: Etiquetas de entrenamiento
    :param X_test: Datos de prueba (features)
    :param y_test: Etiquetas de prueba
    :return: Tuple (modelo entrenado, predicciones)
    :raises ValueError: Si el modelo no está registrado
    """
    model_class = MODEL_DISPATCHER.get(model_name)
    if not model_class:
        raise ValueError(f"Modelo '{model_name}' no soportado.")

    model = model_class()
    model.train(X_train, y_train)
    preds = model.predict(X_test)
    return model, preds


def run_model(model_name: str, X, y, df: pd.DataFrame) -> dict:
    """
    Ejecuta un modelo de machine learning: entrena, predice, guarda resultados y evalúa métricas.

    :param model_name: Nombre del modelo a ejecutar
    :param X: Conjunto de características
    :param y: Variable objetivo
    :param df: DataFrame original (limpio) para agregar predicciones
    :return: Diccionario con métricas de evaluación (MAE, MSE, R2)
    """
    print(f"\n *** Ejecutando modelo: {model_name} ***")

    # Entrenamiento y predicción sobre el mismo conjunto (usualmente validación cruzada)
    model, predictions = test_model(model_name, X, y, X, y)

    # === Guardar predicciones ===
    df_out = df.copy()
    df_out[f"predicted_price_{model_name}"] = predictions
    os.makedirs("outputs", exist_ok=True)
    output_path = f"outputs/predicciones_{model_name}.csv"
    df_out.to_csv(output_path, index=False)
    print(f"📄 Resultados guardados en {output_path}")

    # === Guardar modelo ===
    ext_map = {
        "catboost": ".cbm",
        "lightgbm": ".txt",
        "random_forest": ".joblib",
        "xgboost": ".json"
    }
    ext = ext_map.get(model_name, ".model")
    os.makedirs("models_storage", exist_ok=True)
    model_path = f"models_storage/{model_name}_model{ext}"
    model.save(model_path)
    print(f"📦 Modelo guardado en {model_path}")

    # === Evaluar métricas de desempeño ===
    metrics = evaluate(y, predictions)
    metrics["model"] = model_name
    return metrics
