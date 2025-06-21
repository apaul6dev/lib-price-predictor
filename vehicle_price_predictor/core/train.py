import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

from vehicle_price_predictor.core.preprocessing import Preprocessor
from vehicle_price_predictor.models.dispatcher import MODEL_DISPATCHER

def train_model(data: pd.DataFrame, trainParams: dict) -> dict:
    """
    Entrena todos los modelos disponibles en MODEL_DISPATCHER y devuelve matrices de confusión por modelo.

    :param data: DataFrame con los datos completos
    :param trainParams: Diccionario con parámetros de entrenamiento, incluyendo:
        - test_size (float): Proporción del conjunto de prueba
        - random_state (int): Semilla para reproducibilidad
        - target_column (str): Nombre de la columna objetivo
    :return: Diccionario con nombre del modelo -> matriz de confusión (DataFrame)
    """
    test_size = trainParams.get("test_size", 0.2)
    random_state = trainParams.get("random_state", 42)
    target_column = trainParams.get("target_column", "price_in_euro")

    prep = Preprocessor()
    df_clean = prep.clean_data(data)
    X, y = prep.split_features_target(df_clean, target_column)

    # División en entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    results = {}
    for model_name, model_class in MODEL_DISPATCHER.items():
        try:
            print(f"\n🚀 Entrenando modelo: {model_name}")
            model = model_class()
            model.train(X_train, y_train)
            preds = model.predict(X_test)

            # Convertimos a clases usando cuartiles
            y_class = pd.qcut(y_test, q=4, labels=False, duplicates='drop')
            preds_class = pd.qcut(preds, q=4, labels=False, duplicates='drop')
            matrix = confusion_matrix(y_class, preds_class)

            results[model_name] = pd.DataFrame(
                matrix,
                index=["Real Q1", "Q2", "Q3", "Q4"][:matrix.shape[0]],
                columns=["Pred Q1", "Q2", "Q3", "Q4"][:matrix.shape[1]]
            )
        except Exception as e:
            print(f"❌ Error al entrenar modelo '{model_name}': {e}")
            results[model_name] = None

    return results
