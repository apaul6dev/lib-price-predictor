import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

from AIlibrary.core.model_io import save_model
from AIlibrary.core.preprocessing import Preprocessor
from AIlibrary.models.dispatcher import MODEL_DISPATCHER

def train_model(data: pd.DataFrame, trainParams: dict) -> pd.DataFrame:
    """
    Entrena un modelo de IA con los datos proporcionados y devuelve una matriz de confusión.

    :param data: DataFrame con los datos completos
    :param trainParams: Diccionario con parámetros de entrenamiento, incluyendo:
        - model_name (str): Nombre del modelo a entrenar (clave de MODEL_DISPATCHER)
        - test_size (float): Proporción del conjunto de prueba
        - random_state (int): Semilla para reproducibilidad
        - target_column (str): Nombre de la columna objetivo
    :return: Matriz de confusión como DataFrame
    """
    model_name = trainParams.get("model_name")
    if not model_name:
        raise ValueError("Debes proporcionar un 'model_name' en trainParams.")

    model_class = MODEL_DISPATCHER.get(model_name)
    if not model_class:
        raise ValueError(f"Modelo '{model_name}' no soportado. Opciones: {list(MODEL_DISPATCHER.keys())}")

    test_size = trainParams.get("test_size", 0.2)
    random_state = trainParams.get("random_state", 42)
    target_column = trainParams.get("target_column", "price_in_euro")

    # Preparar datos
    prep = Preprocessor()
    df_clean = prep.clean_data(data)
    X, y = prep.split_features_target(df_clean, target_column)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    # Entrenar y predecir
    print(f"\n🚀 Entrenando modelo: {model_name}")
    model = model_class()
    model.train(X_train, y_train)
    preds = model.predict(X_test)
    
    # Guardar modelo
    saved_path = save_model(model_name, model)
    print(f"📦 Modelo guardado en: {saved_path}")

    # Matriz de confusión en cuartiles (para regresión)
    y_class = pd.qcut(y_test, q=4, labels=False, duplicates="drop")
    preds_class = pd.qcut(preds, q=4, labels=False, duplicates="drop")
    matrix = confusion_matrix(y_class, preds_class)

    return pd.DataFrame(
        matrix,
        index=["Real Q1", "Q2", "Q3", "Q4"][:matrix.shape[0]],
        columns=["Pred Q1", "Q2", "Q3", "Q4"][:matrix.shape[1]]
    )
