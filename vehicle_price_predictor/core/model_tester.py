import pandas as pd
import os

from vehicle_price_predictor.core.preprocessing import Preprocessor

def test_model(data: pd.DataFrame, model_name: str, target_column: str, model) -> pd.Series:
    """
    Recibe datos ya preprocesados y un modelo entrenado para retornar y guardar predicciones.

    :param data: DataFrame limpio con características (features) listas para predecir
    :param model_name: Nombre del modelo (solo para nombrar la columna de salida)
    :param target_column: Nombre de la columna objetivo (si está presente)
    :param model: Instancia del modelo previamente entrenado
    :return: Serie con predicciones
    """
    # Preprocesar internamente para obtener features
    prep = Preprocessor()
    df_clean = prep.clean_data(data)
    X, y = prep.split_features_target(df_clean, target_column)

    # Predecir
    predictions = model.predict(X)
    pred_series = pd.Series(predictions, name=f"predicted_{model_name}")

    # Crear DataFrame de salida
    output_df = df_clean.copy()
    output_df[pred_series.name] = pred_series.values

    # Crear carpeta y guardar archivo
    os.makedirs("outputs", exist_ok=True)
    output_path = f"outputs/predicciones_{model_name}.csv"
    output_df.to_csv(output_path, index=False)
    print(f"📄 Predicciones guardadas en: {output_path}")

    return pred_series
