import pandas as pd

from vehicle_price_predictor.core.preprocessing import Preprocessor
from vehicle_price_predictor.core.model_runner import run_model

if __name__ == "__main__":
    try:
        # Preparar datos
        df = pd.read_csv("data/vehicle_data.csv")
        prep = Preprocessor()
        df_clean = prep.clean_data(df)
        X, y = prep.split_features_target(df_clean, "price_in_euro")

        # Modelos a evaluar
        model_names = ["catboost", "lightgbm", "random_forest", "xgboost"]
        metrics = [run_model(name, X, y, df_clean) for name in model_names]

        # Mostrar comparativa
        df_metrics = pd.DataFrame(metrics).set_index("model")
        print("\n📊 Comparativa de modelos:")
        print(df_metrics.round(3))

        # Guardar métricas
        df_metrics.to_csv("outputs/metricas_modelos.csv")
        print("\n📁 Métricas guardadas en outputs/metricas_modelos.csv")

    except Exception as e:
        print(f"💥 Error durante la ejecución: {e}")
