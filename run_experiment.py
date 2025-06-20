import pandas as pd
import os

from core.evaluator import evaluate
from core.preprocessing import Preprocessor
from models.catboost.model import CatBoostModel
from models.lightgbm.model import LightGBMModel
from models.random_forest.model import RandomForestModel
from models.xgboost.model import XGBoostModel


def run_model(model, model_name: str, X, y, df: pd.DataFrame) -> dict:
    print(f"\n🚀 Entrenando modelo: {model_name}")
    model.train(X, y)
    predictions = model.predict(X)

    # Guardar predicciones
    output_file = f"outputs/predicciones_{model_name.lower()}.csv"
    df_out = df.copy()
    df_out[f"predicted_price_{model_name.lower()}"] = predictions
    df_out.to_csv(output_file, index=False)
    print(f"📄 Resultados guardados en {output_file}")

    # Guardar modelo
    ext_map = {
        "CatBoost": ".cbm",
        "LightGBM": ".txt",
        "RandomForest": ".joblib",
        "XGBoost": ".json"
    }
    ext = ext_map.get(model_name, ".model")
    model_file = f"models_storage/{model_name.lower()}_model{ext}"
    model.save(model_file)
    print(f"📦 Modelo guardado en {model_file}")

    # Calcular métricas con función externa
    metrics = evaluate(y, predictions)
    metrics["model"] = model_name
    return metrics


if __name__ == "__main__":
    try:
        # Crear carpetas de salida
        os.makedirs("outputs", exist_ok=True)
        os.makedirs("models_storage", exist_ok=True)

        # Cargar y preparar datos
        df = pd.read_csv("data/vehicle_data.csv")
        prep = Preprocessor()
        df_clean = prep.clean_data(df)
        X, y = prep.split_features_target(df_clean, "price_in_euro")

        # Definir modelos a evaluar
        models = [
            ("CatBoost", CatBoostModel(iterations=100, learning_rate=0.1, depth=6)),
            ("LightGBM", LightGBMModel(n_estimators=100, learning_rate=0.1, max_depth=6)),
            ("RandomForest", RandomForestModel(n_estimators=100, max_depth=8)),
            ("XGBoost", XGBoostModel(n_estimators=100, learning_rate=0.1, max_depth=6))
        ]

        metrics = []

        for name, model_instance in models:
            result = run_model(model_instance, name, X, y, df_clean)
            metrics.append(result)

        # Mostrar tabla comparativa
        print("\n📊 Comparativa de modelos:")
        df_metrics = pd.DataFrame(metrics).set_index("model")
        print(df_metrics.round(3))

        # Guardar métricas en CSV
        df_metrics.to_csv("outputs/metricas_modelos.csv")
        print("\n📁 Métricas guardadas en outputs/metricas_modelos.csv")

    except Exception as e:
        print(f"💥 Error durante la ejecución: {e}")
