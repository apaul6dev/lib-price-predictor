import pandas as pd
import os

from core.preprocessing import Preprocessor
from models.catboost.model import CatBoostModel
from models.lightgbm.model import LightGBMModel


def run_model(model, model_name: str, X, y, df: pd.DataFrame):
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
    model_file = f"models_storage/{model_name.lower()}_model.txt"
    model.save(model_file)
    print(f"📦 Modelo guardado en {model_file}")

    # Mostrar ejemplo
    print(df_out[[f"predicted_price_{model_name.lower()}"]].head(5))


if __name__ == "__main__":
    try:
        # Preparar carpetas
        os.makedirs("outputs", exist_ok=True)
        os.makedirs("models_storage", exist_ok=True)

        # Cargar y limpiar datos
        df = pd.read_csv("data/vehicle_data.csv")
        prep = Preprocessor()
        df_clean = prep.clean_data(df)
        X, y = prep.split_features_target(df_clean, "price_in_euro")

        # Ejecutar ambos modelos
        models = [
            ("CatBoost", CatBoostModel(iterations=100, learning_rate=0.1, depth=6)),
            ("LightGBM", LightGBMModel(n_estimators=100, learning_rate=0.1, max_depth=6)),
        ]

        for name, model_instance in models:
            run_model(model_instance, name, X, y, df_clean)

    except Exception as e:
        print(f"💥 Error durante la ejecución: {e}")
