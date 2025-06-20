import pandas as pd
import os

from core.preprocessing import Preprocessor
from models.catboost.model import CatBoostModel


def load_data(file_path: str) -> pd.DataFrame:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"❌ Archivo no encontrado: {file_path}")
    return pd.read_csv(file_path)


if __name__ == "__main__":
    try:
        # === 1. Cargar datos crudos ===
        df_raw = load_data("data/vehicle_data.csv")
        print(f"📄 Datos crudos cargados: {df_raw.shape[0]} filas")

        # === 2. Preprocesamiento ===
        prep = Preprocessor()
        df_clean = prep.clean_data(df_raw)
        print(f"🧹 Datos limpios: {df_clean.shape[0]} filas")

        # Guardar datos limpios
        os.makedirs("outputs", exist_ok=True)
        df_clean.to_csv("outputs/datos_limpios_categorizados.csv", index=False)

        # === 3. Separar features y target ===
        X, y = prep.split_features_target(df_clean, target_column="price_in_euro")

        # Verificación explícita
        if y.isna().any():
            raise ValueError("❌ La columna 'price_in_euro' contiene valores NaN.")
        if X.isna().any().any():
            raise ValueError("❌ Hay valores NaN en las características (X).")

        # === 4. Transformación (no necesaria para CatBoost, pero mantenida por consistencia) ===
        X_transformed = prep.transform(X)

        # === 5. Entrenamiento del modelo ===
        model = CatBoostModel(iterations=100, learning_rate=0.1, depth=6)
        model.train(X_transformed, y)
        print("✅ Modelo CatBoost entrenado.")

        # === 6. Predicción ===
        predictions = model.predict(X_transformed)
        df_clean["predicted_price"] = predictions
        df_clean.to_csv("outputs/predicciones_catboost.csv", index=False)

        # === 7. Guardar modelo ===
        os.makedirs("models_storage", exist_ok=True)
        model.save("models_storage/catboost_model.cbm")
        print("📦 Modelo guardado en models_storage/catboost_model.cbm")

        # === 8. Mostrar resultados ===
        print("📊 Ejemplo de predicciones:")
        print(df_clean[["price_in_euro", "predicted_price"]].head(10))

    except Exception as e:
        print(f"💥 Error durante la ejecución: {e}")
