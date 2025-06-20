import pandas as pd
from core.preprocessing import Preprocessor
from models.catboost.model import CatBoostModel


if __name__ == "__main__":
    try:
        # Cargar y limpiar datos
        df = pd.read_csv("data/vehicle_data.csv")
        df = Preprocessor().clean_data(df)

        # Separar features y target
        X, y = Preprocessor().split_features_target(df, "price_in_euro")

        # Entrenar y predecir
        model = CatBoostModel(iterations=100, learning_rate=0.1, depth=6)
        model.train(X, y)
        predictions = model.predict(X)

        # Guardar resultados y modelo
        df["predicted_price"] = predictions
        df.to_csv("outputs/predicciones_catboost.csv", index=False)
        model.save("models_storage/catboost_model.cbm")

        # Mostrar resultados
        print(df[["price_in_euro", "predicted_price"]].head(10))

    except Exception as e:
        print(f"💥 Error durante la ejecución: {e}")
