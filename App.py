import pandas as pd
from vehicle_price_predictor.core.model_tester import test_model
from vehicle_price_predictor.core.train import train_model
from vehicle_price_predictor.core.model_io import load_model

if __name__ == "__main__":
    df = pd.read_csv("data/vehicle_data.csv")

    train_params = {
        "model_name": "lightgbm",  # Opciones: random_forest, xgboost, lightgbm, catboost
        "test_size": 0.2,
        "random_state": 42,
        "target_column": "price_in_euro"
    }

    # Entrenar y guardar
    matrix = train_model(df, train_params)

    print("\n📊 Matriz de confusión:")
    print(matrix)
    matrix.to_csv(f"outputs/matriz_confusion_{train_params['model_name']}.csv")

    # Cargar modelo desde disco
    print(f"\n🔁 Cargando modelo {train_params['model_name']} desde disco...")
    model = load_model(train_params["model_name"])
    print(f"✅ Modelo {train_params['model_name']} cargado correctamente.")
    
    print(f"\n🔁 Testeando el modelo {train_params['model_name']} previamente entrenado...")
    preds = test_model(df, train_params['model_name'], train_params['target_column'], model)
    print(preds.head())
