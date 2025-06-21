import pandas as pd
from vehicle_price_predictor.core.train import train_model

if __name__ == "__main__":
    df = pd.read_csv("data/vehicle_data.csv")

    train_params = {
        "model_name": "xgboost", #random_forest, xgboost, lightgbm, catboost
        "test_size": 0.2,
        "random_state": 42,
        "target_column": "price_in_euro"
    }
        
    matrix = train_model(df, train_params)
    print("\n📊 Matriz de confusión:")
    print(matrix)

    matrix.to_csv(f"outputs/matriz_confusion_{train_params['model_name']}.csv")