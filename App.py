import pandas as pd
from vehicle_price_predictor.core.train import train_model

if __name__ == "__main__":
    df = pd.read_csv("data/vehicle_data.csv")

    train_params = {
        "test_size": 0.2,
        "random_state": 42,
        "target_column": "price_in_euro"
    }

    matrices = train_model(df, train_params)

    for model, matrix in matrices.items():
        print(f"\n📊 Matriz de confusión para {model}:")
        print(matrix)

        if matrix is not None:
            matrix.to_csv(f"outputs/matriz_confusion_{model}.csv")

