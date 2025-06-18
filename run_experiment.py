from core.preprocessing import Preprocessor
import pandas as pd

def load_data(file_path: str) -> pd.DataFrame:
    return pd.read_csv(file_path)

if __name__ == "__main__":
    # Cargar datos crudos
    df = load_data("data/vehicle_data.csv")

    # Crear instancia del preprocesador
    prep = Preprocessor()

    # Limpiar datos
    df_clean = prep.clean_data(df)

    # Separar features y target
    X, y = prep.split_features_target(df_clean, "price_in_euro")

    # Escalar características
    X_scaled = prep.transform(X)

    # Mostrar resultados
    print("✅ Datos limpios y escalados:")
    print(X_scaled.head())
    print("\n🎯 Variable objetivo:")
    print(y.head())
