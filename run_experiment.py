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
 
    print(df_clean.head(10))

