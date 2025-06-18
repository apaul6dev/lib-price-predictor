import pandas as pd

class Preprocessor:
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        initial_rows = df.shape[0]

        # Eliminar columnas irrelevantes
        df.drop(columns=["Unnamed: 0", "offer_description"], inplace=True, errors="ignore")

        # Eliminar filas con valores nulos
        rows_before_na = df.shape[0]
        df = df.dropna()
        na_dropped = rows_before_na - df.shape[0]

        # Eliminar filas duplicadas
        rows_before_dup = df.shape[0]
        df = df.drop_duplicates()
        dup_dropped = rows_before_dup - df.shape[0]


        # Resumen de limpieza
        print(f"🔍 Filas iniciales: {initial_rows}")
        print(f"🧼 Filas eliminadas por valores nulos: {na_dropped}")
        print(f"🧹 Filas eliminadas por duplicados: {dup_dropped}")
        print(f"✅ Filas finales: {df.shape[0]}")
        print(f"📊 Columnas finales: {df.shape[1]}")

        return df
