import pandas as pd

class Preprocessor:
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        # 1. Eliminar columnas irrelevantes
        df.drop(columns=["Unnamed: 0", "offer_description"], inplace=True, errors="ignore")

        # 2. Reemplazar valores no válidos con NA y eliminar filas con nulos
        df.replace(["-", "", "NaN"], pd.NA, inplace=True)
        df.dropna(inplace=True)

        # 3. Normalizar consumo de combustible (litros/100km)
        if "fuel_consumption_l_100km" in df.columns:
            df["fuel_consumption_l_100km"] = (
                df["fuel_consumption_l_100km"]
                .astype(str)
                .str.replace(",", ".", regex=False)
                .str.extract(r"([\d.]+)")[0]
                .astype(float)
            )

        # 4. Limpiar columna de emisiones de CO2
        if "fuel_consumption_g_km" in df.columns:
            df["fuel_consumption_g_km"] = (
                df["fuel_consumption_g_km"]
                .astype(str)
                .str.extract(r"(\d+)")
                .astype(float)
            )

        # 5. Convertir 'registration_date' a datetime y extraer año/mes
        if "registration_date" in df.columns:
            df["registration_date"] = pd.to_datetime(df["registration_date"], format="%m/%Y", errors="coerce")
            df["reg_year"] = df["registration_date"].dt.year
            df["reg_month"] = df["registration_date"].dt.month
            df.drop(columns=["registration_date"], inplace=True)

        # 6. Codificación de variables categóricas
        categorical_cols = ["brand", "model", "color", "transmission_type", "fuel_type"]
        for col in categorical_cols:
            if col in df.columns:
                df[col] = df[col].astype("category")
                mapping = dict(enumerate(df[col].cat.categories))
                print((col, mapping))  # Mostrar la tupla (columna, mapeo)
                df[col] = df[col].cat.codes

        # 7. Validar y convertir datos numéricos
        numeric_cols = [
            "year", "price_in_euro", "power_kw", "power_ps",
            "fuel_consumption_l_100km", "fuel_consumption_g_km", "mileage_in_km"
        ]
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        # 8. Eliminar duplicados
        df.drop_duplicates(inplace=True)

        return df