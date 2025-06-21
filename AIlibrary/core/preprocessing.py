import pandas as pd
from sklearn.preprocessing import StandardScaler

class Preprocessor:
    """
    Clase para realizar la limpieza, transformación y escalado de datos para modelos de machine learning.
    """

    def __init__(self):
        # Escalador estándar de sklearn (aunque no se aplica por defecto)
        self.scaler = StandardScaler()

    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Limpia y transforma el DataFrame de entrada:
        - Elimina columnas irrelevantes
        - Reemplaza valores inválidos
        - Convierte textos numéricos
        - Codifica variables categóricas
        - Normaliza datos y elimina valores nulos y duplicados

        :param df: DataFrame original con datos crudos
        :return: DataFrame limpio y listo para el modelado
        """
        df = df.copy()

        # 1. Eliminar columnas que no aportan valor
        df.drop(columns=["Unnamed: 0", "offer_description"], inplace=True, errors="ignore")

        # 2. Reemplazar cadenas no válidas por valores nulos
        df.replace(["-", "", "NaN"], pd.NA, inplace=True)

        # 3. Normalizar valores de consumo de combustible (litros/100km)
        if "fuel_consumption_l_100km" in df.columns:
            df["fuel_consumption_l_100km"] = (
                df["fuel_consumption_l_100km"]
                .astype(str)                            # Convertir a string
                .str.replace(",", ".", regex=False)     # Usar punto decimal
                .str.extract(r"([\d.]+)")[0]            # Extraer número
                .astype(float)                          # Convertir a float
            )

        # 4. Extraer valor numérico de emisiones de CO2 (g/km)
        if "fuel_consumption_g_km" in df.columns:
            df["fuel_consumption_g_km"] = (
                df["fuel_consumption_g_km"]
                .astype(str)
                .str.extract(r"(\d+)")
                .astype(float)
            )

        # 5. Convertir fecha de registro a datetime y extraer año y mes
        if "registration_date" in df.columns:
            df["registration_date"] = pd.to_datetime(df["registration_date"], format="%m/%Y", errors="coerce")
            df["reg_year"] = df["registration_date"].dt.year
            df["reg_month"] = df["registration_date"].dt.month
            df.drop(columns=["registration_date"], inplace=True)

        # 6. Codificar variables categóricas como códigos numéricos en mayúsculas
        categorical_cols = ["brand", "model", "color", "transmission_type", "fuel_type"]
        for col in categorical_cols:
            if col in df.columns:
                df[col] = df[col].astype(str).str.upper().astype("category")
                mapping = dict(enumerate(df[col].cat.categories))
                # df[col] = mapeo como código entero
                df[col] = df[col].cat.codes

        # 7. Asegurar que las columnas numéricas estén correctamente tipadas
        numeric_cols = [
            "year", "price_in_euro", "power_kw", "power_ps",
            "fuel_consumption_l_100km", "fuel_consumption_g_km", "mileage_in_km"
        ]
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        # 8. Eliminar filas duplicadas
        df.drop_duplicates(inplace=True)

        # 9. Eliminar cualquier fila con valores nulos restantes
        df.dropna(inplace=True)

        return df

    def split_features_target(self, df: pd.DataFrame, target_column: str):
        """
        Separa el DataFrame en variables independientes (features) y dependiente (target).

        :param df: DataFrame limpio
        :param target_column: Nombre de la columna objetivo
        :return: Tuple (X, y) con features y target
        """
        df = df.dropna(subset=[target_column])  # Asegurar que la columna objetivo no tenga nulos
        X = df.drop(columns=[target_column])
        y = df[target_column]
        return X, y

    def transform(self, X: pd.DataFrame):
        """
        Aplica transformación al conjunto de características.
        Actualmente es un pass-through para mantener compatibilidad con pipelines que lo usen.

        :param X: DataFrame de características
        :return: Mismo DataFrame sin cambios
        """
        # Nota: Se podría aplicar self.scaler.fit_transform(X) si se quisiera escalar
        return X
