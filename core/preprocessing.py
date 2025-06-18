import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from typing import Tuple


class Preprocessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.ohe = OneHotEncoder(sparse_output=False, handle_unknown="ignore")

    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        # Eliminar columnas no útiles
        df.drop(columns=["Unnamed: 0", "offer_description"], inplace=True, errors="ignore")

        # Convertir fechas y extraer año y mes
        df["registration_date"] = pd.to_datetime(df["registration_date"], format="%m/%Y", errors="coerce")
        df["reg_year"] = df["registration_date"].dt.year
        df["reg_month"] = df["registration_date"].dt.month

        # Limpiar consumo de combustible
        def parse_fuel_l(val):
            if isinstance(val, str) and "l/100 km" in val:
                val = val.replace(" l/100 km", "").replace(",", ".").strip()
                try:
                    return float(val)
                except ValueError:
                    return None  # Valor no convertible como "- (l/100 km)"
            elif isinstance(val, str) and "kWh/100 km" in val:
                return 0.0  # Asumimos que el consumo en litros es 0 para autos eléctricos
            return None

        df["fuel_consumption_l_100km"] = df["fuel_consumption_l_100km"].apply(parse_fuel_l)

        # Limpiar emisiones CO₂
        df["fuel_consumption_g_km"] = (
            df["fuel_consumption_g_km"]
            .str.replace(" g/km", "", regex=False)
            .str.replace(",", ".", regex=False)
        )
        df["fuel_consumption_g_km"] = pd.to_numeric(df["fuel_consumption_g_km"], errors="coerce")

        # Convertir columnas numéricas
        numeric_cols = ["mileage_in_km", "power_kw", "power_ps", "price_in_euro", "year", "reg_year", "reg_month"]
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        # Eliminar filas con datos faltantes
        return df.dropna()

    def split_features_target(self, df: pd.DataFrame, target_column: str) -> Tuple[pd.DataFrame, pd.Series]:
        X = df.drop(columns=[target_column, "registration_date"])
        y = df[target_column]
        return X, y

    def transform(self, X: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        X = X.copy()

        # Separar numéricas y categóricas
        num_cols = X.select_dtypes(include="number").columns
        cat_cols = X.select_dtypes(include="object").columns

        # Escalar numéricas
        if fit:
            X_num = self.scaler.fit_transform(X[num_cols])
        else:
            X_num = self.scaler.transform(X[num_cols])

        df_num = pd.DataFrame(X_num, columns=num_cols, index=X.index)

        # Codificar categóricas
        if fit:
            X_cat = self.ohe.fit_transform(X[cat_cols])
        else:
            X_cat = self.ohe.transform(X[cat_cols])
        df_cat = pd.DataFrame(X_cat, columns=self.ohe.get_feature_names_out(cat_cols), index=X.index)

        # Unir todo
        return pd.concat([df_num, df_cat], axis=1)
