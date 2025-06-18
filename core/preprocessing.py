import pandas as pd
from sklearn.preprocessing import StandardScaler

class Preprocessor:
    def __init__(self):
        self.scaler = StandardScaler()

    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.dropna()
        return df

    def transform(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        if fit:
            df_scaled = self.scaler.fit_transform(df)
        else:
            df_scaled = self.scaler.transform(df)
        return pd.DataFrame(df_scaled, columns=df.columns)
