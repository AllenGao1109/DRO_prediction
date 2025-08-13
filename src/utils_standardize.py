from sklearn.preprocessing import StandardScaler
import pandas as pd


class DataNormalizer:
    def __init__(self, exclude_columns=None):
        self.exclude_columns = exclude_columns if exclude_columns else []
        self.scaler = None
        self.columns_to_scale = None

    def fit(self, df: pd.DataFrame):
        self.columns_to_scale = [
            col for col in df.columns if col not in self.exclude_columns
        ]
        self.scaler = StandardScaler()
        self.scaler.fit(df[self.columns_to_scale])

    def transform(self, df: pd.DataFrame):
        if self.scaler is None or self.columns_to_scale is None:
            raise ValueError("You must call fit() before transform()")

        df_scaled = df.copy()
        df_scaled[self.columns_to_scale] = self.scaler.transform(
            df[self.columns_to_scale]
        )
        return df_scaled

    def inverse_transform(self, df: pd.DataFrame):
        if self.scaler is None or self.columns_to_scale is None:
            raise ValueError("You must call fit() before inverse_transform()")

        df_original = df.copy()
        df_original[self.columns_to_scale] = self.scaler.inverse_transform(
            df[self.columns_to_scale]
        )
        return df_original

    def fit_transform(self, df: pd.DataFrame):
        self.fit(df)
        return self.transform(df)
