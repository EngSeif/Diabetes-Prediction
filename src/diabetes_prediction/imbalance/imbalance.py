import os
import pandas as pd
import numpy as np
from loguru import logger

# Imbalanced-learn - Resampling
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.combine import SMOTETomek, SMOTEENN


class DataImbalance:
    def __init__(self, df: pd.DataFrame, target_col="diabetes"):
        self.df = df
        self.target_col = target_col
        self.y_train = self.df[self.target_col]
        self.X_train = self.df.drop(self.target_col, axis=1)
        self.output_dir = os.path.join(
            os.path.dirname(__file__), "..", "..", "data", "imbalance_resolve"
        )
        os.makedirs(self.output_dir, exist_ok=True)
    
    def _to_dataframe(self, X_res, y_res):
        X_res = pd.DataFrame(X_res, columns=self.X_train.columns)
        y_res = pd.Series(y_res, name=self.target_col)

        df_res = pd.concat([X_res, y_res], axis=1)

        df_res = df_res.apply(pd.to_numeric, errors="coerce")
        df_res = df_res.dropna()

        return df_res

    def _save(self, x, y, filename: str):
        pd.concat(
            [
                pd.DataFrame(x, columns=self.X_train.columns),
                pd.Series(list(y), name="diabetes_target"),
            ],
            axis=1,
        ).to_csv(os.path.join(self.output_dir, filename), index=False)

    def adasyn(self):
        model = ADASYN(random_state=42)
        X_res, y_res = model.fit_resample(self.X_train, self.y_train)
        return self._to_dataframe(X_res, y_res)

    def smote_df(self):
        smote = SMOTE(random_state=42)
        result = smote.fit_resample(self.X_train, self.y_train)
        self._save(result[0], result[1], "train_smote.csv")
        return self

    def smote_tomek_df(self):
        smote_tomek = SMOTETomek(random_state=42)
        result = smote_tomek.fit_resample(self.X_train, self.y_train)
        self._save(result[0], result[1], "train_smote_tomek.csv")

    def smote_enn_df(self):
        smote_enn = SMOTEENN(random_state=42)
        result = smote_enn.fit_resample(self.X_train, self.y_train)
        self._save(result[0], result[1], "train_smote_enn.csv")

    def run_pipeline(self):
        self.adaysn_df()
        self.smote_df()
        self.smote_enn_df()
        self.smote_tomek_df()
