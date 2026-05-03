import os
import pandas as pd
import numpy as np
from loguru import logger

# Imbalanced-learn - Resampling
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.combine import SMOTETomek, SMOTEENN


class DataImbalance:
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.y_train = self.df["diabetes"].copy()
        self.X_train = self.df.drop("diabetes", axis=1)
        self.output_dir = os.path.join(
            os.path.dirname(__file__), "..", "..", "data", "imbalance_resolve"
        )
        os.makedirs(self.output_dir, exist_ok=True)

    def _save(self, x, y, filename: str):
        pd.concat(
            [
                pd.DataFrame(x, columns=self.X_train.columns),
                pd.Series(list(y), name="diabetes_target"),
            ],
            axis=1,
        ).to_csv(os.path.join(self.output_dir, filename), index=False)

    def adaysn_df(self):
        adasyn = ADASYN(random_state=42)
        result = adasyn.fit_resample(self.X_train, self.y_train)
        self._save(result[0], result[1], "ADASYN.csv")

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
