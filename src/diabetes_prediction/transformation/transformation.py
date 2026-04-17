import pandas as pd
import joblib

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, RobustScaler


class DataTransformation:
    def __init__(self):
        self.preprocessor = ColumnTransformer(
            transformers=[
                ("smoking_ohe", OneHotEncoder(handle_unknown="ignore"), ["smoking_history"]),
                ("age_minmax", MinMaxScaler(), ["age"]),
                ("robust_features", RobustScaler(), [
                    "bmi",
                    "HbA1c_level",
                    "blood_glucose_level",
                    "glucose_hba1c_interaction",
                    "age_hba1c_interaction",
                    "age_bmi_interaction",
                    "bmi_hba1c_interaction",
                    "age_glucose_interaction",
                    "metabolic_load_score"
                ])
            ],
            remainder="passthrough",
            verbose_feature_names_out=False
        )
        self.feature_names_ = None

    def encode_gender(self, df: pd.DataFrame):
        df = df.copy()
        df["gender"] = df["gender"].replace({"Female": 0, "Male": 1})
        return df

    def add_engineered_features(self, df: pd.DataFrame):
        df = df.copy()

        df["glucose_hba1c_interaction"] = df["blood_glucose_level"] * df["HbA1c_level"]
        df["age_hba1c_interaction"] = df["age"] * df["HbA1c_level"]
        df["age_bmi_interaction"] = df["age"] * df["bmi"]
        df["bmi_hba1c_interaction"] = df["bmi"] * df["HbA1c_level"]
        df["age_glucose_interaction"] = df["age"] * df["blood_glucose_level"]

        df["high_hba1c_flag"] = (df["HbA1c_level"] >= 6.5).astype(int)
        df["senior_flag"] = (df["age"] >= 60).astype(int)

        df["cardio_risk_flag"] = ((df["hypertension"] == 1) | (df["heart_disease"] == 1)).astype(int)

        return df

    def prepare_features(self, df: pd.DataFrame):
        df = df.copy()
        df = self.encode_gender(df)
        df = self.add_engineered_features(df)
        return df

    def fit_transform(self, x_train: pd.DataFrame):
        x_train = self.prepare_features(x_train)

        transformed = self.preprocessor.fit_transform(x_train)
        self.feature_names_ = self.preprocessor.get_feature_names_out()

        return pd.DataFrame(
            transformed,
            columns=self.feature_names_,
            index=x_train.index
        )

    def transform(self, x: pd.DataFrame):
        x = self.prepare_features(x)

        transformed = self.preprocessor.transform(x)

        return pd.DataFrame(
            transformed,
            columns=self.feature_names_,
            index=x.index
        )

    def save_preprocessor(self, path="preprocessor.pkl"):
        joblib.dump(self.preprocessor, path)

    def load_preprocessor(self, path="preprocessor.pkl"):
        self.preprocessor = joblib.load(path)