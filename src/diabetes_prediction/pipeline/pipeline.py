"""
1) Load raw data
2) Validate raw/cleaned data 
3) Clean data
4) Split train/validation/test
5) Feature engineering + preprocessing
6) Train model
7) Evaluate model
8) Save/load full pipeline
9) Predict for one new raw sample
"""
import sys
from sklearn.ensemble import RandomForestClassifier
from pathlib import Path
from typing import Dict, Union
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)

from diabetes_prediction.cleaning.cleaning import DataCleaning
from diabetes_prediction.transformation.transformation import DataTransformation


project_root = Path(__file__).resolve().parents[2] 
sys.path.insert(0, str(project_root))


class DiabetesFullPipeline:
    def __init__(
        self,
        raw_data_path: str,
        target_col: str = "diabetes",
        model_path: str = "models/diabetes_model.pkl",
        preprocessor_path: str = "/notebooks/Transformation/preprocessor.pkl",
        random_state: int = 42,
    ):
        self.raw_data_path = Path(raw_data_path)
        self.target_col = target_col
        self.model_path = Path(model_path)
        self.preprocessor_path = Path(preprocessor_path)
        self.random_state = random_state

        self.transformer = DataTransformation()

        self.model = joblib.load(model_path)
        

    
    def load_data(self) -> pd.DataFrame:
        return pd.read_csv(self.raw_data_path)
    

    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        clean_df = DataCleaning(df).run_cleaning()
        return clean_df.reset_index(drop=True)
    
    def split_data(self, df: pd.DataFrame, test_size: float = 0.15, val_size: float = 0.15):
        X = df.drop(columns=[self.target_col])
        y = df[self.target_col]

        temp_size = test_size + val_size

        X_train, X_temp, y_train, y_temp = train_test_split(
            X,
            y,
            test_size=temp_size,
            stratify=y,
            random_state=self.random_state,
        )

        relative_test_size = test_size / temp_size

        X_val, X_test, y_val, y_test = train_test_split(
            X_temp,
            y_temp,
            test_size=relative_test_size,
            stratify=y_temp,
            random_state=self.random_state,
        )

        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def transform_data(self, X_train, X_val, X_test):
        X_train_transformed = self.transformer.fit_transform(X_train)
        X_val_transformed = self.transformer.transform(X_val)
        X_test_transformed = self.transformer.transform(X_test)

        return X_train_transformed, X_val_transformed, X_test_transformed

    def train_model(self, X_train_transformed, y_train):
        self.model.fit(X_train_transformed, y_train)
        return self
    
    def evaluate_model(self, X_transformed, y_true, dataset_name: str = "Dataset") -> Dict:
        y_pred = self.model.predict(X_transformed)

        results = {
            "dataset": dataset_name,
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, zero_division=0),
            "recall": recall_score(y_true, y_pred, zero_division=0),
            "f1_score": f1_score(y_true, y_pred, zero_division=0),
            "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
            "classification_report": classification_report(y_true, y_pred, zero_division=0),
        }

        if hasattr(self.model, "predict_proba"):
            y_proba = self.model.predict_proba(X_transformed)[:, 1]
            results["roc_auc"] = roc_auc_score(y_true, y_proba)

        return results
    
    def load_artifacts(self):
        self.model = joblib.load(self.model_path)
        self.transformer.load_preprocessor(self.preprocessor_path)
        return self
    
    def predict_one(self, sample: Union[dict, pd.Series, pd.DataFrame]) -> Dict:
        X_sample_transformed = self.transformer.transform_one(sample)

        prediction = int(self.model.predict(X_sample_transformed)[0])

        result = {
            "prediction": prediction,
        }

        if hasattr(self.model, "predict_proba"):
            probability = float(self.model.predict_proba(X_sample_transformed)[0][1])
            result["diabetes_probability"] = probability

        return result
    
    def run(self):
        raw_df = self.load_data()
        clean_df = self.clean_data(raw_df)

        X_train, X_val, X_test, y_train, y_val, y_test = self.split_data(clean_df)

        X_train_transformed, X_val_transformed, X_test_transformed = self.transform_data(
            X_train,
            X_val,
            X_test,
        )

        self.train_model(X_train_transformed, y_train)

        validation_results = self.evaluate_model(
            X_val_transformed,
            y_val,
            dataset_name="Validation",
        )

        test_results = self.evaluate_model(
            X_test_transformed,
            y_test,
            dataset_name="Test",
        )

        

        return {
            "raw_shape": raw_df.shape,
            "clean_shape": clean_df.shape,
            "train_shape": X_train.shape,
            "validation_shape": X_val.shape,
            "test_shape": X_test.shape,
            "validation_results": validation_results,
            "test_results": test_results,
            
        }
    

if __name__ == "__main__":
    pipeline = DiabetesFullPipeline(
        raw_data_path="data/raw/diabetes_prediction_dataset.csv",
        model_path="models/RF_model.pkl",
        preprocessor_path="notebooks/Transformation/preprocessor.pkl",
    )

    results = pipeline.run()

    print("\\nPipeline completed successfully")
    print("Raw shape:", results["raw_shape"])
    print("Clean shape:", results["clean_shape"])
    print("Train shape:", results["train_shape"])
    print("Validation shape:", results["validation_shape"])
    print("Test shape:", results["test_shape"])

    print("\\nValidation results:")
    for key, value in results["validation_results"].items():
        if key != "classification_report":
            print(f"{key}: {value}")

    print("\\nTest results:")
    for key, value in results["test_results"].items():
        if key != "classification_report":
            print(f"{key}: {value}")

    example_patient = {
        "gender": "Female",
        "age": 55,
        "hypertension": 0,
        "heart_disease": 0,
        "smoking_history": "never",
        "bmi": 29.5,
        "HbA1c_level": 6.1,
        "blood_glucose_level": 160,
    }

    print("\\nExample final prediction:")
    print(pipeline.predict_one(example_patient))
