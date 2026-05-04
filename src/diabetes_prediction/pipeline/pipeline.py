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
import os
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
from diabetes_prediction.imbalance.imbalance import DataImbalance
from diabetes_prediction.selection.selection import FeatureSelection


PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))


class DiabetesFullPipeline:
    def __init__(
        self,
        raw_data_path: str,
        target_col: str = "diabetes",
        random_state: int = 42,
    ):
        self.raw_data_path = Path(raw_data_path)
        self.target_col = target_col
        self.random_state = random_state

        self.transformer = DataTransformation()
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            class_weight=None,
            random_state=random_state,
            n_jobs=-1,
            )
        self.selected_columns = None
        

    
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


        X_val, X_test, y_val, y_test = train_test_split(
            X_temp,
            y_temp,
            test_size=0.5,
            stratify=y_temp,
            random_state=self.random_state,
        )

        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def transform_data(self, X_train, X_val, X_test):
        X_train_transformed = self.transformer.fit_transform(X_train)
        X_val_transformed = self.transformer.transform(X_val)
        X_test_transformed = self.transformer.transform(X_test)

        return X_train_transformed, X_val_transformed, X_test_transformed
    
    def select_features(self, X_train, X_val, X_test):
        """
        Apply feature selection to drop unneeded columns.
        """
        selector = FeatureSelection(X_train)
        X_train_sel = selector.drop_features()
        self.selected_columns = X_train_sel.columns

        X_val_sel = X_val[self.selected_columns]
        X_test_sel = X_test[self.selected_columns]
        return X_train_sel, X_val_sel, X_test_sel
        

    

    def handle_imbalance(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply ADASYN to balance the classes.
        Returns a new DataFrame with oversampled minority class.
        """
        imbalance_handler = DataImbalance(df)
        imbalance_handler.adaysn_df()  # creates a CSV in data/imbalance_resolve/ADASYN.csv
        # Load the oversampled dataset back
        balanced_df = pd.read_csv(
            os.path.join(imbalance_handler.output_dir,"ADASYN.csv")
        )
        balanced_df = balanced_df.rename(
            columns={"diabetes_target": self.target_col}
        )
        return balanced_df

    def train_model(self, x, y):
        self.model.fit(x, y)
        model_path = PROJECT_ROOT / "models" / "final.pkl"
        model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, model_path)
        joblib.dump(
            self.selected_columns,
            PROJECT_ROOT / "models" / "selected_columns.pkl"
        )
    

    
    def evaluate_model(self, x, y, dataset_name: str = "Dataset") -> Dict:
        y_pred = self.model.predict(x)

        results = {
            "dataset": dataset_name,
            "accuracy": accuracy_score(y, y_pred),
            "precision": precision_score(y, y_pred, zero_division=0),
            "recall": recall_score(y, y_pred, zero_division=0),
            "f1_score": f1_score(y, y_pred, zero_division=0),
            "confusion_matrix": confusion_matrix(y, y_pred).tolist(),
            "classification_report": classification_report(y, y_pred, zero_division=0),
        }

        if hasattr(self.model, "predict_proba"):
            y_proba = self.model.predict_proba(x)[:, 1]
            results["roc_auc"] = roc_auc_score(y, y_proba)

        return results
    
    def load_artifacts(self):
        self.model = joblib.load(PROJECT_ROOT / "models" / "final.pkl")
        self.selected_columns = joblib.load(PROJECT_ROOT / "models" / "selected_columns.pkl")
        self.transformer.load_preprocessor()
        return self
    
    def predict_one(self, sample: dict) -> Dict:
        X_sample_transformed = self.transformer.transform_one(sample)
        X_sample_transformed = X_sample_transformed[self.selected_columns]
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

        X_train, X_val, X_test = self.transform_data(
            X_train,
            X_val,
            X_test,
        )
        X_train, X_val, X_test = self.select_features(
            X_train, X_val, X_test
        )

        train_df = X_train.copy()
        train_df[self.target_col] = y_train.values
        train_df = self.handle_imbalance(train_df)
        X_train_final = train_df.drop(columns=[self.target_col])
        y_train_final = train_df[self.target_col]

        self.train_model(X_train_final, y_train_final)

        validation_results = self.evaluate_model(
            X_val,
            y_val,
            dataset_name="Validation",
        )

        test_results = self.evaluate_model(
            X_test,
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
    



def predict_single_sample(user_input: dict) -> dict:
    
    try:
        pipeline = DiabetesFullPipeline(
                raw_data_path="data/raw/diabetes_prediction_dataset.csv",    
        )
        pipeline.load_artifacts()
        result = pipeline.predict_one(user_input)
        return {"success": True, "result": result}
    except Exception as e:
        return {"success": False, "error": str(e)}


if __name__ == "__main__":
    pipeline = DiabetesFullPipeline(
        raw_data_path="data/raw/diabetes_prediction_dataset.csv",
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
