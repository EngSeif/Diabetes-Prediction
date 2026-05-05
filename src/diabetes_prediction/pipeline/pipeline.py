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
from typing import Dict
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from loguru import logger

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)

from src.diabetes_prediction.cleaning.cleaning import DataCleaning
from src.diabetes_prediction.transformation.transformation import DataTransformation
from src.diabetes_prediction.imbalance.imbalance import DataImbalance
from src.diabetes_prediction.selection.selection import FeatureSelection

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

# ---------------------------------------------------------------------------
# Logger configuration
# ---------------------------------------------------------------------------
logger.remove()  # remove the default handler (no console output)
logger.add(
    PROJECT_ROOT / "../../../reports" / "pipeline.log",
    rotation="10 MB",
    retention="30 days",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {extra[stage]} | {message}",
    level="DEBUG",
)


# Convenience: a stage-bound logger factory
def stage_logger(stage: str):
    """Return a logger pre-bound to the given stage name."""
    return logger.bind(stage=stage)


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

    # ------------------------------------------------------------------
    # 1. Load data
    # ------------------------------------------------------------------
    def load_data(self) -> pd.DataFrame:
        log = stage_logger("LOAD DATA")
        log.info("Reading raw CSV from '{}'", self.raw_data_path)
        df = pd.read_csv(self.raw_data_path)
        log.info("Loaded {} rows × {} columns", *df.shape)
        return df

    # ------------------------------------------------------------------
    # 3. Clean data
    # ------------------------------------------------------------------
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        log = stage_logger("CLEAN DATA")
        log.info("Starting data cleaning on {} rows", len(df))
        clean_df = DataCleaning(df).run_cleaning()
        clean_df = clean_df.reset_index(drop=True)
        log.info(
            "Cleaning complete — {} rows remaining ({} dropped)",
            len(clean_df),
            len(df) - len(clean_df),
        )
        return clean_df

    # ------------------------------------------------------------------
    # 4. Split data
    # ------------------------------------------------------------------
    def split_data(
        self, df: pd.DataFrame, test_size: float = 0.15, val_size: float = 0.15
    ):
        log = stage_logger("SPLIT DATA")
        log.info(
            "Splitting data (test={:.0%}, val={:.0%}, train={:.0%})",
            test_size,
            val_size,
            1 - test_size - val_size,
        )
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

        log.info(
            "Split sizes — train: {}, val: {}, test: {}",
            len(X_train),
            len(X_val),
            len(X_test),
        )
        return X_train, X_val, X_test, y_train, y_val, y_test

    # ------------------------------------------------------------------
    # 5a. Transform data
    # ------------------------------------------------------------------
    def transform_data(self, X_train, X_val, X_test):
        log = stage_logger("TRANSFORM DATA")
        log.info("Fitting transformer on training set ({} rows)", len(X_train))
        X_train_transformed = self.transformer.fit_transform(X_train)
        log.debug("Transforming validation set")
        X_val_transformed = self.transformer.transform(X_val)
        log.debug("Transforming test set")
        X_test_transformed = self.transformer.transform(X_test)
        log.info(
            "Transformation complete — output shape: train={}, val={}, test={}",
            X_train_transformed.shape,
            X_val_transformed.shape,
            X_test_transformed.shape,
        )
        return X_train_transformed, X_val_transformed, X_test_transformed

    # ------------------------------------------------------------------
    # 5b. Feature selection
    # ------------------------------------------------------------------
    def select_features(self, X_train, X_val, X_test):
        log = stage_logger("FEATURE SELECTION")
        log.info("Running feature selection on {} columns", X_train.shape[1])
        selector = FeatureSelection(X_train)
        X_train_sel = selector.drop_features()
        self.selected_columns = X_train_sel.columns

        X_val_sel = X_val[self.selected_columns]
        X_test_sel = X_test[self.selected_columns]
        log.info(
            "Feature selection complete — {} columns selected: {}",
            len(self.selected_columns),
            list(self.selected_columns),
        )
        return X_train_sel, X_val_sel, X_test_sel

    # ------------------------------------------------------------------
    # Imbalance handling
    # ------------------------------------------------------------------
    def handle_imbalance(self, df: pd.DataFrame) -> pd.DataFrame:
        log = stage_logger("HANDLE IMBALANCE")
        class_counts = df[self.target_col].value_counts().to_dict()
        log.info("Class distribution before ADASYN: {}", class_counts)
        imbalance_handler = DataImbalance(df)
        balanced_df = imbalance_handler.adasyn()
        balanced_df = balanced_df.rename(columns={"diabetes_target": self.target_col})
        class_counts_after = balanced_df[self.target_col].value_counts().to_dict()
        log.info(
            "Class distribution after ADASYN: {} (total rows: {})",
            class_counts_after,
            len(balanced_df),
        )
        return balanced_df

    # ------------------------------------------------------------------
    # 6. Train model
    # ------------------------------------------------------------------
    def train_model(self, x, y):
        log = stage_logger("TRAIN MODEL")
        log.info(
            "Training RandomForestClassifier on {} samples with {} features",
            len(x),
            x.shape[1],
        )
        self.model.fit(x, y)
        model_path = PROJECT_ROOT / "models" / "final.pkl"
        model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, model_path)
        joblib.dump(
            self.selected_columns, PROJECT_ROOT / "models" / "selected_columns.pkl"
        )
        log.info("Model and selected columns saved to '{}'", model_path.parent)

    # ------------------------------------------------------------------
    # 7. Evaluate model
    # ------------------------------------------------------------------
    def evaluate_model(self, x, y, dataset_name: str = "Dataset") -> Dict:
        log = stage_logger("EVALUATE MODEL")
        log.info("Evaluating model on '{}' set ({} samples)", dataset_name, len(y))
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

        log.info(
            "[{}] accuracy={:.4f} | precision={:.4f} | recall={:.4f} | f1={:.4f} | roc_auc={:.4f}",
            dataset_name,
            results["accuracy"],
            results["precision"],
            results["recall"],
            results["f1_score"],
            results.get("roc_auc", float("nan")),
        )
        return results

    # ------------------------------------------------------------------
    # 8. Load artifacts
    # ------------------------------------------------------------------
    def load_artifacts(self):
        log = stage_logger("LOAD ARTIFACTS")
        log.info("Loading model, selected columns, and preprocessor from disk")
        self.model = joblib.load(PROJECT_ROOT / "models" / "final.pkl")
        self.selected_columns = joblib.load(
            PROJECT_ROOT / "models" / "selected_columns.pkl"
        )
        self.transformer.load_preprocessor()
        log.info("Artifacts loaded successfully")
        return self

    # ------------------------------------------------------------------
    # 9. Predict one sample
    # ------------------------------------------------------------------
    def predict_one(self, sample: dict) -> Dict:
        log = stage_logger("PREDICT ONE")
        log.info("Predicting for sample: {}", sample)
        X_sample_transformed = self.transformer.transform_one(sample)
        X_sample_transformed = X_sample_transformed[self.selected_columns]
        prediction = int(self.model.predict(X_sample_transformed)[0])

        result = {
            "prediction": prediction,
        }

        if hasattr(self.model, "predict_proba"):
            probability = float(self.model.predict_proba(X_sample_transformed)[0][1])
            result["diabetes_probability"] = probability

        log.info("Prediction result: {}", result)
        return result

    # ------------------------------------------------------------------
    # Full pipeline run
    # ------------------------------------------------------------------
    def run(self):
        log = stage_logger("PIPELINE")
        log.info("=" * 60)
        log.info("Starting DiabetesFullPipeline")
        log.info("=" * 60)

        raw_df = self.load_data()
        clean_df = self.clean_data(raw_df)

        X_train, X_val, X_test, y_train, y_val, y_test = self.split_data(clean_df)

        X_train, X_val, X_test = self.transform_data(
            X_train,
            X_val,
            X_test,
        )
        X_train, X_val, X_test = self.select_features(X_train, X_val, X_test)

        train_df = X_train.copy()
        train_df = train_df.apply(pd.to_numeric, errors="coerce")
        train_df = train_df.dropna()
        # Align y_train to the same index that survived dropna()
        y_train_aligned = y_train.loc[train_df.index]
        train_df[self.target_col] = y_train_aligned
        train_df = self.handle_imbalance(train_df)
        X_train_final = train_df.drop(columns=[self.target_col])
        y_train_final = train_df[self.target_col]

        self.train_model(X_train_final, y_train_final)

        validation_results = self.evaluate_model(
            X_val[self.selected_columns],
            y_val,
            dataset_name="Validation",
        )

        test_results = self.evaluate_model(
            X_test[self.selected_columns],
            y_test,
            dataset_name="Test",
        )

        log.info("=" * 60)
        log.info("DiabetesFullPipeline finished successfully")
        log.info("=" * 60)

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
    """
    Predict diabetes for a single raw sample using the notebook-trained
    preprocessor (preprocessor.pkl) and Random Forest model (RF_model.pkl).

    This bypasses the full pipeline and directly uses the same artifacts
    that the Random_Forest.ipynb notebook produced, ensuring identical
    preprocessing and prediction behaviour.
    """
    log = stage_logger("PREDICT SINGLE SAMPLE")
    log.info("Predicting for input: {}", user_input)
    try:
        # --- paths relative to project root ---
        preprocessor_path = (
            PROJECT_ROOT / "notebooks" / "Transformation" / "preprocessor.pkl"
        )
        model_path = PROJECT_ROOT / "models" / "RF_model.pkl"

        # --- load artifacts ---
        log.debug("Loading preprocessor from '{}'", preprocessor_path)
        preprocessor = joblib.load(preprocessor_path)
        log.debug("Loading model from '{}'", model_path)
        model = joblib.load(model_path)

        # --- build a one-row DataFrame from the raw dict ---
        transformer = DataTransformation()
        transformer.preprocessor = preprocessor  # use the saved preprocessor

        log.debug("Transforming input sample")
        X_sample = transformer.transform_one(user_input)  # prepare_features + transform

        # --- apply the same feature selection the notebook used ---
        cols_to_drop = [
            "age",
            "blood_glucose_level",
            "age_bmi_interaction",
            "age_hba1c_interaction",
            "hypertension",
        ]
        # only drop columns that actually exist (some may have been
        # removed already by the preprocessor's remainder handling)
        cols_to_drop = [c for c in cols_to_drop if c in X_sample.columns]
        log.debug("Dropping columns: {}", cols_to_drop)
        X_sample = X_sample.drop(columns=cols_to_drop)

        # --- predict ---
        prediction = int(model.predict(X_sample)[0])

        result = {"prediction": prediction}

        if hasattr(model, "predict_proba"):
            probability = float(model.predict_proba(X_sample)[0][1])
            result["diabetes_probability"] = probability

        log.info("Prediction result: {}", result)
        return {"success": True, "result": result}

    except Exception as e:
        log.error("Prediction failed: {}", e)
        return {"success": False, "error": str(e)}


if __name__ == "__main__":
    pipeline = DiabetesFullPipeline(
        raw_data_path="data/raw/diabetes_prediction_dataset.csv",
    )

    results = pipeline.run()

    print("\nPipeline completed successfully")
    print("Raw shape:", results["raw_shape"])
    print("Clean shape:", results["clean_shape"])
    print("Train shape:", results["train_shape"])
    print("Validation shape:", results["validation_shape"])
    print("Test shape:", results["test_shape"])

    print("\nValidation results:")
    for key, value in results["validation_results"].items():
        if key != "classification_report":
            print(f"{key}: {value}")

    print("\nTest results:")
    for key, value in results["test_results"].items():
        if key != "classification_report":
            print(f"{key}: {value}")

    # --- quick smoke-test of predict_single_sample ---
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
    # print("\nExample prediction (via notebook artifacts):")
    # print(predict_single_sample(example_patient))
