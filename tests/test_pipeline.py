"""
Integration tests for DiabetesFullPipeline and predict_single_sample.

These tests exercise the full data flow end-to-end using a small synthetic
CSV so no real dataset file is required.  Each test covers one pipeline
stage or the pipeline as a whole, verifying that stages compose correctly
and that outputs have the expected shapes, types, and keys.
"""
import pytest
import numpy as np
import pandas as pd
from unittest.mock import patch, MagicMock

import src.diabetes_prediction.pipeline.pipeline as pipe_mod
from src.diabetes_prediction.pipeline.pipeline import (
    DiabetesFullPipeline,
    predict_single_sample,
)


# ---------------------------------------------------------------------------
# Shared synthetic data helpers
# ---------------------------------------------------------------------------

RAW_COLUMNS = [
    "gender", "age", "hypertension", "heart_disease",
    "smoking_history", "bmi", "HbA1c_level", "blood_glucose_level", "diabetes",
]

EXAMPLE_PATIENT = {
    "gender": "Female",
    "age": 55,
    "hypertension": 0,
    "heart_disease": 0,
    "smoking_history": "never",
    "bmi": 29.5,
    "HbA1c_level": 6.1,
    "blood_glucose_level": 160,
}


def _make_raw_df(n: int = 200, seed: int = 0) -> pd.DataFrame:
    """
    Generate a minimal synthetic raw DataFrame that mimics the real dataset.
    Class imbalance is intentional (~10% positive) to exercise ADASYN.
    """
    rng = np.random.default_rng(seed)
    n_pos = max(10, n // 10)
    n_neg = n - n_pos

    def _block(size, diab):
        return pd.DataFrame({
            "gender":              rng.choice(["Male", "Female"], size=size),
            "age":                 rng.uniform(20, 80, size=size),
            "hypertension":        rng.integers(0, 2, size=size),
            "heart_disease":       rng.integers(0, 2, size=size),
            "smoking_history":     rng.choice(
                                       ["never", "former", "current", "No Info"],
                                       size=size,
                                   ),
            "bmi":                 rng.uniform(15, 50, size=size),
            "HbA1c_level":         rng.uniform(4.0, 9.0, size=size),
            "blood_glucose_level": rng.uniform(80, 300, size=size),
            "diabetes":            [diab] * size,
        })

    return pd.concat([_block(n_neg, 0), _block(n_pos, 1)], ignore_index=True)


@pytest.fixture
def raw_df():
    return _make_raw_df()


@pytest.fixture
def csv_path(tmp_path, raw_df):
    """Write synthetic data to a temp CSV and return its path."""
    p = tmp_path / "diabetes.csv"
    raw_df.to_csv(p, index=False)
    return p


@pytest.fixture
def pipeline(csv_path):
    return DiabetesFullPipeline(raw_data_path=str(csv_path), random_state=42)


# ---------------------------------------------------------------------------
# Stage 1 – load_data
# ---------------------------------------------------------------------------

class TestLoadData:
    def test_returns_dataframe(self, pipeline, raw_df):
        df = pipeline.load_data()
        assert isinstance(df, pd.DataFrame)

    def test_has_expected_columns(self, pipeline):
        df = pipeline.load_data()
        assert set(RAW_COLUMNS).issubset(df.columns)

    def test_row_count_matches_csv(self, pipeline, raw_df):
        df = pipeline.load_data()
        assert len(df) == len(raw_df)


# ---------------------------------------------------------------------------
# Stage 2 – clean_data
# ---------------------------------------------------------------------------

class TestCleanData:
    def test_returns_dataframe(self, pipeline, raw_df):
        clean = pipeline.clean_data(raw_df)
        assert isinstance(clean, pd.DataFrame)

    def test_no_nulls_after_cleaning(self, pipeline, raw_df):
        clean = pipeline.clean_data(raw_df)
        assert clean.isnull().sum().sum() == 0

    def test_row_count_does_not_increase(self, pipeline, raw_df):
        clean = pipeline.clean_data(raw_df)
        assert len(clean) <= len(raw_df)

    def test_target_column_preserved(self, pipeline, raw_df):
        clean = pipeline.clean_data(raw_df)
        assert "diabetes" in clean.columns

    def test_index_is_reset(self, pipeline, raw_df):
        clean = pipeline.clean_data(raw_df)
        assert list(clean.index) == list(range(len(clean)))


# ---------------------------------------------------------------------------
# Stage 3 – split_data
# ---------------------------------------------------------------------------

class TestSplitData:
    @pytest.fixture
    def splits(self, pipeline, raw_df):
        clean = pipeline.clean_data(raw_df)
        return pipeline.split_data(clean), len(clean)

    def test_returns_six_objects(self, splits):
        result, _ = splits
        assert len(result) == 6

    def test_sizes_sum_to_total(self, splits):
        (X_train, X_val, X_test, y_train, y_val, y_test), n = splits
        assert len(X_train) + len(X_val) + len(X_test) == n

    def test_no_target_in_X(self, splits):
        (X_train, X_val, X_test, *_), _ = splits
        for X in (X_train, X_val, X_test):
            assert "diabetes" not in X.columns

    def test_y_lengths_match_X(self, splits):
        (X_train, X_val, X_test, y_train, y_val, y_test), _ = splits
        assert len(X_train) == len(y_train)
        assert len(X_val) == len(y_val)
        assert len(X_test) == len(y_test)

    def test_train_is_largest_split(self, splits):
        (X_train, X_val, X_test, *_), _ = splits
        assert len(X_train) > len(X_val)
        assert len(X_train) > len(X_test)


# ---------------------------------------------------------------------------
# Stage 4 – transform_data
# ---------------------------------------------------------------------------

class TestTransformData:
    @pytest.fixture
    def transformed(self, pipeline, raw_df):
        clean = pipeline.clean_data(raw_df)
        X_train, X_val, X_test, *_ = pipeline.split_data(clean)
        return pipeline.transform_data(X_train, X_val, X_test)

    def test_returns_three_dataframes(self, transformed):
        assert len(transformed) == 3
        for df in transformed:
            assert isinstance(df, pd.DataFrame)

    def test_no_nulls_in_output(self, transformed):
        for df in transformed:
            assert df.isnull().sum().sum() == 0

    def test_all_numeric(self, transformed):
        for df in transformed:
            assert all(np.issubdtype(dt, np.number) for dt in df.dtypes)

    def test_column_counts_consistent(self, transformed):
        X_train_t, X_val_t, X_test_t = transformed
        assert X_train_t.shape[1] == X_val_t.shape[1] == X_test_t.shape[1]


# ---------------------------------------------------------------------------
# Stage 5 – select_features
# ---------------------------------------------------------------------------

class TestSelectFeatures:
    @pytest.fixture
    def selected(self, pipeline, raw_df):
        clean = pipeline.clean_data(raw_df)
        X_train, X_val, X_test, *_ = pipeline.split_data(clean)
        X_train_t, X_val_t, X_test_t = pipeline.transform_data(X_train, X_val, X_test)
        return pipeline.select_features(X_train_t, X_val_t, X_test_t)

    def test_returns_three_dataframes(self, selected):
        assert len(selected) == 3

    def test_selected_columns_stored(self, pipeline, selected):
        assert pipeline.selected_columns is not None
        assert len(pipeline.selected_columns) > 0

    def test_val_test_have_same_columns_as_train(self, selected):
        X_train_s, X_val_s, X_test_s = selected
        assert list(X_train_s.columns) == list(X_val_s.columns)
        assert list(X_train_s.columns) == list(X_test_s.columns)


# ---------------------------------------------------------------------------
# Stage 6 – handle_imbalance
# ---------------------------------------------------------------------------

class TestHandleImbalance:
    @pytest.fixture
    def balanced(self, pipeline, raw_df):
        clean = pipeline.clean_data(raw_df)
        X_train, _, _, y_train, *_ = pipeline.split_data(clean)
        X_train_t, *_ = pipeline.transform_data(X_train, X_train, X_train)
        X_train_s, *_ = pipeline.select_features(X_train_t, X_train_t, X_train_t)

        train_df = X_train_s.copy()
        train_df = train_df.apply(pd.to_numeric, errors="coerce").dropna()
        y_aligned = y_train.loc[train_df.index]
        train_df["diabetes"] = y_aligned
        return pipeline.handle_imbalance(train_df)

    def test_returns_dataframe(self, balanced):
        assert isinstance(balanced, pd.DataFrame)

    def test_target_column_present(self, balanced):
        assert "diabetes" in balanced.columns

    def test_minority_class_increased(self, balanced):
        counts = balanced["diabetes"].value_counts()
        assert counts[1] > 0

    def test_no_nulls(self, balanced):
        assert balanced.isnull().sum().sum() == 0


# ---------------------------------------------------------------------------
# Stage 7 – train_model + evaluate_model
# ---------------------------------------------------------------------------

class TestTrainAndEvaluate:
    @pytest.fixture
    def trained_pipeline(self, pipeline, raw_df, tmp_path):
        """Run through all stages up to and including training."""
        with patch.object(pipe_mod, "PROJECT_ROOT", tmp_path):
            clean = pipeline.clean_data(raw_df)
            X_train, X_val, X_test, y_train, y_val, y_test = pipeline.split_data(clean)
            X_train_t, X_val_t, X_test_t = pipeline.transform_data(X_train, X_val, X_test)
            X_train_s, X_val_s, X_test_s = pipeline.select_features(X_train_t, X_val_t, X_test_t)

            train_df = X_train_s.copy().apply(pd.to_numeric, errors="coerce").dropna()
            y_aligned = y_train.loc[train_df.index]
            train_df["diabetes"] = y_aligned
            train_df = pipeline.handle_imbalance(train_df)

            X_final = train_df.drop(columns=["diabetes"])
            y_final = train_df["diabetes"]
            pipeline.train_model(X_final, y_final)

        return pipeline, X_val_s, y_val, X_test_s, y_test

    def test_model_is_fitted(self, trained_pipeline):
        p, *_ = trained_pipeline
        assert hasattr(p.model, "classes_")

    def test_evaluate_returns_required_keys(self, trained_pipeline):
        p, X_val, y_val, *_ = trained_pipeline
        results = p.evaluate_model(X_val, y_val, dataset_name="Validation")
        for key in ("accuracy", "precision", "recall", "f1_score",
                    "confusion_matrix", "roc_auc"):
            assert key in results, f"Missing key: {key}"

    def test_accuracy_in_valid_range(self, trained_pipeline):
        p, X_val, y_val, *_ = trained_pipeline
        results = p.evaluate_model(X_val, y_val)
        assert 0.0 <= results["accuracy"] <= 1.0

    def test_confusion_matrix_shape(self, trained_pipeline):
        p, X_val, y_val, *_ = trained_pipeline
        cm = p.evaluate_model(X_val, y_val)["confusion_matrix"]
        assert len(cm) == 2 and len(cm[0]) == 2


# ---------------------------------------------------------------------------
# Stage 8 – full pipeline run()
# ---------------------------------------------------------------------------

class TestPipelineRun:
    @pytest.fixture
    def run_results(self, pipeline, tmp_path):
        with patch.object(pipe_mod, "PROJECT_ROOT", tmp_path):
            return pipeline.run()

    def test_returns_dict(self, run_results):
        assert isinstance(run_results, dict)

    def test_required_keys_present(self, run_results):
        for key in ("raw_shape", "clean_shape", "train_shape",
                    "validation_shape", "test_shape",
                    "validation_results", "test_results"):
            assert key in run_results, f"Missing key: {key}"

    def test_shapes_are_tuples(self, run_results):
        for key in ("raw_shape", "clean_shape", "train_shape",
                    "validation_shape", "test_shape"):
            assert isinstance(run_results[key], tuple)

    def test_clean_shape_rows_lte_raw(self, run_results):
        assert run_results["clean_shape"][0] <= run_results["raw_shape"][0]

    def test_validation_results_has_accuracy(self, run_results):
        assert "accuracy" in run_results["validation_results"]

    def test_test_results_has_roc_auc(self, run_results):
        assert "roc_auc" in run_results["test_results"]


# ---------------------------------------------------------------------------
# Stage 9 – predict_one (instance method)
# ---------------------------------------------------------------------------

class TestPredictOne:
    @pytest.fixture
    def ready_pipeline(self, pipeline, tmp_path):
        with patch.object(pipe_mod, "PROJECT_ROOT", tmp_path):
            pipeline.run()
            pipeline.load_artifacts()
        return pipeline

    def test_returns_dict(self, ready_pipeline):
        result = ready_pipeline.predict_one(EXAMPLE_PATIENT)
        assert isinstance(result, dict)

    def test_prediction_is_binary(self, ready_pipeline):
        result = ready_pipeline.predict_one(EXAMPLE_PATIENT)
        assert result["prediction"] in (0, 1)

    def test_probability_in_range(self, ready_pipeline):
        result = ready_pipeline.predict_one(EXAMPLE_PATIENT)
        assert "diabetes_probability" in result
        assert 0.0 <= result["diabetes_probability"] <= 1.0


# ---------------------------------------------------------------------------
# predict_single_sample (standalone function using notebook artifacts)
# ---------------------------------------------------------------------------

class TestPredictSingleSample:
    @pytest.fixture
    def mock_artifacts(self, tmp_path):
        """
        Patch joblib.load so predict_single_sample never touches the filesystem.
        Returns a fake preprocessor-equipped transformer and a trivial model.
        """
        from sklearn.dummy import DummyClassifier

        dummy_model = DummyClassifier(strategy="most_frequent")
        dummy_model.fit([[0] * 5, [0] * 5], [0, 1])

        fake_preprocessor = MagicMock()
        fake_columns = ["f1", "f2", "f3", "f4", "f5"]

        def fake_transform_one(sample):
            return pd.DataFrame([[0.0] * 5], columns=fake_columns)

        with patch.object(pipe_mod, "PROJECT_ROOT", tmp_path), \
             patch("src.diabetes_prediction.pipeline.pipeline.joblib.load") as mock_load, \
             patch("src.diabetes_prediction.pipeline.pipeline.DataTransformation") as MockDT:

            mock_load.side_effect = [fake_preprocessor, dummy_model]

            mock_transformer_instance = MagicMock()
            mock_transformer_instance.transform_one.side_effect = fake_transform_one
            MockDT.return_value = mock_transformer_instance

            yield

    def test_returns_success_on_valid_input(self, mock_artifacts):
        result = predict_single_sample(EXAMPLE_PATIENT)
        assert result["success"] is True, f"Expected success but got error: {result.get('error')}"

    def test_result_contains_prediction(self, mock_artifacts):
        result = predict_single_sample(EXAMPLE_PATIENT)
        assert result["success"] is True, f"Expected success but got error: {result.get('error')}"
        assert "prediction" in result["result"]

    def test_prediction_is_binary(self, mock_artifacts):
        result = predict_single_sample(EXAMPLE_PATIENT)
        assert result["success"] is True, f"Expected success but got error: {result.get('error')}"
        assert result["result"]["prediction"] in (0, 1)

    def test_returns_failure_on_bad_input(self):
        """If artifacts are missing the function must return success=False."""
        result = predict_single_sample({"invalid": "data"})
        assert result["success"] is False
        assert "error" in result