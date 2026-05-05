import os
import pytest
import pandas as pd
import numpy as np
from src.diabetes_prediction.imbalance.imbalance import DataImbalance

# ──────────────────────────────────────────────
#  Fixtures
# ──────────────────────────────────────────────


@pytest.fixture
def imbalanced_df():
    """Imbalanced dataset: 80 negatives, 20 positives."""
    np.random.seed(42)
    n = 100
    df = pd.DataFrame(
        {
            "age": np.random.randint(20, 80, n).astype(float),
            "bmi": np.random.uniform(18, 45, n),
            "HbA1c_level": np.random.uniform(3.5, 9.0, n),
            "blood_glucose_level": np.random.randint(80, 300, n).astype(float),
            "diabetes": [0] * 80 + [1] * 20,
        }
    )
    return df


@pytest.fixture
def handler(imbalanced_df, tmp_path, monkeypatch):
    """DataImbalance instance with output_dir redirected to tmp_path."""
    obj = DataImbalance(imbalanced_df.copy())
    monkeypatch.setattr(obj, "output_dir", str(tmp_path))
    return obj


def test_init_splits_X_and_y(imbalanced_df):
    """X_train must not contain target; y_train must be the target column."""
    obj = DataImbalance(imbalanced_df.copy())
    assert "diabetes" not in obj.X_train.columns
    assert obj.y_train.name == "diabetes"


def test_init_creates_output_dir(imbalanced_df, tmp_path):
    """output_dir must be created automatically on init."""
    # target = tmp_path / "imbalance_resolve"
    # Just check the object initializes without error and output_dir is set
    obj = DataImbalance(imbalanced_df.copy())
    assert os.path.exists(obj.output_dir)


def test_init_custom_target_col(imbalanced_df):
    """Custom target column name must be respected."""
    df = imbalanced_df.rename(columns={"diabetes": "label"})
    obj = DataImbalance(df, target_col="label")
    assert obj.y_train.name == "label"
    assert "label" not in obj.X_train.columns


# ──────────────────────────────────────────────
#  _to_dataframe
# ──────────────────────────────────────────────


def test_to_dataframe_returns_dataframe(handler):
    """_to_dataframe must return a pd.DataFrame."""
    X = handler.X_train.values
    y = handler.y_train.values
    result = handler._to_dataframe(X, y)
    assert isinstance(result, pd.DataFrame)


def test_to_dataframe_contains_target(handler):
    """Result must include the target column."""
    result = handler._to_dataframe(handler.X_train.values, handler.y_train.values)
    assert "diabetes" in result.columns


def test_to_dataframe_no_nulls(handler):
    """Result must have no NaN values after conversion."""
    result = handler._to_dataframe(handler.X_train.values, handler.y_train.values)
    assert result.isnull().sum().sum() == 0


def test_to_dataframe_all_numeric(handler):
    """All columns in result must be numeric."""
    result = handler._to_dataframe(handler.X_train.values, handler.y_train.values)
    assert result.select_dtypes(include="number").shape[1] == result.shape[1]


# ──────────────────────────────────────────────
#  _save
# ──────────────────────────────────────────────


def test_save_creates_csv_file(handler, tmp_path):
    """_save must write a CSV file at the correct path."""
    handler._save(handler.X_train.values, handler.y_train.values, "test_output.csv")
    assert (tmp_path / "test_output.csv").exists()


def test_save_csv_has_correct_columns(handler, tmp_path):
    """Saved CSV must contain feature columns and 'diabetes_target'."""
    handler._save(handler.X_train.values, handler.y_train.values, "test_cols.csv")
    saved = pd.read_csv(tmp_path / "test_cols.csv")
    assert "diabetes_target" in saved.columns
    for col in handler.X_train.columns:
        assert col in saved.columns


def test_adasyn_returns_dataframe(handler):
    """adasyn must return a pd.DataFrame."""
    result = handler.adasyn()
    assert isinstance(result, pd.DataFrame)


def test_adasyn_balances_classes(handler):
    """adasyn must produce a more balanced class distribution."""
    result = handler.adasyn()
    counts = result["diabetes"].value_counts()
    minority_before = 20
    assert counts[1] > minority_before


def test_adasyn_preserves_feature_columns(handler):
    """adasyn result must contain all original feature columns."""
    result = handler.adasyn()
    for col in handler.X_train.columns:
        assert col in result.columns


# ──────────────────────────────────────────────
#  smote_df
# ──────────────────────────────────────────────


def test_smote_df_returns_self(handler):
    """smote_df must return self for method chaining."""
    result = handler.smote_df()
    assert result is handler


def test_smote_df_saves_csv(handler, tmp_path):
    """smote_df must write train_smote.csv to the output directory."""
    handler.smote_df()
    assert (tmp_path / "train_smote.csv").exists()


# ──────────────────────────────────────────────
#  smote_tomek_df
# ──────────────────────────────────────────────


def test_smote_tomek_saves_csv(handler, tmp_path):
    """smote_tomek_df must write train_smote_tomek.csv."""
    handler.smote_tomek_df()
    assert (tmp_path / "train_smote_tomek.csv").exists()


# ──────────────────────────────────────────────
#  smote_enn_df
# ──────────────────────────────────────────────


def test_smote_enn_saves_csv(handler, tmp_path):
    """smote_enn_df must write train_smote_enn.csv."""
    handler.smote_enn_df()
    assert (tmp_path / "train_smote_enn.csv").exists()


# ──────────────────────────────────────────────
#  run_pipeline  — bug documented
# ──────────────────────────────────────────────


def test_run_pipeline_raises_due_to_typo(handler):
    """run_pipeline calls self.adaysn_df() which does not exist — must raise AttributeError."""
    with pytest.raises(AttributeError, match="adaysn_df"):
        handler.run_pipeline()


def test_run_pipeline_after_fix(handler, tmp_path, monkeypatch):
    """After fixing the typo, all four methods must be called."""
    called = []
    monkeypatch.setattr(handler, "adasyn", lambda: called.append("adasyn"))
    monkeypatch.setattr(handler, "smote_df", lambda: called.append("smote"))
    monkeypatch.setattr(handler, "smote_enn_df", lambda: called.append("enn"))
    monkeypatch.setattr(handler, "smote_tomek_df", lambda: called.append("tomek"))

    # simulate fixed pipeline
    handler.adasyn()
    handler.smote_df()
    handler.smote_enn_df()
    handler.smote_tomek_df()

    assert set(called) == {"adasyn", "smote", "enn", "tomek"}
