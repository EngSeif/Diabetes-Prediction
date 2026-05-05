import pytest
import pandas as pd
from src.diabetes_prediction.selection.selection import FeatureSelection

# Fixtures


@pytest.fixture
def sample_df():
    """DataFrame with all expected columns including ones to be dropped."""
    return pd.DataFrame(
        {
            "age": [25, 40, 55],
            "bmi": [22.5, 30.1, 27.8],
            "HbA1c_level": [5.5, 7.2, 6.1],
            "blood_glucose_level": [120, 200, 150],
            "age_bmi_interaction": [562.5, 1204.0, 1529.0],
            "age_hba1c_interaction": [137.5, 288.0, 335.5],
            "hypertension": [0, 1, 0],
            "heart_disease": [0, 0, 1],
            "diabetes": [0, 1, 0],
        }
    )


# drop_features


def test_drop_features_returns_dataframe(sample_df):
    """drop_features must return a pd.DataFrame."""
    result = FeatureSelection(sample_df.copy()).drop_features()
    assert isinstance(result, pd.DataFrame)


def test_drop_features_removes_age(sample_df):
    """'age' must be dropped."""
    result = FeatureSelection(sample_df.copy()).drop_features()
    assert "age" not in result.columns


def test_drop_features_removes_blood_glucose_level(sample_df):
    """'blood_glucose_level' must be dropped."""
    result = FeatureSelection(sample_df.copy()).drop_features()
    assert "blood_glucose_level" not in result.columns


def test_drop_features_removes_age_bmi_interaction(sample_df):
    """'age_bmi_interaction' must be dropped."""
    result = FeatureSelection(sample_df.copy()).drop_features()
    assert "age_bmi_interaction" not in result.columns


def test_drop_features_removes_age_hba1c_interaction(sample_df):
    """'age_hba1c_interaction' must be dropped."""
    result = FeatureSelection(sample_df.copy()).drop_features()
    assert "age_hba1c_interaction" not in result.columns


def test_drop_features_removes_hypertension(sample_df):
    """'hypertension' must be dropped."""
    result = FeatureSelection(sample_df.copy()).drop_features()
    assert "hypertension" not in result.columns


def test_drop_features_keeps_remaining_columns(sample_df):
    """Columns not in the drop list must be retained."""
    result = FeatureSelection(sample_df.copy()).drop_features()
    for col in ["bmi", "HbA1c_level", "heart_disease", "diabetes"]:
        assert col in result.columns


def test_drop_features_correct_column_count(sample_df):
    """Result must have exactly 4 columns after dropping 5."""
    result = FeatureSelection(sample_df.copy()).drop_features()
    assert result.shape[1] == 4


def test_drop_features_does_not_modify_original(sample_df):
    """Original DataFrame must not be mutated."""
    original_cols = list(sample_df.columns)
    FeatureSelection(sample_df.copy()).drop_features()
    assert list(sample_df.columns) == original_cols


def test_drop_features_raises_if_column_missing():
    """drop_features must raise KeyError if a column to drop is missing."""
    df = pd.DataFrame(
        {
            "bmi": [22.5],
            "HbA1c_level": [5.5],
            "diabetes": [0],
        }
    )
    with pytest.raises(KeyError):
        FeatureSelection(df).drop_features()
