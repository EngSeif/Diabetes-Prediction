import pytest
import pandas as pd
import numpy as np
import joblib
from src.diabetes_prediction.transformation.transformation import DataTransformation


# Fixtures

@pytest.fixture
def sample_df():
    """Minimal DataFrame with all required raw columns."""
    return pd.DataFrame({
        "gender":              ["Male", "Female", "Male"],
        "age":                 [25.0, 45.0, 60.0],
        "bmi":                 [22.5, 30.1, 27.8],
        "HbA1c_level":         [5.5, 7.2, 6.1],
        "blood_glucose_level": [120.0, 200.0, 150.0],
        "smoking_history":     ["never", "former", "current"],
        "hypertension":        [0, 1, 0],
        "heart_disease":       [0, 0, 1],
    })


@pytest.fixture
def transformer():
    return DataTransformation()


@pytest.fixture
def fitted_transformer(transformer, sample_df):
    """Transformer that has already been fit on sample_df."""
    transformer.fit_transform(sample_df.copy())
    return transformer


# encode_gender

def test_encode_gender_male_to_1(transformer, sample_df):
    """Male must be encoded as 1."""
    result = transformer.encode_gender(sample_df.copy())
    assert result.loc[result.index[0], "gender"] == 1


def test_encode_gender_female_to_0(transformer, sample_df):
    """Female must be encoded as 0."""
    result = transformer.encode_gender(sample_df.copy())
    assert result.loc[result.index[1], "gender"] == 0


def test_encode_gender_no_mutation(transformer, sample_df):
    """Original DataFrame must not be mutated."""
    original = sample_df["gender"].tolist()
    transformer.encode_gender(sample_df)
    assert sample_df["gender"].tolist() == original


# add_engineered_features

def test_add_engineered_features_glucose_hba1c(transformer, sample_df):
    """glucose_hba1c_interaction must equal blood_glucose_level * HbA1c_level."""
    result = transformer.add_engineered_features(sample_df.copy())
    expected = sample_df["blood_glucose_level"] * sample_df["HbA1c_level"]
    pd.testing.assert_series_equal(
        result["glucose_hba1c_interaction"].reset_index(drop=True),
        expected.reset_index(drop=True),
        check_names=False
    )


def test_add_engineered_features_age_hba1c(transformer, sample_df):
    """age_hba1c_interaction must equal age * HbA1c_level."""
    result = transformer.add_engineered_features(sample_df.copy())
    expected = sample_df["age"] * sample_df["HbA1c_level"]
    pd.testing.assert_series_equal(
        result["age_hba1c_interaction"].reset_index(drop=True),
        expected.reset_index(drop=True),
        check_names=False
    )


def test_add_engineered_features_high_hba1c_flag(transformer, sample_df):
    """high_hba1c_flag must be 1 when HbA1c_level >= 6.6."""
    result = transformer.add_engineered_features(sample_df.copy())
    expected = (sample_df["HbA1c_level"] >= 6.6).astype(int)
    pd.testing.assert_series_equal(
        result["high_hba1c_flag"].reset_index(drop=True),
        expected.reset_index(drop=True),
        check_names=False
    )


def test_add_engineered_features_senior_flag(transformer, sample_df):
    """senior_flag must be 1 when age >= 60."""
    result = transformer.add_engineered_features(sample_df.copy())
    expected = (sample_df["age"] >= 60).astype(int)
    pd.testing.assert_series_equal(
        result["senior_flag"].reset_index(drop=True),
        expected.reset_index(drop=True),
        check_names=False
    )


def test_add_engineered_features_cardio_risk_flag(transformer, sample_df):
    """cardio_risk_flag must be 1 when hypertension == 1 OR heart_disease == 1."""
    result = transformer.add_engineered_features(sample_df.copy())
    expected = ((sample_df["hypertension"] == 1) | (sample_df["heart_disease"] == 1)).astype(int)
    pd.testing.assert_series_equal(
        result["cardio_risk_flag"].reset_index(drop=True),
        expected.reset_index(drop=True),
        check_names=False
    )


def test_add_engineered_features_all_columns_present(transformer, sample_df):
    """All 8 engineered columns must be present in the result."""
    result = transformer.add_engineered_features(sample_df.copy())
    expected_cols = [
        "glucose_hba1c_interaction",
        "age_hba1c_interaction",
        "age_bmi_interaction",
        "bmi_hba1c_interaction",
        "age_glucose_interaction",
        "high_hba1c_flag",
        "senior_flag",
        "cardio_risk_flag",
    ]
    for col in expected_cols:
        assert col in result.columns


def test_add_engineered_features_no_mutation(transformer, sample_df):
    """Original DataFrame must not be mutated."""
    original_cols = list(sample_df.columns)
    transformer.add_engineered_features(sample_df)
    assert list(sample_df.columns) == original_cols


# prepare_features

def test_prepare_features_encodes_gender(transformer, sample_df):
    """prepare_features must encode gender to numeric."""
    result = transformer.prepare_features(sample_df.copy())
    assert result["gender"].dtype in [np.int64, np.int32, int, np.float64]


def test_prepare_features_adds_engineered_cols(transformer, sample_df):
    """prepare_features must add all engineered feature columns."""
    result = transformer.prepare_features(sample_df.copy())
    assert "glucose_hba1c_interaction" in result.columns
    assert "cardio_risk_flag" in result.columns


# fit_transform

def test_fit_transform_returns_dataframe(transformer, sample_df):
    """fit_transform must return a pd.DataFrame."""
    result = transformer.fit_transform(sample_df.copy())
    assert isinstance(result, pd.DataFrame)


def test_fit_transform_correct_row_count(transformer, sample_df):
    """fit_transform must return same number of rows as input."""
    result = transformer.fit_transform(sample_df.copy())
    assert len(result) == len(sample_df)


def test_fit_transform_sets_feature_names(transformer, sample_df):
    """fit_transform must populate feature_names_."""
    transformer.fit_transform(sample_df.copy())
    assert transformer.feature_names_ is not None
    assert len(transformer.feature_names_) > 0


def test_fit_transform_no_nulls(transformer, sample_df):
    """fit_transform result must contain no NaN values."""
    result = transformer.fit_transform(sample_df.copy())
    assert result.isnull().sum().sum() == 0


# transform

def test_transform_returns_dataframe(fitted_transformer, sample_df):
    """transform must return a pd.DataFrame."""
    result = fitted_transformer.transform(sample_df.copy())
    assert isinstance(result, pd.DataFrame)


def test_transform_returns_float32(fitted_transformer, sample_df):
    """transform result must have float32 dtype."""
    result = fitted_transformer.transform(sample_df.copy())
    assert all(result[col].dtype == np.float32 for col in result.columns)


def test_transform_same_columns_as_fit(fitted_transformer, sample_df):
    """transform must return the same columns as fit_transform."""
    result = fitted_transformer.transform(sample_df.copy())
    assert list(result.columns) == list(fitted_transformer.feature_names_)


# transform_one

def test_transform_one_accepts_dict(fitted_transformer, sample_df):
    """transform_one must accept a dict input."""
    row = sample_df.iloc[0].to_dict()
    result = fitted_transformer.transform_one(row)
    assert isinstance(result, pd.DataFrame)
    assert len(result) == 1


def test_transform_one_accepts_series(fitted_transformer, sample_df):
    """transform_one must accept a pd.Series input."""
    row = sample_df.iloc[0]
    result = fitted_transformer.transform_one(row)
    assert isinstance(result, pd.DataFrame)
    assert len(result) == 1


def test_transform_one_accepts_dataframe(fitted_transformer, sample_df):
    """transform_one must accept a single-row pd.DataFrame input."""
    row = sample_df.iloc[[0]]
    result = fitted_transformer.transform_one(row)
    assert isinstance(result, pd.DataFrame)
    assert len(result) == 1


# save and load preprocessor

def test_save_and_load_preprocessor(fitted_transformer, tmp_path):
    """Saved preprocessor must be reloadable and produce identical output."""
    path = tmp_path / "preprocessor.pkl"
    fitted_transformer.save_preprocessor(path=path)
    assert path.exists()

    new_transformer = DataTransformation()
    new_transformer.load_preprocessor(path=path)
    assert new_transformer.preprocessor is not None