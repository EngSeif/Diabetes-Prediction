import pytest
import pandas as pd
import numpy as np
import great_expectations as gx
from src.diabetes_prediction.validation.validation import DataValidator


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def clean_df():
    """Valid DataFrame that should pass all validation checks."""
    return pd.DataFrame({
        "gender":              ["Male", "Female", "Male", "Female"],
        "age":                 [25.0, 45.0, 60.0, 30.0],
        "bmi":                 [22.5, 30.1, 27.8, 24.0],
        "HbA1c_level":         [5.5, 6.2, 5.8, 5.1],
        "blood_glucose_level": [120.0, 180.0, 150.0, 110.0],
        "smoking_history":     ["never", "former", "current", "No Info"],
        "hypertension":        [0, 1, 0, 0],
        "heart_disease":       [0, 0, 1, 0],
        "diabetes":            [0, 0, 0, 0],
    })


@pytest.fixture
def validator(clean_df):
    return DataValidator(clean_df.copy())


@pytest.fixture
def suite():
    """Fresh ephemeral GX suite for unit testing expectations."""
    context = gx.get_context(mode="ephemeral")
    return context.suites.add(
        gx.ExpectationSuite(name="test_suite")
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _exp_type(e) -> str:
    """Return a lower-cased class name for the expectation object.

    GX v1.x expectation objects do not expose a `.type` attribute at the
    instance level; the class name is the reliable runtime identifier.

    Class names are CamelCase, so lowercasing removes word boundaries:
      ExpectColumnValuesToNotBeNull  →  expectcolumnvaluestonotbenull
      ExpectColumnValuesToBeBetween  →  expectcolumnvaluestobebetween
      ExpectColumnValuesToBeInSet    →  expectcolumnvaluestobeinset

    Search for the corresponding boundary-free substrings in tests.
    """
    return type(e).__name__.lower()


def _column(e) -> str:
    """Return the column an expectation targets (GX v1.x uses e.column)."""
    return e.column


# ---------------------------------------------------------------------------
# _add_completeness_expectations
# ---------------------------------------------------------------------------

def test_completeness_adds_one_expectation_per_column(validator, suite, clean_df):
    """Must add exactly one null-check expectation per column."""
    validator._add_completeness_expectations(suite, clean_df)
    null_checks = [
        e for e in suite.expectations
        if "tonotbenull" in _exp_type(e)
    ]
    assert len(null_checks) == len(clean_df.columns)


def test_completeness_covers_all_columns(validator, suite, clean_df):
    """Every column must have a corresponding not-null expectation."""
    validator._add_completeness_expectations(suite, clean_df)
    covered = {_column(e) for e in suite.expectations if "tonotbenull" in _exp_type(e)}
    assert covered == set(clean_df.columns)


# ---------------------------------------------------------------------------
# _add_accuracy_expectations
# ---------------------------------------------------------------------------

def test_accuracy_adds_age_range(validator, suite):
    """Must add a between expectation for age with range (0, 120)."""
    validator._add_accuracy_expectations(suite)
    age_exp = [
        e for e in suite.expectations
        if "tobebetween" in _exp_type(e) and _column(e) == "age"
    ]
    assert len(age_exp) == 1
    assert age_exp[0].min_value == 0
    assert age_exp[0].max_value == 120


def test_accuracy_adds_bmi_range(validator, suite):
    """Must add a between expectation for bmi with range (10, 70)."""
    validator._add_accuracy_expectations(suite)
    bmi_exp = [
        e for e in suite.expectations
        if "tobebetween" in _exp_type(e) and _column(e) == "bmi"
    ]
    assert len(bmi_exp) == 1
    assert bmi_exp[0].min_value == 10
    assert bmi_exp[0].max_value == 70


def test_accuracy_adds_gender_set(validator, suite):
    """Must add an in-set expectation for gender with Male and Female."""
    validator._add_accuracy_expectations(suite)
    gender_exp = [
        e for e in suite.expectations
        if "tobeinset" in _exp_type(e) and _column(e) == "gender"
    ]
    assert len(gender_exp) == 1
    assert set(gender_exp[0].value_set) == {"Male", "Female"}


def test_accuracy_adds_smoking_history_set(validator, suite):
    """Must add an in-set expectation for smoking_history."""
    validator._add_accuracy_expectations(suite)
    smoke_exp = [
        e for e in suite.expectations
        if "tobeinset" in _exp_type(e) and _column(e) == "smoking_history"
    ]
    assert len(smoke_exp) == 1
    assert "never" in smoke_exp[0].value_set
    assert "former" in smoke_exp[0].value_set


def test_accuracy_adds_binary_checks_for_all_three(validator, suite):
    """Must add binary in-set checks for hypertension, heart_disease, diabetes."""
    validator._add_accuracy_expectations(suite)
    binary_cols = {
        _column(e) for e in suite.expectations
        if "tobeinset" in _exp_type(e)
        and set(e.value_set) == {0, 1}
    }
    assert {"hypertension", "heart_disease", "diabetes"}.issubset(binary_cols)


# ---------------------------------------------------------------------------
# _add_consistency_expectations
# ---------------------------------------------------------------------------

def test_consistency_creates_high_hba1c_flag(validator, suite, clean_df):
    """Must create _high_hba1c_no_diabetes flag column."""
    df_copy = clean_df.copy()
    validator._add_consistency_expectations(suite, df_copy)
    assert "_high_hba1c_no_diabetes" in df_copy.columns


def test_consistency_high_hba1c_flag_logic(validator, suite, clean_df):
    """_high_hba1c_no_diabetes must be 1 only when HbA1c >= 7.5 and diabetes == 0."""
    df_copy = clean_df.copy()
    df_copy.loc[0, "HbA1c_level"] = 8.0
    df_copy.loc[0, "diabetes"] = 0
    validator._add_consistency_expectations(suite, df_copy)
    assert df_copy.loc[0, "_high_hba1c_no_diabetes"] == 1


def test_consistency_creates_high_glucose_flag(validator, suite, clean_df):
    """Must create _high_glucose_no_diabetes flag column."""
    df_copy = clean_df.copy()
    validator._add_consistency_expectations(suite, df_copy)
    assert "_high_glucose_no_diabetes" in df_copy.columns


def test_consistency_high_glucose_flag_logic(validator, suite, clean_df):
    """_high_glucose_no_diabetes must be 1 when glucose > 300 and diabetes == 0."""
    df_copy = clean_df.copy()
    df_copy.loc[0, "blood_glucose_level"] = 310.0
    df_copy.loc[0, "diabetes"] = 0
    validator._add_consistency_expectations(suite, df_copy)
    assert df_copy.loc[0, "_high_glucose_no_diabetes"] == 1


def test_consistency_creates_young_with_conditions_flag(validator, suite, clean_df):
    """Must create _young_with_conditions flag column."""
    df_copy = clean_df.copy()
    validator._add_consistency_expectations(suite, df_copy)
    assert "_young_with_conditions" in df_copy.columns


def test_consistency_young_flag_logic(validator, suite, clean_df):
    """_young_with_conditions must be 1 when age < 10 and hypertension or heart_disease == 1."""
    df_copy = clean_df.copy()
    df_copy.loc[0, "age"] = 5.0
    df_copy.loc[0, "hypertension"] = 1
    validator._add_consistency_expectations(suite, df_copy)
    assert df_copy.loc[0, "_young_with_conditions"] == 1


# ---------------------------------------------------------------------------
# _add_uniqueness_expectations
# ---------------------------------------------------------------------------

def test_uniqueness_creates_is_duplicate_flag(validator, suite, clean_df):
    """Must create _is_duplicate flag column."""
    df_copy = clean_df.copy()
    validator._add_uniqueness_expectations(suite, df_copy)
    assert "_is_duplicate" in df_copy.columns


def test_uniqueness_flags_duplicate_rows(validator, suite, clean_df):
    """_is_duplicate must be 1 for the second occurrence of a duplicate row."""
    df_copy = pd.concat([clean_df, clean_df.iloc[[0]]], ignore_index=True)
    v = DataValidator(df_copy.copy())
    df_work = df_copy.copy()
    v._add_uniqueness_expectations(suite, df_work)
    assert df_work["_is_duplicate"].iloc[-1] == 1


def test_uniqueness_no_duplicates_all_zero(validator, suite, clean_df):
    """_is_duplicate must be 0 for all rows when no duplicates exist."""
    df_copy = clean_df.copy()
    validator._add_uniqueness_expectations(suite, df_copy)
    assert df_copy["_is_duplicate"].sum() == 0


# ---------------------------------------------------------------------------
# _add_outlier_expectations
# ---------------------------------------------------------------------------

def test_outlier_creates_flag_columns(validator, suite, clean_df):
    """Must create an IQR flag column for each numeric column."""
    df_copy = clean_df.copy()
    validator._add_outlier_expectations(suite, df_copy)
    for col in ["age", "bmi", "HbA1c_level", "blood_glucose_level"]:
        assert f"_iqr_outlier_{col}" in df_copy.columns


def test_outlier_skips_missing_column(validator, suite, clean_df):
    """Must not raise an error when a numeric column is missing."""
    df_copy = clean_df.drop(columns=["bmi"])
    try:
        validator._add_outlier_expectations(suite, df_copy)
    except Exception as e:
        pytest.fail(f"Raised unexpected exception: {e}")


def test_outlier_flags_extreme_value(validator, suite):
    """Must flag an extreme outlier value as 1."""
    df = pd.DataFrame({
        "age":                 [25.0] * 10 + [999.0],
        "bmi":                 [22.5] * 11,
        "HbA1c_level":         [5.5] * 11,
        "blood_glucose_level": [120.0] * 11,
    })
    v = DataValidator(df.copy())
    df_copy = df.copy()
    context = gx.get_context(mode="ephemeral")
    s = context.suites.add(gx.ExpectationSuite(name="outlier_suite"))
    v._add_outlier_expectations(s, df_copy)
    assert df_copy["_iqr_outlier_age"].iloc[-1] == 1

