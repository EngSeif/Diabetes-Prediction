import pytest
import pandas as pd
from src.diabetes_prediction.cleaning.cleaning import DataCleaning

# ──────────────────────────────────────────────
#  Fixtures
# ──────────────────────────────────────────────


@pytest.fixture
def sample_df():
    return pd.DataFrame(
        {
            "gender": ["Male", "Female", "Other", "Male", "Female"],
            "age": [25, 30, 40, 8, 12],
            "smoking_history": ["never", "never", "never", "current", "No Info"],
        }
    )


# ──────────────────────────────────────────────
#  remove_invalid_gender
# ──────────────────────────────────────────────


def test_removes_other_gender(sample_df):
    """Rows with gender == 'Other' must be dropped."""
    result = DataCleaning(sample_df.copy()).remove_invalid_gender().df
    assert "Other" not in result["gender"].values


def test_keeps_valid_genders(sample_df):
    """Male and Female rows must be retained."""
    result = DataCleaning(sample_df.copy()).remove_invalid_gender().df
    assert set(result["gender"].unique()) == {"Male", "Female"}


def test_no_other_gender_returns_unchanged():
    """DataFrame with no 'Other' rows must remain the same size."""
    df = pd.DataFrame(
        {
            "gender": ["Male", "Female"],
            "age": [20, 30],
            "smoking_history": ["never", "never"],
        }
    )
    result = DataCleaning(df.copy()).remove_invalid_gender().df
    assert len(result) == len(df)


# ──────────────────────────────────────────────
#  remove_inaccurate_infants
# ──────────────────────────────────────────────


def test_removes_young_smokers(sample_df):
    """age < 10 with a smoking history must be dropped."""
    result = DataCleaning(sample_df.copy()).remove_inaccurate_infants().df
    infant_smokers = result[
        (result["age"] < 10)
        & (result["smoking_history"].isin(["not current", "current", "ever", "former"]))
    ]
    assert infant_smokers.empty


def test_keeps_young_non_smokers():
    """age < 10 with 'never' or 'No Info' smoking history is kept."""
    df = pd.DataFrame(
        {
            "gender": ["Male", "Female"],
            "age": [5, 7],
            "smoking_history": ["never", "No Info"],
        }
    )
    result = DataCleaning(df.copy()).remove_inaccurate_infants().df
    assert len(result) == 2


def test_keeps_adults_with_smoking_history():
    """Adults (age >= 10) with smoking history must NOT be removed."""
    df = pd.DataFrame(
        {
            "gender": ["Male", "Male"],
            "age": [10, 25],
            "smoking_history": ["current", "former"],
        }
    )
    result = DataCleaning(df.copy()).remove_inaccurate_infants().df
    assert len(result) == 2


# ──────────────────────────────────────────────
#  remove_duplicates
# ──────────────────────────────────────────────


def test_removes_exact_duplicates():
    """Exact duplicate rows must be reduced to one occurrence."""
    df = pd.DataFrame(
        {
            "gender": ["Male", "Male", "Female"],
            "age": [25, 25, 30],
            "smoking_history": ["never", "never", "never"],
        }
    )
    result = DataCleaning(df.copy()).remove_duplicates().df
    assert len(result) == 2


def test_keeps_first_occurrence():
    """First duplicate is retained, subsequent copies are dropped."""
    df = pd.DataFrame(
        {
            "gender": ["Male", "Male"],
            "age": [25, 25],
            "smoking_history": ["never", "never"],
        }
    )
    result = DataCleaning(df.copy()).remove_duplicates().df
    assert result.index.tolist() == [0]


def test_no_duplicates_unchanged(sample_df):
    """DataFrame with no duplicates must not lose any rows."""
    result = DataCleaning(sample_df.copy()).remove_duplicates().df
    assert len(result) == len(sample_df)


# ──────────────────────────────────────────────
#  run_cleaning  (integration of all steps)
# ──────────────────────────────────────────────


def test_run_cleaning_returns_dataframe(sample_df):
    """run_cleaning must return a pd.DataFrame."""
    result = DataCleaning(sample_df.copy()).run_cleaning()
    assert isinstance(result, pd.DataFrame)


def test_run_cleaning_applies_all_steps():
    """Full pipeline must remove Other gender, infant smokers, and duplicates."""
    df = pd.DataFrame(
        {
            "gender": ["Male", "Other", "Female", "Male"],
            "age": [25, 40, 8, 25],
            "smoking_history": ["never", "never", "current", "never"],
        }
    )
    result = DataCleaning(df).run_cleaning()
    assert "Other" not in result["gender"].values  # gender cleaned
    assert len(result) == 1  # infant + duplicate removed


def test_run_cleaning_empty_df():
    """Empty DataFrame must pass through without raising an error."""
    df = pd.DataFrame(columns=["gender", "age", "smoking_history"])
    result = DataCleaning(df).run_cleaning()
    assert result.empty
