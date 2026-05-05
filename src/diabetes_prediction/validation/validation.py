import pandas as pd
import great_expectations as gx


class DataValidator:

    def __init__(self, df: pd.DataFrame):
        self.df = df

    def _add_completeness_expectations(self, suite, df):
        for col in df.columns:
            suite.add_expectation(
                gx.expectations.ExpectColumnValuesToNotBeNull(column=col)
            )

    def _add_accuracy_expectations(self, suite):
        numeric_ranges = {
            "age": (0, 120),
            "bmi": (10, 70),
            "HbA1c_level": (3, 15),
            "blood_glucose_level": (50, 400),
        }
        for col, (min_val, max_val) in numeric_ranges.items():
            suite.add_expectation(
                gx.expectations.ExpectColumnValuesToBeBetween(
                    column=col, min_value=min_val, max_value=max_val
                )
            )

        suite.add_expectation(
            gx.expectations.ExpectColumnValuesToBeInSet(
                column="gender",
                value_set=["Male", "Female"],
            )
        )
        suite.add_expectation(
            gx.expectations.ExpectColumnValuesToBeInSet(
                column="smoking_history",
                value_set=[
                    "never",
                    "former",
                    "current",
                    "ever",
                    "not current",
                    "No Info",
                ],
            )
        )

        for binary_col in ["hypertension", "heart_disease", "diabetes"]:
            suite.add_expectation(
                gx.expectations.ExpectColumnValuesToBeInSet(
                    column=binary_col,
                    value_set=[0, 1],
                )
            )

    def _add_consistency_expectations(self, suite, df_consistency):
        df_consistency["_high_hba1c_no_diabetes"] = (
            (df_consistency["HbA1c_level"] >= 7.5) & (df_consistency["diabetes"] == 0)
        ).astype(int)
        suite.add_expectation(
            gx.expectations.ExpectColumnValuesToBeInSet(
                column="_high_hba1c_no_diabetes",
                value_set=[0],
                notes="High HbA1c (>=7.5%) must not appear in non-diabetic records.",
            )
        )

        df_consistency["_high_glucose_no_diabetes"] = (
            (df_consistency["blood_glucose_level"] > 300)
            & (df_consistency["diabetes"] == 0)
        ).astype(int)
        suite.add_expectation(
            gx.expectations.ExpectColumnValuesToBeInSet(
                column="_high_glucose_no_diabetes",
                value_set=[0],
                notes="Blood glucose > 300 must not appear in non-diabetic records.",
            )
        )

        df_consistency["_young_with_conditions"] = (
            (df_consistency["age"] < 10)
            & (
                (df_consistency["hypertension"] == 1)
                | (df_consistency["heart_disease"] == 1)
            )
        ).astype(int)
        suite.add_expectation(
            gx.expectations.ExpectColumnValuesToBeInSet(
                column="_young_with_conditions",
                value_set=[0],
                notes="Patients under 10 should not have hypertension or heart disease.",
            )
        )

    def _add_uniqueness_expectations(self, suite, df_consistency):
        df_consistency["_is_duplicate"] = (
            df_consistency[self.df.columns.tolist()]
            .duplicated(keep="first")
            .astype(int)
        )
        suite.add_expectation(
            gx.expectations.ExpectColumnValuesToBeInSet(
                column="_is_duplicate",
                value_set=[0],
                notes="Each row should be unique across all columns.",
            )
        )

    def _add_outlier_expectations(self, suite, df):
        cols = ["age", "bmi", "HbA1c_level", "blood_glucose_level"]

        for col in cols:
            if col not in df.columns:
                continue

            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR

            flag_col = f"_iqr_outlier_{col}"
            df[flag_col] = ((df[col] < lower) | (df[col] > upper)).astype(int)

            suite.add_expectation(
                gx.expectations.ExpectColumnValuesToBeInSet(
                    column=flag_col,
                    value_set=[0],
                    notes=(
                        f"Column '{col}' should have no IQR outliers. "
                        f"Valid range: [{lower:.2f}, {upper:.2f}] "
                        f"(Q1={Q1:.2f}, Q3={Q3:.2f}, IQR={IQR:.2f})."
                    ),
                )
            )

    def run_validation(self):
        df = self.df
        context = gx.get_context(mode="ephemeral")

        data_source = context.data_sources.add_pandas(name="pandas")
        data_asset = data_source.add_dataframe_asset(name="dataset")
        batch_def = data_asset.add_batch_definition_whole_dataframe("batch")

        suite = context.suites.add(gx.ExpectationSuite(name="dataset_validation_suite"))

        df_consistency = df.copy()

        self._add_completeness_expectations(suite, df)
        self._add_accuracy_expectations(suite)
        self._add_consistency_expectations(suite, df_consistency)
        self._add_uniqueness_expectations(suite, df_consistency)
        self._add_outlier_expectations(suite, df_consistency)

        # batch = batch_def.get_batch(batch_parameters={"dataframe": df_consistency})

        validation_def = context.validation_definitions.add(
            gx.ValidationDefinition(
                name="dataset_validation",
                data=batch_def,
                suite=suite,
            )
        )

        results = validation_def.run(batch_parameters={"dataframe": df_consistency})
        self._print_report(results)
        return results

    def _print_report(self, results):
        success = results.success
        W = 58

        print("=" * W)
        print("    DATA VALIDATION REPORT  (Great Expectations v1.x)")
        print("=" * W)
        print(f"  Overall Result : {'PASSED' if success else 'FAILED'}")
        print("=" * W)

        # ── Column Data Types ──────────────────────────────────
        if df is not None:
            print(f"\n{'─' * W}")
            print("  COLUMN DATA TYPES")
            print(f"{'─' * W}")
            for col, dtype in df.dtypes.items():
                if not col.startswith("_"):  # skip internal flag cols
                    print(f"   {col:<30} {str(dtype)}")

        # ── Route results into buckets ─────────────────────────
        completeness, accuracy, consistency, uniqueness, outlier = [], [], [], [], []

        for r in results.results:
            col = r.expectation_config.kwargs.get("column", "")
            exp_type = r.expectation_config.type

            if col.startswith("_iqr_outlier_"):
                outlier.append(r)
            elif col.startswith("_is_duplicate"):
                uniqueness.append(r)
            elif col.startswith("_high_") or col.startswith("_young_"):
                consistency.append(r)
            elif "not_be_null" in exp_type:
                completeness.append(r)
            else:
                accuracy.append(r)

        def print_section(title, description, items):
            print(f"\n{'─' * W}")
            print(f"  {title}")
            if description:
                print(f"  {description}")
            print(f"{'─' * W}")
            for exp_result in items:
                exp_type = exp_result.expectation_config.type
                col = exp_result.expectation_config.kwargs.get("column", "table-level")
                status = "PASS" if exp_result.success else "FAIL"

                # Clean up column name for internal flag cols
                display_col = (
                    col.replace("_iqr_outlier_", "")
                    .replace("_is_duplicate", "all columns")
                    .replace("_high_", "")
                    .replace("_young_", "")
                )

                print(f"\n  [{status}] {exp_type}")
                print(f"   Column : {display_col}")

                if not exp_result.success and exp_result.result:
                    r = exp_result.result
                    if r.get("unexpected_count"):
                        print(f"   Issues : {r['unexpected_count']} unexpected values")
                    if r.get("partial_unexpected_list"):
                        print(f"   Sample : {r['partial_unexpected_list'][:3]}")

                # Print notes for outlier cols
                notes = exp_result.expectation_config.notes
                if notes and col.startswith("_iqr_outlier_"):
                    print(f"   Info   : {notes}")

        # ── Print each section ─────────────────────────────────
        print_section(
            "[1] COMPLETENESS CHECKS",
            "Detects missing / null values in each column.",
            completeness,
        )
        print_section(
            "[2] ACCURACY CHECKS",
            "Detects values outside expected ranges or allowed sets.",
            accuracy,
        )
        print_section(
            "[3] CONSISTENCY CHECKS",
            "Detects logical contradictions between columns.",
            consistency,
        )
        print_section(
            "[4] UNIQUENESS CHECK",
            "Detects duplicate rows across all columns.",
            uniqueness,
        )
        print_section(
            "[5] IQR OUTLIER CHECKS",
            "Detects extreme numeric values via Q1 - 1.5×IQR  /  Q3 + 1.5×IQR.",
            outlier,
        )

        print("\n" + "=" * W)


if __name__ == "__main__":
    df = pd.read_csv("./data/raw/diabetes_prediction_dataset.csv")
    validator = DataValidator(df)
    validator.run_validation()
