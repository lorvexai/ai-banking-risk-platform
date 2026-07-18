# exercises/data_quality.py
# Exercise 15.1: Great Expectations rules for credit data
# Difficulty: ★★★☆☆  |  Estimated time: 25 minutes
# AWB-AI-2025 | chapter_15/exercises/
#
# TASK: Implement 5 Great Expectations rules that catch:
#   1. Null borrower_id values
#   2. Negative requested_amount_gbp
#   3. application_date in the future
#   4. interest_rate outside range 0.01–30.0%
#   5. loan_term_months not in {12, 24, 36, 48, 60}
#
# Run against exercises/sample_credit_data.csv and confirm
# all 3 deliberately injected errors are detected.
#
# Success: all 5 expectations defined; 3 synthetic errors
# caught; validation result shows success=False (errors found)
#
# Solution: chapter_15/solutions/ex1_data_quality.py
import great_expectations as ge
from datetime import date
import pandas as pd


def build_credit_expectations(
    df: pd.DataFrame,
) -> "ge.dataset.PandasDataset":
    """Build AWB credit data quality expectation suite.

    Applies 5 Great Expectations rules to the credit
    application DataFrame. These rules mirror AWB's
    production BCBS 239 P3 (Accuracy) controls.

    Args:
        df: Pandas DataFrame of credit applications with
            columns: borrower_id, requested_amount_gbp,
            application_date, interest_rate,
            loan_term_months.

    Returns:
        GE dataset with expectations applied and results
        available via .validate().

    TODO: Add the 5 expectations below.
    Hint: ge_df.expect_column_values_to_not_be_null(col)
    """
    ge_df = ge.from_pandas(df)

    # TODO 1: borrower_id must not be null
    # ge_df.expect_column_values_to_not_be_null(
    #     "borrower_id"
    # )

    # TODO 2: requested_amount_gbp must be positive
    # ge_df.expect_column_values_to_be_between(
    #     "requested_amount_gbp",
    #     min_value=0.01,
    # )

    # TODO 3: application_date must not be in the future
    today = date.today().isoformat()
    # ge_df.expect_column_values_to_be_between(
    #     "application_date",
    #     max_value=today,
    # )

    # TODO 4: interest_rate between 0.01 and 30.0
    # ge_df.expect_column_values_to_be_between(
    #     "interest_rate",
    #     min_value=0.01,
    #     max_value=30.0,
    # )

    # TODO 5: loan_term_months in {12, 24, 36, 48, 60}
    valid_terms = [12, 24, 36, 48, 60]
    # ge_df.expect_column_values_to_be_in_set(
    #     "loan_term_months", valid_terms
    # )

    return ge_df


if __name__ == "__main__":
    df = pd.read_csv("exercises/sample_credit_data.csv")
    ge_df = build_credit_expectations(df)
    result = ge_df.validate()
    passed = sum(
        1 for r in result["results"] if r["success"]
    )
    total = len(result["results"])
    print(f"Passed {passed}/{total} expectations")
    if not result["success"]:
        print("FAILED expectations (errors caught):")
        for r in result["results"]:
            if not r["success"]:
                exp = r["expectation_config"]
                print(f"  - {exp['expectation_type']}")
                print(
                    f"    column: "
                    f"{exp['kwargs'].get('column','')}"
                )
