"""AWB Consumer Loan — Feature Engineering.

Transforms raw application, Open Banking, and credit bureau
inputs into the 14 model features required by MR-2026-041.

Open Banking features are imputed at population median
(stratified by income decile) when connection is unavailable.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

log = logging.getLogger(__name__)

OB_FEATURES = [
    "ob_income_volatility",
    "ob_discretionary_spend_ratio",
    "ob_overdraft_days_90",
    "ob_gambling_flag",
    "ob_bill_payment_timeliness",
    "ob_current_account_tenure_yrs",
    "ob_net_cashflow_trend",
]

# Population medians by income decile (fitted on training set)
# Decile 1 = lowest income, 10 = highest
OB_MEDIANS: dict[int, dict[str, float]] = {
    1: {"ob_income_volatility": 0.42, "ob_discretionary_spend_ratio": 0.19,
        "ob_overdraft_days_90": 8.2, "ob_gambling_flag": 0.14,
        "ob_bill_payment_timeliness": 0.81, "ob_current_account_tenure_yrs": 3.1,
        "ob_net_cashflow_trend": -42.0},
    5: {"ob_income_volatility": 0.22, "ob_discretionary_spend_ratio": 0.26,
        "ob_overdraft_days_90": 3.1, "ob_gambling_flag": 0.08,
        "ob_bill_payment_timeliness": 0.91, "ob_current_account_tenure_yrs": 5.8,
        "ob_net_cashflow_trend": 85.0},
    10: {"ob_income_volatility": 0.14, "ob_discretionary_spend_ratio": 0.33,
         "ob_overdraft_days_90": 0.8, "ob_gambling_flag": 0.04,
         "ob_bill_payment_timeliness": 0.97, "ob_current_account_tenure_yrs": 9.2,
         "ob_net_cashflow_trend": 310.0},
}


@dataclass
class RawApplication:
    """Raw inputs before feature engineering."""
    application_id: str
    # Application form (14 fields)
    requested_amount_gbp: float
    loan_term_months: int
    purpose: str
    gross_annual_income: float
    employment_status: str
    employment_tenure_months: int
    residential_status: str
    time_at_address_months: int
    num_dependants: int
    monthly_housing_cost: float
    existing_monthly_commitments: float
    # Credit bureau
    bureau_score: int
    bureau_adverse_flag: int      # 1 if CCJ/default/bankruptcy
    bureau_utilisation: float
    # Open Banking (optional)
    ob_connected: bool = False
    ob_income_volatility: Optional[float] = None
    ob_discretionary_spend_ratio: Optional[float] = None
    ob_overdraft_days_90: Optional[float] = None
    ob_gambling_flag: Optional[float] = None
    ob_bill_payment_timeliness: Optional[float] = None
    ob_current_account_tenure_yrs: Optional[float] = None
    ob_net_cashflow_trend: Optional[float] = None


class FeatureEngineer:
    """Transform raw application inputs into model features.

    Handles:
    - Derived ratio computation
    - Open Banking imputation for non-connected applicants
    - Validation of feature ranges
    """

    def engineer(self, app: RawApplication) -> dict:
        """Produce the 14-feature dict for MR-2026-041.

        Args:
            app: Raw application data.

        Returns:
            dict mapping FEATURE_NAMES to float values.
        """
        # Derived ratios
        net_monthly = app.gross_annual_income / 12 * 0.72
        dti = (
            (app.existing_monthly_commitments +
             app.requested_amount_gbp / app.loan_term_months)
            / max(net_monthly, 1.0)
        )
        housing_ratio = app.monthly_housing_cost / max(net_monthly, 1.0)
        amount_to_income = app.requested_amount_gbp / max(
            app.gross_annual_income, 1.0
        )

        features = {
            "debt_to_income":           round(min(dti, 5.0), 4),
            "housing_cost_ratio":       round(min(housing_ratio, 1.0), 4),
            "credit_bureau_score":      float(app.bureau_score),
            "existing_credit_utilisation": round(
                min(app.bureau_utilisation, 1.0), 4
            ),
            "adverse_history_flag":     float(app.bureau_adverse_flag),
            "employment_tenure_months": float(app.employment_tenure_months),
            "requested_amount_to_income": round(amount_to_income, 4),
        }

        # Open Banking features
        if app.ob_connected and app.ob_income_volatility is not None:
            features.update({
                "ob_income_volatility":         app.ob_income_volatility,
                "ob_discretionary_spend_ratio": app.ob_discretionary_spend_ratio,
                "ob_overdraft_days_90":         app.ob_overdraft_days_90,
                "ob_gambling_flag":             app.ob_gambling_flag,
                "ob_bill_payment_timeliness":   app.ob_bill_payment_timeliness,
                "ob_current_account_tenure_yrs": app.ob_current_account_tenure_yrs,
                "ob_net_cashflow_trend":         app.ob_net_cashflow_trend,
            })
        else:
            features.update(self._impute_ob(app.gross_annual_income))
            log.info(
                "app=%s: Open Banking unavailable — using imputed values",
                app.application_id,
            )

        return features

    def _impute_ob(self, gross_income: float) -> dict:
        """Impute Open Banking features at population median.

        Stratified by income decile (decile 1–10 based on
        ONS income distribution, 2025).
        """
        decile = self._income_decile(gross_income)
        # Find nearest decile with medians defined
        candidates = [1, 5, 10]
        nearest = min(candidates, key=lambda d: abs(d - decile))
        return OB_MEDIANS[nearest]

    @staticmethod
    def _income_decile(gross_income: float) -> int:
        """Map gross annual income to ONS decile (2025 thresholds)."""
        thresholds = [
            14_000, 18_500, 22_000, 27_000, 32_000,
            38_000, 46_000, 57_000, 75_000
        ]
        for i, t in enumerate(thresholds, 1):
            if gross_income < t:
                return i
        return 10
