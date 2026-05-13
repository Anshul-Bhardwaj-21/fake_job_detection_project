from pathlib import Path
from .config import KAGGLE_DATASET, SAMPLE_DATASET


def get_dataset_path():
    if KAGGLE_DATASET.exists():
        return KAGGLE_DATASET
    return SAMPLE_DATASET


def risk_level(score):
    if score <= 30:
        return "Low Risk"
    if score <= 70:
        return "Medium Risk"
    return "High Risk"


def warning_indicators(prepared_row):
    row = prepared_row.iloc[0]
    warnings = []
    if row.get("profile_missing", 0) == 1:
        warnings.append("Company profile is missing")
    if row.get("has_company_logo", 0) == 0:
        warnings.append("Company logo is missing")
    if row.get("has_questions", 0) == 0:
        warnings.append("Screening questions are missing")
    if row.get("suspicious_keyword_count", 0) >= 1:
        warnings.append("Suspicious keywords found")
    if row.get("fee_keyword_count", 0) >= 1:
        warnings.append("Payment, deposit, or fee request found")
    if row.get("urgency_keyword_count", 0) >= 1:
        warnings.append("Urgency or guaranteed-selection language found")
    if row.get("contact_risk_keyword_count", 0) >= 1:
        warnings.append("Unprofessional contact channel or email pattern found")
    if row.get("sensitive_info_keyword_count", 0) >= 1:
        warnings.append("Sensitive document or bank-detail request found")
    if row.get("salary_missing", 0) == 1:
        warnings.append("Salary information is missing")
    if row.get("requirements_length", 0) < 5:
        warnings.append("Requirements are too brief or missing")
    return warnings or ["No strong suspicious indicator found"]
