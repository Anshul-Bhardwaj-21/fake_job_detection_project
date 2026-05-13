import pandas as pd
import re
from .config import (
    BINARY_COLUMNS,
    CATEGORICAL_COLUMNS,
    CONTACT_RISK_KEYWORDS,
    FEE_KEYWORDS,
    PROCESSED_DATASET,
    PROCESSED_DIR,
    SALARY_KEYWORDS,
    SENSITIVE_INFO_KEYWORDS,
    SUSPICIOUS_KEYWORDS,
    TARGET_COLUMN,
    TEXT_COLUMNS,
    URGENCY_KEYWORDS,
)


def load_dataset(path):
    data = pd.read_csv(path)
    return normalize_schema(data)


def normalize_schema(data: pd.DataFrame) -> pd.DataFrame:
    """Ensure required columns exist for both Kaggle and sample dataset."""
    data = data.copy()

    for col in TEXT_COLUMNS:
        if col not in data.columns:
            data[col] = ""

    for col in CATEGORICAL_COLUMNS:
        if col not in data.columns:
            data[col] = "Unknown"

    for col in BINARY_COLUMNS:
        if col not in data.columns:
            data[col] = 0

    if TARGET_COLUMN not in data.columns:
        raise ValueError(f"Dataset must contain target column: {TARGET_COLUMN}")

    return data


def clean_text(text: str) -> str:
    if pd.isna(text):
        return ""
    text = str(text).lower()
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def prepare_dataframe(data: pd.DataFrame) -> pd.DataFrame:
    data = normalize_schema(data)

    for col in TEXT_COLUMNS:
        data[col] = data[col].fillna("").astype(str)

    for col in CATEGORICAL_COLUMNS:
        data[col] = data[col].fillna("Unknown").astype(str)

    for col in BINARY_COLUMNS:
        data[col] = pd.to_numeric(data[col], errors="coerce").fillna(0).astype(int)

    data[TARGET_COLUMN] = pd.to_numeric(data[TARGET_COLUMN], errors="coerce").fillna(0).astype(int)

    # Remove duplicates
    data = data.drop_duplicates()

    data["combined_text"] = data[TEXT_COLUMNS].agg(" ".join, axis=1)
    data["clean_text"] = data["combined_text"].apply(clean_text)

    # Length features
    data["description_length"] = data["description"].apply(lambda x: len(str(x).split()))
    data["requirements_length"] = data["requirements"].apply(lambda x: len(str(x).split()))
    data["company_profile_length"] = data["company_profile"].apply(lambda x: len(str(x).split()))

    # Missing indicators
    data["profile_missing"] = data["company_profile"].apply(lambda x: 1 if len(str(x).strip()) == 0 else 0)
    data["salary_missing"] = data["combined_text"].apply(
        lambda x: 0 if count_keywords(x, SALARY_KEYWORDS) > 0 else 1
    )

    # Suspicious and scam-pattern keyword features
    data["suspicious_keyword_count"] = data["combined_text"].apply(count_suspicious_keywords)
    data["fee_keyword_count"] = data["combined_text"].apply(lambda x: count_keywords(x, FEE_KEYWORDS))
    data["urgency_keyword_count"] = data["combined_text"].apply(lambda x: count_keywords(x, URGENCY_KEYWORDS))
    data["contact_risk_keyword_count"] = data["combined_text"].apply(lambda x: count_keywords(x, CONTACT_RISK_KEYWORDS))
    data["sensitive_info_keyword_count"] = data["combined_text"].apply(lambda x: count_keywords(x, SENSITIVE_INFO_KEYWORDS))

    return data


def save_processed_data(data: pd.DataFrame):
    """Save the processed dataset."""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    data.to_csv(PROCESSED_DATASET, index=False)
    print(f"Processed data saved to: {PROCESSED_DATASET}")


def count_suspicious_keywords(text: str) -> int:
    return count_keywords(text, SUSPICIOUS_KEYWORDS)


def count_keywords(text: str, keywords: list[str]) -> int:
    text = str(text).lower()
    return sum(1 for keyword in keywords if keyword in text)


def make_document_style_records(data: pd.DataFrame) -> pd.DataFrame:
    """Collapse structured rows into one text block to mimic PDFs and URLs."""
    document_rows = data.copy()
    text_columns = [col for col in TEXT_COLUMNS if col in document_rows.columns]
    document_rows["description"] = document_rows[text_columns].fillna("").astype(str).agg(" ".join, axis=1)

    for col in ("company_profile", "requirements", "benefits", "extra_text"):
        document_rows[col] = ""

    document_rows["has_company_logo"] = 0
    document_rows["has_questions"] = 0
    return document_rows


def build_user_record(title, company_profile, description, requirements, benefits,
                      employment_type="Unknown", required_experience="Unknown",
                      required_education="Unknown", industry="Unknown", function="Unknown",
                      telecommuting=0, has_company_logo=0, has_questions=0,
                      extra_text=""):
    return pd.DataFrame([{
        "title": title,
        "company_profile": company_profile,
        "description": description,
        "requirements": requirements,
        "benefits": benefits,
        "extra_text": extra_text,
        "employment_type": employment_type,
        "required_experience": required_experience,
        "required_education": required_education,
        "industry": industry,
        "function": function,
        "telecommuting": int(telecommuting),
        "has_company_logo": int(has_company_logo),
        "has_questions": int(has_questions),
        "fraudulent": 0,
    }])
