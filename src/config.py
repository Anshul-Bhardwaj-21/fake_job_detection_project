from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data" / "raw"
PROCESSED_DIR = ROOT_DIR / "data" / "processed"
REPORT_DIR = ROOT_DIR / "reports" / "figures"
MODEL_DIR = ROOT_DIR / "models"

KAGGLE_DATASET = DATA_DIR / "fake_job_postings.csv"
SAMPLE_DATASET = DATA_DIR / "sample_fake_job_postings.csv"
PROCESSED_DATASET = PROCESSED_DIR / "processed_jobs.csv"
MODEL_PATH = MODEL_DIR / "fake_job_model.pkl"
TFIDF_PATH = MODEL_DIR / "tfidf_vectorizer.pkl"
LABEL_ENCODERS_PATH = MODEL_DIR / "label_encoders.pkl"
FEATURE_CONFIG_PATH = MODEL_DIR / "feature_config.pkl"
METRICS_PATH = MODEL_DIR / "metrics.json"

TEXT_COLUMNS = ["title", "company_profile", "description", "requirements", "benefits"]
CATEGORICAL_COLUMNS = ["employment_type", "required_experience", "required_education", "industry", "function"]
BINARY_COLUMNS = ["telecommuting", "has_company_logo", "has_questions"]
TARGET_COLUMN = "fraudulent"

NUMERIC_COLUMNS = [
    "telecommuting",
    "has_company_logo",
    "has_questions",
    "description_length",
    "requirements_length",
    "company_profile_length",
    "suspicious_keyword_count",
    "profile_missing",
    "salary_missing",
]

SUSPICIOUS_KEYWORDS = [
    "registration fee",
    "joining fee",
    "urgent hiring",
    "no experience",
    "daily payment",
    "guaranteed income",
    "easy money",
    "work from home",
    "limited seats",
    "training fee",
    "pay first",
    "no interview",
]
