from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data" / "raw"
PROCESSED_DIR = ROOT_DIR / "data" / "processed"
REPORT_DIR = ROOT_DIR / "reports" / "figures"
MODEL_DIR = ROOT_DIR / "models"

KAGGLE_DATASET = DATA_DIR / "fake_job_postings.csv"
SAMPLE_DATASET = DATA_DIR / "sample_fake_job_postings.csv"
INDIAN_SYNTHETIC_DATASET = DATA_DIR / "indian_fake_job_postings_synthetic.csv"
PROCESSED_DATASET = PROCESSED_DIR / "processed_jobs.csv"
MODEL_PATH = MODEL_DIR / "fake_job_model.pkl"
TFIDF_PATH = MODEL_DIR / "tfidf_vectorizer.pkl"
LABEL_ENCODERS_PATH = MODEL_DIR / "label_encoders.pkl"
FEATURE_CONFIG_PATH = MODEL_DIR / "feature_config.pkl"
METRICS_PATH = MODEL_DIR / "metrics.json"

TEXT_COLUMNS = ["title", "company_profile", "description", "requirements", "benefits", "extra_text"]
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
    "fee_keyword_count",
    "urgency_keyword_count",
    "contact_risk_keyword_count",
    "sensitive_info_keyword_count",
    "profile_missing",
    "salary_missing",
]

SUSPICIOUS_KEYWORDS = [
    "registration fee",
    "joining fee",
    "processing fee",
    "document fee",
    "verification fee",
    "security deposit",
    "refundable deposit",
    "urgent hiring",
    "immediate joining",
    "no experience",
    "daily payment",
    "daily income",
    "guaranteed income",
    "guaranteed job",
    "100% placement",
    "easy money",
    "work from home",
    "work from mobile",
    "limited seats",
    "limited slots",
    "training fee",
    "pay first",
    "no interview",
    "whatsapp",
    "telegram",
    "aadhar",
    "aadhaar",
    "bank details",
]

FEE_KEYWORDS = [
    "registration fee",
    "joining fee",
    "processing fee",
    "document fee",
    "documentation fee",
    "verification fee",
    "security deposit",
    "refundable deposit",
    "training fee",
    "pay first",
    "deposit",
    "fee required",
    "starter kit",
]

URGENCY_KEYWORDS = [
    "urgent hiring",
    "immediate joining",
    "apply immediately",
    "limited seats",
    "limited slots",
    "selected today",
    "instant selection",
    "fast selection",
    "no interview",
    "guaranteed job",
    "100% placement",
]

CONTACT_RISK_KEYWORDS = [
    "whatsapp",
    "telegram",
    "dm now",
    "personal mobile",
    "send resume to mobile",
    "gmail.com",
    "protonmail",
    "non-official link",
]

SENSITIVE_INFO_KEYWORDS = [
    "aadhar",
    "aadhaar",
    "pan card",
    "bank details",
    "account number",
    "ifsc",
    "upi",
    "id proof",
    "passport copy",
]

SALARY_KEYWORDS = [
    "salary",
    "pay",
    "compensation",
    "ctc",
    "lpa",
    "stipend",
    "per month",
    "per annum",
    "rs",
    "rupees",
]
