from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from scipy import sparse
from .config import CATEGORICAL_COLUMNS, BINARY_COLUMNS

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


def build_pipeline(classifier=None):
    if classifier is None:
        classifier = LogisticRegression(max_iter=1000, class_weight="balanced")

    # TF-IDF vectorizer for text
    tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))

    # One-hot encoder for categorical
    ohe = OneHotEncoder(handle_unknown="ignore")

    # Scaler for numeric (though for sparse, we might not scale)
    scaler = StandardScaler(with_mean=False)

    preprocessor = ColumnTransformer(
        transformers=[
            ("text", tfidf, "clean_text"),
            ("cat", ohe, CATEGORICAL_COLUMNS),
            ("num", scaler, NUMERIC_COLUMNS),
        ],
        remainder="drop"
    )

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", classifier)
    ])

    return pipeline


def build_features_separate(data):
    """Build features separately for saving vectorizers and handling sparse matrices."""
    # TF-IDF
    tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    X_text = tfidf.fit_transform(data["clean_text"])

    # Categorical
    ohe = OneHotEncoder(handle_unknown="ignore")
    X_cat = ohe.fit_transform(data[CATEGORICAL_COLUMNS])

    # Numeric
    scaler = StandardScaler(with_mean=False)
    X_num = scaler.fit_transform(data[NUMERIC_COLUMNS])

    # Combine sparse matrices
    X_combined = sparse.hstack([X_text, X_cat, X_num])

    return X_combined, tfidf, ohe, scaler
