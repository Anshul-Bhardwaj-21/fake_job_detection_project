from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from scipy import sparse
from .config import CATEGORICAL_COLUMNS, NUMERIC_COLUMNS


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
    tfidf, ohe, scaler, X_combined = fit_feature_transformers(data)
    return X_combined, tfidf, ohe, scaler


def fit_feature_transformers(data, max_features=8000):
    """Fit feature transformers and return transformed features."""
    # TF-IDF
    tfidf = TfidfVectorizer(max_features=max_features, ngram_range=(1, 2), min_df=2)
    X_text = tfidf.fit_transform(data["clean_text"])

    # Categorical
    ohe = OneHotEncoder(handle_unknown="ignore")
    X_cat = ohe.fit_transform(data[CATEGORICAL_COLUMNS])

    # Numeric
    scaler = StandardScaler(with_mean=False)
    X_num = scaler.fit_transform(data[NUMERIC_COLUMNS])

    # Combine sparse matrices
    X_combined = sparse.hstack([X_text, X_cat, X_num])

    return tfidf, ohe, scaler, X_combined


def transform_features(data, tfidf, ohe, scaler):
    """Transform prepared rows with already fitted feature transformers."""
    X_text = tfidf.transform(data["clean_text"])
    X_cat = ohe.transform(data[CATEGORICAL_COLUMNS])
    X_num = scaler.transform(data[NUMERIC_COLUMNS])
    return sparse.hstack([X_text, X_cat, X_num])
