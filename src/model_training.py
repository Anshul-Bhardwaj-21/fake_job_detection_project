import json
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import ComplementNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.base import clone
from .features import fit_feature_transformers, transform_features
from .preprocessing import make_document_style_records, prepare_dataframe, save_processed_data
from .config import TARGET_COLUMN, MODEL_PATH, TFIDF_PATH, LABEL_ENCODERS_PATH, FEATURE_CONFIG_PATH, METRICS_PATH, REPORT_DIR


def train_and_evaluate(raw_data: pd.DataFrame, dataset_sources=None):
    data = prepare_dataframe(raw_data)
    save_processed_data(data)

    train_df, test_df = train_test_split(
        data, test_size=0.2, random_state=42, stratify=data[TARGET_COLUMN]
    )
    train_augmented = prepare_dataframe(
        pd.concat([train_df, make_document_style_records(train_df)], ignore_index=True)
    )

    tfidf, ohe, scaler, X_train = fit_feature_transformers(train_augmented)
    X_test = transform_features(test_df, tfidf, ohe, scaler)
    y_train = train_augmented[TARGET_COLUMN]
    y_test = test_df[TARGET_COLUMN]

    candidate_models = {
        "Complement Naive Bayes": ComplementNB(),
        "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
        "Decision Tree": DecisionTreeClassifier(random_state=42, class_weight="balanced"),
        "Random Forest": RandomForestClassifier(n_estimators=120, random_state=42, class_weight="balanced"),
        "Linear SVM": LinearSVC(class_weight="balanced", random_state=42, max_iter=10000),
        "Calibrated Linear SVM": CalibratedClassifierCV(
            estimator=LinearSVC(class_weight="balanced", random_state=42, max_iter=10000),
            cv=3,
        ),
    }

    results = []
    trained = {}

    for name, clf in candidate_models.items():
        try:
            clf.fit(X_train, y_train)
            preds = clf.predict(X_test)

            result = {
                "model": name,
                "accuracy": float(accuracy_score(y_test, preds)),
                "precision": float(precision_score(y_test, preds, zero_division=0)),
                "recall": float(recall_score(y_test, preds, zero_division=0)),
                "f1_score": float(f1_score(y_test, preds, zero_division=0)),
            }
            results.append(result)
            trained[name] = clf
        except Exception as e:
            print(f"Error training {name}: {e}")
            continue

    results_df = pd.DataFrame(results).sort_values(by=["f1_score", "recall"], ascending=False)
    best_name = results_df.iloc[0]["model"]
    best_model = trained[best_name]

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    # Generate confusion matrix for best model
    best_preds = best_model.predict(X_test)
    cm = confusion_matrix(y_test, best_preds)
    doc_test_df = prepare_dataframe(make_document_style_records(test_df))
    X_doc_test = transform_features(doc_test_df, tfidf, ohe, scaler)
    doc_test_preds = best_model.predict(X_doc_test)
    doc_cm = confusion_matrix(y_test, doc_test_preds)

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['Real', 'Fake'], yticklabels=['Real', 'Fake'])
    ax.set_title(f"Confusion Matrix - {best_name}")
    ax.set_ylabel('True Label')
    ax.set_xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(REPORT_DIR / "confusion_matrix.png", dpi=160)
    plt.close(fig)

    # Model comparison chart
    fig, ax = plt.subplots(figsize=(10, 6))
    results_df.set_index('model')[['accuracy', 'precision', 'recall', 'f1_score']].plot(kind='bar', ax=ax)
    ax.set_title("Model Comparison")
    ax.set_ylabel("Score")
    ax.set_xlabel("Model")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(REPORT_DIR / "model_comparison.png", dpi=160)
    plt.close(fig)

    # Class distribution
    fig, ax = plt.subplots(figsize=(6, 4))
    data[TARGET_COLUMN].value_counts().sort_index().plot(kind="bar", ax=ax, color=['skyblue', 'salmon'])
    ax.set_xticklabels(["Real", "Fake"], rotation=0)
    ax.set_title("Class Distribution")
    ax.set_ylabel("Count")
    ax.set_xlabel("Job Type")
    plt.tight_layout()
    plt.savefig(REPORT_DIR / "class_distribution.png", dpi=160)
    plt.close(fig)

    # Classification report for best model
    report = classification_report(y_test, best_preds, target_names=['Real', 'Fake'], output_dict=True)
    doc_report = classification_report(y_test, doc_test_preds, target_names=['Real', 'Fake'], output_dict=True)

    # Refit the selected model on all available data for deployment after honest holdout evaluation.
    final_data = prepare_dataframe(
        pd.concat([data, make_document_style_records(data)], ignore_index=True)
    )
    final_tfidf, final_ohe, final_scaler, X_final = fit_feature_transformers(final_data)
    final_model = clone(candidate_models[best_name])
    final_model.fit(X_final, final_data[TARGET_COLUMN])

    joblib.dump(final_model, MODEL_PATH)
    joblib.dump(final_tfidf, TFIDF_PATH)
    joblib.dump(final_ohe, LABEL_ENCODERS_PATH)
    joblib.dump(final_scaler, FEATURE_CONFIG_PATH)

    metrics = {
        "best_model": best_name,
        "results": results,
        "confusion_matrix": cm.tolist(),
        "document_style_confusion_matrix": doc_cm.tolist(),
        "classification_report": report,
        "document_style_classification_report": doc_report,
        "dataset_sources": dataset_sources or [],
        "dataset_info": {
            "total_samples": len(data),
            "training_samples": len(train_augmented),
            "test_samples": len(test_df),
            "final_training_samples": len(final_data),
            "feature_count": X_train.shape[1]
        }
    }

    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    return metrics
