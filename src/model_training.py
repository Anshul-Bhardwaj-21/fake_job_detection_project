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
from sklearn.neighbors import KNeighborsClassifier
from .features import build_features_separate
from .preprocessing import prepare_dataframe, save_processed_data
from .config import TARGET_COLUMN, MODEL_PATH, TFIDF_PATH, LABEL_ENCODERS_PATH, FEATURE_CONFIG_PATH, METRICS_PATH, REPORT_DIR


def train_and_evaluate(raw_data: pd.DataFrame):
    data = prepare_dataframe(raw_data)
    save_processed_data(data)

    X_combined, tfidf, ohe, scaler = build_features_separate(data)
    y = data[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X_combined, y, test_size=0.2, random_state=42, stratify=y
    )

    candidate_models = {
        "Complement Naive Bayes": ComplementNB(),
        "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
        "Decision Tree": DecisionTreeClassifier(random_state=42, class_weight="balanced"),
        "Random Forest": RandomForestClassifier(n_estimators=120, random_state=42, class_weight="balanced"),
        "Linear SVM": LinearSVC(class_weight="balanced", random_state=42, max_iter=10000),
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

    # Save model and components
    joblib.dump(best_model, MODEL_PATH)
    joblib.dump(tfidf, TFIDF_PATH)
    joblib.dump(ohe, LABEL_ENCODERS_PATH)
    joblib.dump(scaler, FEATURE_CONFIG_PATH)

    # Generate confusion matrix for best model
    best_preds = best_model.predict(X_test)
    cm = confusion_matrix(y_test, best_preds)

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

    metrics = {
        "best_model": best_name,
        "results": results,
        "confusion_matrix": cm.tolist(),
        "classification_report": report,
        "dataset_info": {
            "total_samples": len(data),
            "training_samples": X_train.shape[0],
            "test_samples": X_test.shape[0],
            "feature_count": X_combined.shape[1]
        }
    }

    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    return metrics

    return metrics
