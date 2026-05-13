import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import silhouette_score
from .config import REPORT_DIR


def run_clustering(data, max_k=8):
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    # Use TF-IDF on clean text, reduce features for clustering
    vectorizer = TfidfVectorizer(max_features=2000, ngram_range=(1, 2))
    X = vectorizer.fit_transform(data["clean_text"])

    inertias = []
    silhouette_scores = []
    k_values = list(range(2, max_k + 1))

    for k in k_values:
        model = MiniBatchKMeans(n_clusters=k, random_state=42, batch_size=1000, n_init=10)
        labels = model.fit_predict(X)
        inertias.append(model.inertia_)
        try:
            silhouette_scores.append(float(silhouette_score(X, labels, sample_size=1000)))
        except Exception:
            silhouette_scores.append(0.0)

    best_index = max(range(len(silhouette_scores)), key=lambda i: silhouette_scores[i])
    best_k = k_values[best_index]
    final_model = MiniBatchKMeans(n_clusters=best_k, random_state=42, batch_size=1000, n_init=10)
    data = data.copy()
    data["cluster"] = final_model.fit_predict(X)

    # Elbow method plot
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(k_values, inertias, marker="o", linestyle="-", color="b")
    ax.set_xlabel("Number of Clusters (k)")
    ax.set_ylabel("WCSS / Inertia")
    ax.set_title("Elbow Method for Optimal k")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(REPORT_DIR / "clustering_elbow.png", dpi=160)
    plt.close(fig)

    # Cluster summary
    summary = data.groupby("cluster").agg(
        total_jobs=("cluster", "count"),
        fake_jobs=("fraudulent", "sum"),
        avg_suspicious_keywords=("suspicious_keyword_count", "mean"),
        avg_fee_keywords=("fee_keyword_count", "mean"),
        avg_urgency_keywords=("urgency_keyword_count", "mean"),
        avg_contact_risk_keywords=("contact_risk_keyword_count", "mean"),
        avg_sensitive_info_keywords=("sensitive_info_keyword_count", "mean"),
        avg_profile_missing=("profile_missing", "mean"),
        avg_salary_missing=("salary_missing", "mean"),
    ).reset_index()
    summary["fake_ratio"] = summary["fake_jobs"] / summary["total_jobs"]
    summary.to_csv(REPORT_DIR / "cluster_summary.csv", index=False)

    return {
        "best_k": best_k,
        "silhouette_scores": dict(zip(k_values, silhouette_scores)),
        "cluster_summary": summary.to_dict(orient="records"),
    }
