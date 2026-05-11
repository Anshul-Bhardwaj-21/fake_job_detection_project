import json
from src.utils import get_dataset_path
from src.preprocessing import load_dataset, prepare_dataframe
from src.model_training import train_and_evaluate
from src.association_rules import generate_simple_association_rules
from src.clustering import run_clustering
from src.config import REPORT_DIR, KAGGLE_DATASET, SAMPLE_DATASET


def main():
    dataset_path = get_dataset_path()

    if dataset_path == KAGGLE_DATASET and KAGGLE_DATASET.exists():
        print(f"Using real Kaggle dataset: {dataset_path}")
    elif dataset_path == SAMPLE_DATASET:
        print("Using sample dataset only. For final training, download Kaggle dataset.")
        print("Run: python scripts/download_dataset.py")
    else:
        print(f"Using dataset: {dataset_path}")

    raw_data = load_dataset(dataset_path)
    prepared = prepare_dataframe(raw_data)

    print("Training and evaluating models...")
    metrics = train_and_evaluate(raw_data)

    print("Generating association rules...")
    rules = generate_simple_association_rules(prepared)
    print(rules.head(10) if not rules.empty else "No rules generated.")

    print("Running clustering...")
    clustering_result = run_clustering(prepared)

    with open(REPORT_DIR / "clustering_result.json", "w", encoding="utf-8") as f:
        json.dump(clustering_result, f, indent=2)

    print("\nTraining complete.")
    print("Best model:", metrics["best_model"])
    print("Generated files are saved in models/ and reports/figures/")


if __name__ == "__main__":
    main()
