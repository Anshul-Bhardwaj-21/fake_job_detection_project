import json
from src.datasets import load_training_dataset
from src.preprocessing import prepare_dataframe
from src.model_training import train_and_evaluate
from src.association_rules import generate_simple_association_rules
from src.clustering import run_clustering
from src.config import REPORT_DIR


def cleanup_reports():
    """Remove old report files before a fresh training run."""
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    removed = 0
    for pattern in ["*.png", "*.csv", "*.json"]:
        for old_file in REPORT_DIR.glob(pattern):
            try:
                old_file.unlink()
                removed += 1
            except Exception:
                pass
    if removed > 0:
        print(f"Removed {removed} old report file(s) from {REPORT_DIR}.")


def main():
    raw_data, dataset_sources = load_training_dataset()

    print("Using training datasets:")
    for source in dataset_sources:
        print(f"- {source['name']}: {source['rows']} rows ({source['path']})")
        if "warning" in source:
            print(f"  Warning: {source['warning']}")

    cleanup_reports()

    prepared = prepare_dataframe(raw_data)

    print("Training and evaluating models...")
    metrics = train_and_evaluate(raw_data, dataset_sources=dataset_sources)

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
