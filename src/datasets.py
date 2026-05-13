import re
from pathlib import Path

import pandas as pd

from .config import (
    BINARY_COLUMNS,
    CATEGORICAL_COLUMNS,
    INDIAN_SYNTHETIC_DATASET,
    SAMPLE_DATASET,
    TARGET_COLUMN,
    TEXT_COLUMNS,
)
from .preprocessing import load_dataset
from .utils import get_dataset_path


INDIAN_SYNTHETIC_URL = (
    "https://huggingface.co/api/datasets/"
    "Aioshi/smolified-fakejob/parquet/default/train/0.parquet"
)


def normalize_indian_synthetic_dataset(raw_data: pd.DataFrame) -> pd.DataFrame:
    """Convert the Hugging Face Indian synthetic fake-job set into project schema."""
    labels = raw_data["assistant"].fillna("").str.extract(
        r"Classification:\s*(Fake|Real)", flags=re.IGNORECASE, expand=False
    )
    valid = labels.str.lower().isin(["fake", "real"])

    job_text = raw_data.loc[valid, "user"].fillna("").astype(str)
    job_text = job_text.str.replace(r"^\s*Job Posting:\s*", "", regex=True)

    normalized = pd.DataFrame({
        "title": "Synthetic Indian Job Posting",
        "company_profile": "",
        "description": job_text,
        "requirements": "",
        "benefits": "",
        "extra_text": "",
        "employment_type": "Unknown",
        "required_experience": "Unknown",
        "required_education": "Unknown",
        "industry": "Unknown",
        "function": "Unknown",
        "telecommuting": 0,
        "has_company_logo": 0,
        "has_questions": 0,
        "fraudulent": labels.loc[valid].str.lower().map({"real": 0, "fake": 1}).astype(int),
    })

    return normalized.reset_index(drop=True)


def download_indian_synthetic_dataset(path: Path = INDIAN_SYNTHETIC_DATASET) -> Path:
    """Download and save the additional Indian fake-job dataset as CSV."""
    raw_data = pd.read_parquet(INDIAN_SYNTHETIC_URL)
    normalized = normalize_indian_synthetic_dataset(raw_data)
    path.parent.mkdir(parents=True, exist_ok=True)
    normalized.to_csv(path, index=False)
    return path


def load_training_dataset(include_auxiliary: bool = True):
    """Load the main dataset plus any available auxiliary training datasets."""
    primary_path = get_dataset_path()
    frames = [load_dataset(primary_path)]
    sources = [{
        "name": "primary",
        "path": str(primary_path),
        "rows": len(frames[0]),
    }]

    if include_auxiliary and INDIAN_SYNTHETIC_DATASET.exists():
        auxiliary = load_dataset(INDIAN_SYNTHETIC_DATASET)
        frames.append(auxiliary)
        sources.append({
            "name": "indian_synthetic_huggingface",
            "path": str(INDIAN_SYNTHETIC_DATASET),
            "rows": len(auxiliary),
        })

    combined = pd.concat(frames, ignore_index=True)
    dedupe_columns = [*TEXT_COLUMNS, *CATEGORICAL_COLUMNS, *BINARY_COLUMNS, TARGET_COLUMN]
    combined = combined.drop_duplicates(subset=[col for col in dedupe_columns if col in combined.columns])

    if primary_path == SAMPLE_DATASET:
        sources[0]["warning"] = "sample dataset only; download the full Kaggle/Hugging Face mirror for real training"

    return combined, sources
