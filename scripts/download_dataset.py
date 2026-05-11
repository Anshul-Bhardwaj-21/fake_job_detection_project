#!/usr/bin/env python3
"""
Dataset Download Script for Fake Job Posting Detection Project

This script downloads the Kaggle dataset "Real or Fake: Fake Job Posting Prediction"
and places it in the correct location for the project.

Requirements:
- kaggle package installed
- Kaggle API credentials configured (kaggle.json in ~/.kaggle/ or C:\\Users\\USERNAME\\.kaggle\\)

Usage:
    python scripts/download_dataset.py
"""

import os
import sys
from pathlib import Path
import zipfile
import shutil

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from config import DATA_DIR, KAGGLE_DATASET

def check_kaggle_credentials():
    """Check if Kaggle API credentials are available."""
    kaggle_dir = Path.home() / ".kaggle"
    kaggle_json = kaggle_dir / "kaggle.json"

    if not kaggle_json.exists():
        return False

    # Check if credentials have content
    try:
        with open(kaggle_json, 'r') as f:
            content = f.read().strip()
            return len(content) > 0
    except:
        return False

def download_kaggle_dataset():
    """Download and extract the Kaggle dataset."""
    try:
        import kaggle
    except ImportError:
        print("Error: kaggle package not installed. Run: pip install kaggle")
        return False

    if not check_kaggle_credentials():
        print("Kaggle API credentials not found.")
        print("To set up Kaggle API:")
        print("1. Go to https://www.kaggle.com/account")
        print("2. Click 'Create New API Token' to download kaggle.json")
        print("3. Place kaggle.json in one of these locations:")
        print("   - ~/.kaggle/kaggle.json (Linux/Mac)")
        print("   - C:\\Users\\YOUR_USERNAME\\.kaggle\\kaggle.json (Windows)")
        print("4. Or manually download the dataset from:")
        print("   https://www.kaggle.com/datasets/shivamb/real-or-fake-fake-jobposting-prediction")
        print("   and place fake_job_postings.csv in data/raw/")
        return False

    try:
        print("Downloading dataset from Kaggle...")
        kaggle.api.competition_download_files(
            "real-or-fake-fake-jobposting-prediction",
            path=str(DATA_DIR),
            quiet=False
        )

        # Find the downloaded zip file
        zip_files = list(DATA_DIR.glob("*.zip"))
        if not zip_files:
            print("Error: No zip file found after download.")
            return False

        zip_path = zip_files[0]
        print(f"Extracting {zip_path}...")

        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(DATA_DIR)

        # Find the CSV file
        csv_files = list(DATA_DIR.glob("*.csv"))
        if not csv_files:
            print("Error: No CSV file found in extracted data.")
            return False

        # Rename/move to expected name
        source_csv = csv_files[0]
        if source_csv.name != "fake_job_postings.csv":
            target_path = DATA_DIR / "fake_job_postings.csv"
            shutil.move(str(source_csv), str(target_path))
            print(f"Renamed {source_csv.name} to fake_job_postings.csv")

        # Clean up zip file
        zip_path.unlink()

        print(f"Dataset downloaded successfully to: {KAGGLE_DATASET}")
        return True

    except Exception as e:
        print(f"Error downloading dataset: {e}")
        return False

def main():
    """Main function."""
    print("Fake Job Posting Detection - Dataset Download")
    print("=" * 50)

    # Ensure data/raw directory exists
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Check if dataset already exists
    if KAGGLE_DATASET.exists():
        print(f"Dataset already exists at: {KAGGLE_DATASET}")
        print("Skipping download. Delete the file if you want to re-download.")
        return

    # Try to download
    if download_kaggle_dataset():
        print("\nDataset ready! You can now run: python train.py")
    else:
        print("\nFailed to download dataset automatically.")
        print("Please download manually from:")
        print("https://www.kaggle.com/datasets/shivamb/real-or-fake-fake-jobposting-prediction")
        print("and place the CSV file as: data/raw/fake_job_postings.csv")

if __name__ == "__main__":
    main()