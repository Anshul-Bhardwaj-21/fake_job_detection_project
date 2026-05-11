# Fake Job Posting Detection using Data Mining and Machine Learning

A robust mini project for **Data Mining and Machine Learning (Experiment 11: Mini Project)**.

This project detects whether a job posting is genuine or fake using text preprocessing, TF-IDF vectorization, machine learning classification, risk score estimation, association rule mining, clustering, and Streamlit-based user interface.

## Features

- Fake vs real job posting classification
- Fraud probability and risk score generation
- Suspicious keyword detection
- TF-IDF based text feature extraction
- Multiple model comparison (Naive Bayes, Logistic Regression, Decision Tree, Random Forest, SVM)
- Confusion matrix and evaluation metrics
- Association rule mining using Apriori algorithm
- K-Means clustering analysis
- Streamlit UI for demo
- Kaggle dataset support with automatic download
- Sample dataset included as fallback

## Dataset

**Real or Fake: Fake Job Posting Prediction**

Kaggle dataset: `shivamb/real-or-fake-fake-jobposting-prediction`

Expected file: `fake_job_postings.csv`

## Project Structure

```
fake-job-posting-detection/
├── app.py                          # Streamlit web application
├── train.py                        # Main training script
├── requirements.txt                # Python dependencies
├── README.md                       # Project documentation
├── scripts/
│   └── download_dataset.py         # Kaggle dataset downloader
├── data/
│   ├── raw/
│   │   ├── fake_job_postings.csv   # Kaggle dataset (auto-downloaded)
│   │   └── sample_fake_job_postings.csv  # Fallback sample data
│   └── processed/
│       └── processed_jobs.csv      # Cleaned processed data
├── models/
│   ├── fake_job_model.pkl          # Trained ML model
│   ├── tfidf_vectorizer.pkl        # TF-IDF vectorizer
│   ├── label_encoders.pkl          # Categorical encoders
│   ├── feature_config.pkl          # Feature scaler
│   └── metrics.json                # Model evaluation metrics
├── reports/
│   └── figures/
│       ├── class_distribution.png
│       ├── confusion_matrix.png
│       ├── model_comparison.png
│       ├── clustering_elbow.png
│       ├── association_rules.csv
│       └── cluster_summary.csv
└── src/
    ├── __init__.py
    ├── config.py                   # Configuration and paths
    ├── preprocessing.py            # Data cleaning and feature engineering
    ├── features.py                 # Feature extraction pipeline
    ├── model_training.py           # Model training and evaluation
    ├── association_rules.py        # Apriori association rules
    ├── clustering.py               # K-Means clustering
    └── utils.py                    # Utility functions
```

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Download Dataset

**Option A: Automatic Download (Recommended)**

```bash
python scripts/download_dataset.py
```

This script will:
- Check for Kaggle API credentials
- Download the dataset from Kaggle
- Extract and place as `data/raw/fake_job_postings.csv`

**Option B: Manual Download**

If automatic download fails:
1. Go to [Kaggle Dataset](https://www.kaggle.com/datasets/shivamb/real-or-fake-fake-jobposting-prediction)
2. Download `fake_job_postings.csv`
3. Place it in `data/raw/fake_job_postings.csv`

### 3. Train the Model

```bash
python train.py
```

This creates all model files and evaluation reports.

### 4. Run Streamlit App

```bash
streamlit run app.py
```

## Dataset Options

### Primary: Kaggle Dataset
- **File**: `data/raw/fake_job_postings.csv`
- **Source**: Kaggle "Real or Fake: Fake Job Posting Prediction"
- **Size**: ~18,000 job postings
- **Features**: Text fields, categorical data, binary indicators

### Fallback: Sample Dataset
- **File**: `data/raw/sample_fake_job_postings.csv`
- **Usage**: Automatically used if Kaggle dataset not found
- **Purpose**: Demo and testing

## Data Preprocessing

### Text Processing
- Combine title, company_profile, description, requirements, benefits
- Lowercase, remove URLs, punctuation, extra spaces

### Feature Engineering
- **Text Features**: TF-IDF vectorization (max_features=5000, ngram_range=(1,2))
- **Numeric Features**:
  - description_length, requirements_length, company_profile_length
  - suspicious_keyword_count
  - profile_missing, salary_missing
  - telecommuting, has_company_logo, has_questions

### Suspicious Keywords
- registration fee, joining fee, urgent hiring
- no experience, daily payment, guaranteed income
- easy money, work from home, limited seats
- training fee, pay first, no interview

## Machine Learning Models

Trains and compares:
1. **Complement Naive Bayes** - Good for imbalanced text data
2. **Logistic Regression** - Interpretable, supports predict_proba
3. **Decision Tree** - Rule-based, handles mixed features
4. **Random Forest** - Ensemble, robust to overfitting
5. **Linear SVM** - Effective for high-dimensional data

**Selection Criteria**: Best F1-score and recall for fraudulent class (prioritize catching fakes).

## Evaluation Metrics

For each model:
- Accuracy, Precision, Recall, F1-Score
- Confusion Matrix
- Classification Report

Saved to `models/metrics.json`

## Association Rules

Uses mlxtend Apriori algorithm to find patterns like:
- "Missing Logo, Suspicious Keywords Present → Fraudulent"
- Minimum support: 0.01, Minimum confidence: 0.5

Saved to `reports/figures/association_rules.csv`

## Clustering

K-Means clustering on TF-IDF features:
- Uses MiniBatchKMeans for efficiency
- Elbow method plot saved as `clustering_elbow.png`
- Cluster summaries with fraud ratios

## How to Demo to Faculty

1. **Dataset Loading**:
   - Show `python scripts/download_dataset.py` output
   - Display dataset info from training logs

2. **Data Preprocessing**:
   - Show `data/processed/processed_jobs.csv`
   - Explain text cleaning and feature engineering

3. **Model Training**:
   - Run `python train.py`
   - Show model comparison chart
   - Display best model metrics

4. **Evaluation**:
   - Show confusion matrix
   - Explain F1-score and recall priority

5. **Association Rules**:
   - Display `reports/figures/association_rules.csv`
   - Explain fraud patterns

6. **Clustering**:
   - Show elbow method plot
   - Display cluster summaries

7. **Streamlit Demo**:
   - Input suspicious job posting
   - Show prediction, risk score, warning indicators

## Important Notes

- **No Fake Results**: All metrics come from actual model training
- **Imbalanced Data**: Uses class weighting and appropriate metrics
- **Sparse Matrices**: Efficient handling of high-dimensional text features
- **Error Handling**: Graceful fallbacks and clear error messages
- **College Project**: Suitable for DMML mini project demonstration

## Security Disclaimer

This is a decision-support system providing risk-based predictions, not legal proof. Users should always verify company identity manually before applying or sharing sensitive information.
models/model_metrics.json
reports/figures/confusion_matrix.png
reports/figures/model_comparison.png
reports/figures/elbow_method.png
reports/figures/cluster_distribution.png
reports/figures/class_distribution.png
reports/figures/association_rules.csv
```

## Run Streamlit App

```bash
streamlit run app.py
```

## How to Demo to Faculty

1. Open Streamlit app.
2. Paste a suspicious job post:

```text
Work from home and earn Rs. 60,000 per month. No experience required. Immediate joining. Registration fee required.
```

3. Show output:
   - Fake/Suspicious prediction
   - Risk score
   - Warning indicators

4. Open generated charts from `reports/figures/`.
5. Explain syllabus mapping:
   - Section A: preprocessing, attributes, Apriori, association rules
   - Section B: classification, clustering, evaluation, trends

## Important Note

This project is a decision-support system. It gives risk-based prediction, not legal proof. Users should verify company identity manually before applying or sharing sensitive information.
