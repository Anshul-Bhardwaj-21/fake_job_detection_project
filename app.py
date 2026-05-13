import json
from pathlib import Path
import joblib
import streamlit as st
import pandas as pd
import numpy as np
from scipy import sparse
import requests
from bs4 import BeautifulSoup
import PyPDF2
import io

from src.config import MODEL_PATH, TFIDF_PATH, LABEL_ENCODERS_PATH, FEATURE_CONFIG_PATH, METRICS_PATH, REPORT_DIR, CATEGORICAL_COLUMNS, NUMERIC_COLUMNS, SUSPICIOUS_KEYWORDS
from src.preprocessing import build_user_record, prepare_dataframe
from src.utils import risk_level, warning_indicators

# Feedback storage
FEEDBACK_FILE = Path("data/feedback.json")

def estimate_fake_probability(model, features, prediction):
    """Return a usable fake-risk score for models with or without predict_proba."""
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(features)[0]
        classes = list(getattr(model, "classes_", [0, 1]))
        if 1 in classes:
            return float(probabilities[classes.index(1)])
        return float(probabilities[-1])

    if hasattr(model, "decision_function"):
        margin = float(model.decision_function(features)[0])
        margin = np.clip(margin, -20, 20)
        return float(1 / (1 + np.exp(-margin)))

    return 0.70 if prediction == 1 else 0.30

def save_feedback(data_row, prediction, feedback):
    """Save user feedback for reinforcement learning."""
    feedback_data = {
        "timestamp": str(pd.Timestamp.now()),
        "features": data_row.to_dict('records')[0],
        "model_prediction": int(prediction),
        "user_feedback": int(feedback),  # 1=accurate, 0=inaccurate
        "corrected_label": 1 - prediction if feedback == 0 else prediction  # Flip if user said inaccurate
    }

    # Load existing feedback
    if FEEDBACK_FILE.exists():
        try:
            with open(FEEDBACK_FILE, 'r') as f:
                feedback_list = json.load(f)
        except:
            feedback_list = []
    else:
        feedback_list = []

    feedback_list.append(feedback_data)

    # Save updated feedback
    FEEDBACK_FILE.parent.mkdir(exist_ok=True)
    with open(FEEDBACK_FILE, 'w') as f:
        json.dump(feedback_list, f, indent=2)

def extract_text_from_pdf(pdf_file):
    """Extract text content from uploaded PDF file."""
    try:
        pdf_reader = PyPDF2.PdfReader(io.BytesIO(pdf_file.read()))
        text = ""
        for page in pdf_reader.pages:
            page_text = page.extract_text() or ""
            text += page_text + "\n"
        return text.strip()
    except Exception as e:
        st.error(f"Error reading PDF: {str(e)}")
        return ""

def extract_text_from_url(url):
    """Extract text content from a job posting URL."""
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()

        soup = BeautifulSoup(response.content, 'html.parser')

        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.decompose()

        # Get text content
        text = soup.get_text()

        # Clean up whitespace
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = ' '.join(chunk for chunk in chunks if chunk)

        return text[:10000]  # Limit to first 10k characters

    except Exception as e:
        st.error(f"Error fetching URL: {str(e)}")
        return ""

def analyze_document(text, title="Document Analysis"):
    """Analyze extracted text using the ML model."""
    if not text.strip():
        st.warning("No text content found to analyze.")
        return

    # Create a record with the extracted text
    record = build_user_record(
        title=title,
        company_profile="",
        description=text,
        requirements="",
        benefits="",
    )

    prepared = prepare_dataframe(record)

    # Build features
    X_text = tfidf.transform(prepared["clean_text"])
    X_cat = ohe.transform(prepared[CATEGORICAL_COLUMNS])
    X_num = scaler.transform(prepared[NUMERIC_COLUMNS])
    X_combined = sparse.hstack([X_text, X_cat, X_num])

    prediction = int(model.predict(X_combined)[0])
    probability = estimate_fake_probability(model, X_combined, prediction)

    suspicious_terms = [kw for kw in SUSPICIOUS_KEYWORDS if kw in text.lower()]
    heuristic_flag = (
        prepared.iloc[0].get("suspicious_keyword_count", 0) >= 1
        or prepared.iloc[0].get("profile_missing", 0) == 1
        or prepared.iloc[0].get("salary_missing", 0) == 1
    )

    score = round(probability * 100, 2)
    level = risk_level(score)
    warnings = warning_indicators(prepared)

    st.write(f"### Analysis Results for: {title}")

    c1, c2, c3 = st.columns(3)
    c1.metric("Prediction", "Fake / Suspicious" if prediction == 1 else "Genuine")
    c2.metric("Risk Score", f"{score}/100")
    c3.metric("Risk Level", level)

    if prediction == 1:
        st.error("⚠️ This document appears suspicious. Verify the source carefully!")
    else:
        st.success("✅ Model prediction is Genuine, but always verify manually.")

    if heuristic_flag and prediction == 0:
        st.warning("⚠️ The model predicts Genuine, but suspicious content patterns were detected. Please verify manually.")

    if warnings:
        st.write("### Warning Indicators Found:")
        for item in warnings:
            st.write(f"- {item}")

    if suspicious_terms:
        st.write("### Suspicious Keywords Detected")
        for term in suspicious_terms:
            st.write(f"- {term}")

    # Show extracted text preview
    with st.expander("📄 View Extracted Text"):
        st.text_area("Document Content", text[:2000] + "..." if len(text) > 2000 else text, height=200)

    return prepared, prediction

def retrain_with_feedback():
    """Retrain model with accumulated user feedback using online learning."""
    if not FEEDBACK_FILE.exists():
        st.warning("No feedback data available for retraining.")
        return

    try:
        with open(FEEDBACK_FILE, 'r') as f:
            feedback_data = json.load(f)

        if len(feedback_data) < 10:  # Need minimum feedback for meaningful retraining
            st.warning(f"Need at least 10 feedback samples. Currently have {len(feedback_data)}.")
            return

        # Load original training data
        from src.utils import get_dataset_path
        from src.preprocessing import load_dataset
        import pandas as pd

        dataset_path = get_dataset_path()
        original_data = load_dataset(dataset_path)
        original_prepared = prepare_dataframe(original_data)

        # Convert feedback to DataFrame
        feedback_df = pd.DataFrame([f['features'] for f in feedback_data])
        feedback_labels = pd.Series([f['corrected_label'] for f in feedback_data])

        # Combine original data with feedback
        combined_data = pd.concat([original_prepared, feedback_df], ignore_index=True)
        combined_labels = pd.concat([original_prepared['fraudulent'], feedback_labels], ignore_index=True)

        # Retrain model with combined data
        from src.features import build_features_separate
        from sklearn.linear_model import LogisticRegression

        X_combined, tfidf_new, ohe_new, scaler_new = build_features_separate(combined_data)
        y_combined = combined_labels

        # Use online learning - partial_fit for incremental learning
        if hasattr(model, 'partial_fit'):
            # For models that support partial_fit
            model.partial_fit(X_combined, y_combined)
        else:
            # Retrain from scratch with combined data
            model.fit(X_combined, y_combined)

        # Save updated model
        joblib.dump(model, MODEL_PATH)
        joblib.dump(tfidf_new, TFIDF_PATH)
        joblib.dump(ohe_new, LABEL_ENCODERS_PATH)
        joblib.dump(scaler_new, FEATURE_CONFIG_PATH)

        st.success(f"Model retrained with {len(feedback_data)} feedback samples! AI is learning... 🧠")

    except Exception as e:
        st.error(f"Retraining failed: {str(e)}")

st.set_page_config(page_title="Fake Job Posting Detection", page_icon="🛡️", layout="wide")

st.title("🛡️ Fake Job Posting Detection")
st.caption("Data Mining and Machine Learning Mini Project")

if not MODEL_PATH.exists():
    st.error("Model not found. Run `python train.py` first.")
    st.stop()

# Load model and components
model = joblib.load(MODEL_PATH)
tfidf = joblib.load(TFIDF_PATH)
ohe = joblib.load(LABEL_ENCODERS_PATH)
scaler = joblib.load(FEATURE_CONFIG_PATH)

with st.sidebar:
    st.header("Project Modules")
    page = st.radio("Go to", ["Predict Job Risk", "Model Metrics", "Generated Reports", "About Project"])

if page == "Predict Job Risk":
    st.subheader("Analyze a Job Posting")

    col1, col2 = st.columns(2)

    with col1:
        title = st.text_input("Job Title", "Online Data Entry Operator")
        employment_type = st.selectbox("Employment Type", ["Unknown", "Full-time", "Part-time", "Contract", "Internship", "Temporary"])
        required_experience = st.selectbox("Required Experience", ["Unknown", "Internship", "Entry level", "Associate", "Mid-Senior level", "Not Applicable"])
        required_education = st.selectbox("Required Education", ["Unknown", "High School", "Bachelor's Degree", "Master's Degree", "Not Specified"])
        industry = st.text_input("Industry", "Internet")
        function = st.text_input("Function", "Administrative")

    with col2:
        telecommuting = st.checkbox("Remote / Work From Home", value=True)
        has_company_logo = st.checkbox("Company Logo Present", value=False)
        has_questions = st.checkbox("Screening Questions Present", value=False)

    company_profile = st.text_area("Company Profile", "")
    description = st.text_area("Job Description", "Work from home and earn Rs. 60,000 per month. No experience required. Immediate joining. Registration fee required for training material.", height=120)
    requirements = st.text_area("Requirements", "Mobile phone and internet connection", height=80)
    benefits = st.text_area("Benefits", "Daily payment and guaranteed income", height=80)

    # New input fields
    col3, col4 = st.columns(2)
    with col3:
        appointment_letter = st.text_area("Appointment Letter (if available)", "", height=100)
    with col4:
        job_offer_ad = st.text_area("Job Offer Advertisement (if available)", "", height=100)

    # Search functionality (Coming Soon)
    if st.button("🔍 Search Similar Jobs", help="Coming Soon - Advanced job search functionality"):
        st.info("🔧 Search functionality coming soon! This will help you find similar job postings and compare risks.")

    if st.button("Analyze Job Posting", type="primary"):
        record = build_user_record(
            title=title,
            company_profile=company_profile,
            description=description,
            requirements=requirements,
            benefits=benefits,
            extra_text=f"{appointment_letter} {job_offer_ad}".strip(),
            employment_type=employment_type,
            required_experience=required_experience,
            required_education=required_education,
            industry=industry,
            function=function,
            telecommuting=int(telecommuting),
            has_company_logo=int(has_company_logo),
            has_questions=int(has_questions),
        )
        prepared = prepare_dataframe(record)

        # Build features separately
        X_text = tfidf.transform(prepared["clean_text"])
        X_cat = ohe.transform(prepared[CATEGORICAL_COLUMNS])
        X_num = scaler.transform(prepared[NUMERIC_COLUMNS])
        X_combined = sparse.hstack([X_text, X_cat, X_num])

        prediction = int(model.predict(X_combined)[0])
        probability = estimate_fake_probability(model, X_combined, prediction)

        score = round(probability * 100, 2)
        level = risk_level(score)
        warnings = warning_indicators(prepared)

        c1, c2, c3 = st.columns(3)
        c1.metric("Prediction", "Fake / Suspicious" if prediction == 1 else "Genuine")
        c2.metric("Risk Score", f"{score}/100")
        c3.metric("Risk Level", level)

        if prediction == 1:
            st.error("This job posting looks suspicious. Verify carefully before applying.")
        else:
            st.success("This job posting looks relatively safer, but manual verification is still recommended.")

        st.write("### Warning Indicators")
        for item in warnings:
            st.write(f"- {item}")

        st.info("This is a risk-based ML prediction, not legal proof. Always verify company website, email domain, and recruiter identity.")

        # Reinforcement Learning Feedback System
        st.write("---")
        st.write("### 🤖 Help Improve Our AI")
        st.write("Was this prediction accurate? Your feedback helps our model learn and improve!")

        col_feedback1, col_feedback2, col_feedback3 = st.columns([1, 1, 2])
        with col_feedback1:
            if st.button("👍 Accurate", key="thumbs_up"):
                save_feedback(prepared, prediction, 1)  # 1 = positive feedback
                st.success("Thank you! Your feedback helps improve our AI.")

        with col_feedback2:
            if st.button("👎 Inaccurate", key="thumbs_down"):
                save_feedback(prepared, prediction, 0)  # 0 = negative feedback
                st.info("Thank you! We'll use this to improve our model.")

        with col_feedback3:
            if st.button("🔄 Retrain Model", help="Retrain the model with collected feedback"):
                retrain_with_feedback()
                st.success("Model retrained with user feedback! 🎉")

    # Advanced Analysis Section
    st.header("🔍 Advanced Analysis")
    st.write("Upload job documents or analyze job postings from URLs to verify authenticity.")

    # File Upload Section
    st.subheader("📎 Upload Job Document")
    uploaded_file = st.file_uploader("Choose a PDF file", type=['pdf'])

    if uploaded_file is not None:
        if st.button("Analyze Uploaded Document", key="analyze_file"):
            with st.spinner("Extracting text from PDF..."):
                extracted_text = extract_text_from_pdf(uploaded_file)

            if extracted_text:
                analyze_document(extracted_text, f"Uploaded: {uploaded_file.name}")

    # URL Analysis Section
    st.subheader("🌐 Analyze Job Posting URL")
    url_input = st.text_input("Enter job posting URL", placeholder="https://example.com/job-posting")

    if url_input and st.button("Analyze URL", key="analyze_url"):
        if not url_input.startswith(('http://', 'https://')):
            st.error("Please enter a valid URL starting with http:// or https://")
        else:
            with st.spinner("Fetching and analyzing content..."):
                extracted_text = extract_text_from_url(url_input)

            if extracted_text:
                analyze_document(extracted_text, f"URL: {url_input}")

    st.markdown("---")

elif page == "Model Metrics":
    st.subheader("Model Metrics")
    if METRICS_PATH.exists():
        metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
        st.write("Best Model:", metrics.get("best_model"))
        st.write("### Model Comparison")
        st.dataframe(pd.DataFrame(metrics.get("results", [])))

        st.write("### Confusion Matrix")
        cm = metrics.get("confusion_matrix", [[0, 0], [0, 0]])
        cm_df = pd.DataFrame(cm, index=["Actual Real", "Actual Fake"], columns=["Predicted Real", "Predicted Fake"])
        st.dataframe(cm_df)
        st.write("- Top-left cell = Real jobs correctly predicted as Real")
        st.write("- Bottom-right cell = Fake jobs correctly predicted as Fake")
        st.write("- Real class is much larger, so this is expected for an imbalanced dataset.")

        st.write("### Classification Report")
        report = metrics.get("classification_report", {})
        if report:
            report_df = pd.DataFrame(report).transpose()
            st.dataframe(report_df)
    else:
        st.warning("Metrics file not found. Run training first.")

elif page == "Generated Reports":
    st.subheader("Generated Charts and Analysis")
    images = [
        "class_distribution.png",
        "model_comparison.png",
        "confusion_matrix.png",
        "elbow_method.png",
        "cluster_distribution.png",
    ]
    for image in images:
        path = REPORT_DIR / image
        if path.exists():
            st.image(str(path), caption=image, use_container_width=True)
        else:
            st.warning(f"Missing: {image}")

    rules_path = REPORT_DIR / "association_rules.csv"
    if rules_path.exists():
        try:
            rules_df = pd.read_csv(rules_path)
            if not rules_df.empty:
                st.write("### Association Rules")
                st.dataframe(rules_df)
            else:
                st.write("### Association Rules")
                st.info("No association rules were generated with current parameters. Try adjusting min_support or min_confidence in the training script.")
        except pd.errors.EmptyDataError:
            st.write("### Association Rules")
            st.info("No association rules were generated with current parameters. Try adjusting min_support or min_confidence in the training script.")

elif page == "About Project":
    st.subheader("About")
    st.write("""
    This mini project detects fake job postings using Data Mining and Machine Learning.
    It covers preprocessing, TF-IDF, classification, risk score estimation, Apriori-style association rules,
    K-Means clustering, and evaluation metrics.

    **Created by: Anshul Bhardwaj**
    """)
