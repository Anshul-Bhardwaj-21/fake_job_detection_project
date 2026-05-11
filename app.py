import json
from pathlib import Path
import joblib
import streamlit as st
import pandas as pd
from scipy import sparse

from src.config import MODEL_PATH, TFIDF_PATH, LABEL_ENCODERS_PATH, FEATURE_CONFIG_PATH, METRICS_PATH, REPORT_DIR, CATEGORICAL_COLUMNS, NUMERIC_COLUMNS
from src.preprocessing import build_user_record, prepare_dataframe
from src.utils import risk_level, warning_indicators

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

    if st.button("Analyze Job Posting", type="primary"):
        record = build_user_record(
            title=title,
            company_profile=company_profile,
            description=description,
            requirements=requirements,
            benefits=benefits,
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

        if hasattr(model, "predict_proba"):
            probability = float(model.predict_proba(X_combined)[0][1])
        else:
            probability = 0.70 if prediction == 1 else 0.30

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

elif page == "Model Metrics":
    st.subheader("Model Metrics")
    if METRICS_PATH.exists():
        metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
        st.write("Best Model:", metrics.get("best_model"))
        st.dataframe(pd.DataFrame(metrics.get("results", [])))
        st.write("Confusion Matrix:")
        st.write(metrics.get("confusion_matrix"))
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
