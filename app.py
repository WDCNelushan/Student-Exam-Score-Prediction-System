import os
import streamlit as st
import numpy as np
import joblib
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(page_title="Student Exam Score Predictor", layout="wide")

# Load trained model
BASE_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE_DIR, "best_model.pkl")
model = None
if os.path.exists(MODEL_PATH):
    try:
        model = joblib.load(MODEL_PATH)
    except Exception as e:
        st.error(f"Failed to load model: {e}")
else:
    st.warning(f"Model file not found at {MODEL_PATH}. Predictions will be disabled.")

st.markdown(
    """
    <style>
    :root { color-scheme: dark; }
    .stApp { background: #030416; color: #e6eef8; }
    header, footer, #MainMenu, [data-testid="collapsedControl"] { visibility: hidden !important; height: 0 !important; padding: 0 !important; margin: 0 !important; }
    .stApp .main .block-container { max-width: 1200px !important; margin: 0 auto !important; padding: 0.5rem 2rem 1.75rem !important; background: linear-gradient(180deg,#05081a,#081024) !important; }
    body { overflow-x: hidden !important; }
    .page-title { color: #f8fafc; font-size: 2.2rem; font-weight: 800; margin: 0; letter-spacing: -0.02em; line-height: 1.05; }
    .page-subtitle { color: #b9c6d8; margin: 0.55rem 0 0; font-size: 0.92rem; max-width: 980px; white-space: normal; line-height: 1.45; }
    .hero { background: rgba(9, 15, 34, 0.9); border-radius: 22px; padding: 1.25rem 1.5rem; border: 1px solid rgba(56, 189, 248, 0.14); box-shadow: 0 14px 40px rgba(0, 0, 0, 0.22); margin: 0 0 1rem; }
    .section-card { background: rgba(12, 19, 37, 0.92); border-radius: 22px; padding: 1.25rem 1.4rem; border: 1px solid rgba(113, 105, 255, 0.18); box-shadow: 0 14px 36px rgba(0, 0, 0, 0.2); margin-bottom: 1.25rem; }
    .section-card h2 { margin: 0 0 0.65rem 0; font-size: 1.8rem; color: #ffffff; }
    .section-card p { color: #b9c6d8; margin: 0 0 1rem 0; font-size: 0.96rem; line-height: 1.55; }
    .card { background: linear-gradient(145deg, #111128 0%, #0d0d22 100%); border-radius: 20px; padding: 1.5rem; border: 1px solid rgba(99, 102, 241, 0.12); box-shadow: 0 8px 22px rgba(0, 0, 0, 0.25), 0 0 40px rgba(99, 102, 241, 0.04), inset 0 1px 0 rgba(255, 255, 255, 0.03); height: 100%; }
    .card h2 { color: #ffffff; font-size: 1.35rem; margin-bottom: 0.3rem; font-weight: 700; }
    .card p { color: #7a8ba8; margin-top: 0; font-size: 0.92rem; line-height: 1.5; }
    .habit-label { color: #e6eef8; font-weight: 600; font-size: 1.1rem; margin: 0 0 0.35rem 0; }
    .stSlider span { color: #66e0ff !important; }
    .stSlider [data-baseweb="slider"] div[role="slider"] { background: #6366f1 !important; }
    .stButton>button { background: linear-gradient(135deg, #7c3aed 0%, #06b6d4 100%) !important; color: #fff !important; border-radius: 999px !important; padding: 0.9rem 2rem !important; font-weight: 600 !important; font-size: 0.95rem !important; border: none !important; box-shadow: 0 8px 32px rgba(124, 58, 237, 0.3), 0 4px 12px rgba(6, 182, 212, 0.2) !important; transition: all 0.3s ease !important; width: auto !important; }
    .stButton>button:hover { transform: translateY(-2px) !important; box-shadow: 0 12px 40px rgba(124, 58, 237, 0.4), 0 6px 16px rgba(6, 182, 212, 0.3) !important; }
    .result-box { background: linear-gradient(180deg, rgba(15, 19, 36, 0.95) 0%, rgba(8, 16, 31, 0.98) 100%); border-radius: 22px; padding: 1.75rem 1.5rem; text-align: center; min-height: 220px; display: flex; flex-direction: column; align-items: center; justify-content: center; border: 1px solid rgba(56, 189, 248, 0.32); box-shadow: 0 0 28px rgba(56, 189, 248, 0.15), inset 0 0 14px rgba(255, 255, 255, 0.04); margin-top: 1.5rem; }
    .result-title { color: #a0c7f3; font-size: 0.8rem; letter-spacing: 0.25em; text-transform: uppercase; margin-bottom: 0.9rem; font-weight: 700; }
    .result-value { font-weight: 900; font-size: 48px !important; line-height: 1; background: linear-gradient(135deg, #38bdf8 0%, #22d3ee 45%, #818cf8 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; text-shadow: 0 0 18px rgba(56, 189, 248, 0.65); filter: drop-shadow(0 0 24px rgba(56, 189, 248, 0.4)); }
    .result-value-placeholder { font-weight: 700; font-size: 44px !important; color: #3a3a5c; line-height: 1; }
    .small-note { color: #7a8ba8; margin-top: 1rem; font-size: 0.88rem; }
    .divider { height: 1px; background: linear-gradient(90deg, transparent, rgba(99,102,241,0.3), transparent); margin: 1.5rem 0; }
    @media (max-width: 1024px) {
        .page-title { font-size: 2rem; }
        .page-subtitle { font-size: 0.92rem; max-width: 100%; }
        .hero { padding: 1.25rem 1.5rem; }
        .section-card { padding: 1rem 1.25rem; }
        .section-card h2 { font-size: 1.6rem; }
        .card { padding: 1.25rem; }
        .result-box { padding: 1.5rem 1.25rem; min-height: 200px; }
        .result-value { font-size: 42px !important; }
        .result-value-placeholder { font-size: 38px !important; }
    }
    @media (max-width: 720px) {
        .page-title { font-size: 1.8rem; }
        .page-subtitle { font-size: 0.9rem; }
        .hero { padding: 1rem 1rem; }
        .section-card { padding: 0.9rem 1rem; }
        .section-card h2 { font-size: 1.4rem; }
        .section-card p { font-size: 0.88rem; }
        .card { padding: 1rem; }
        .result-box { min-height: 180px; }
        .result-value { font-size: 36px !important; }
        .result-value-placeholder { font-size: 32px !important; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Header
st.markdown(
    """
    <div class='hero'>
        <h1 class='page-title'>Student Exam Score Prediction System</h1>
        <p class='page-subtitle'>Analyze study habits and predict exam performance using machine learning. Adjust the parameters and get instant predictions.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# Layout
left_col, right_col = st.columns([2, 1], gap="large")

with left_col:
    st.markdown("<div class='section-card'><h2>📚 Student Habits</h2><p>Use the sliders below to describe the study routine and lifestyle habits.</p></div>", unsafe_allow_html=True)

    st.markdown("<p class='habit-label'>Study Hours per day</p>", unsafe_allow_html=True)
    study_hours = st.slider("Study Hours per day", 0.0, 12.0, 2.0, step=0.1, format="%.2f", label_visibility="collapsed")

    st.markdown("<p class='habit-label'>Attendance Percentage</p>", unsafe_allow_html=True)
    attendance = st.slider("Attendance Percentage", 0.0, 100.0, 80.0, label_visibility="collapsed")

    st.markdown("<p class='habit-label'>Sleep Hours per day</p>", unsafe_allow_html=True)
    sleep_hours = st.slider("Sleep Hours per day", 0.0, 12.0, 7.0, step=0.1, format="%.2f", label_visibility="collapsed")

    st.markdown("<p class='habit-label'>Mental Health Score (1-10)</p>", unsafe_allow_html=True)
    mental_health = st.slider("Mental Health Score (1-10)", 1, 10, 7, label_visibility="collapsed")

    st.markdown("<p class='habit-label'>Social Media Usage (hours per day)</p>", unsafe_allow_html=True)
    social_media = st.slider("Social Media Usage (hours per day)", 0.0, 12.0, 2.0, step=0.1, format="%.2f", label_visibility="collapsed")

with right_col:
    st.markdown("<div class='section-card'><h2>🎯 Prediction Score</h2><p>Press the button to calculate your predicted exam score based on the habits entered.</p></div>", unsafe_allow_html=True)

    predicted_score_text = "Adjust the sliders and press predict"
    prediction_display = "—"
    show_result = False

    if st.button("Predict Exam Score"):
        if model is not None:
            input_data = np.array([[study_hours, attendance, sleep_hours, mental_health, social_media]])
            predicted_score = model.predict(input_data)[0]
            predicted_score = max(0, min(100, predicted_score))
            prediction_display = f"{predicted_score:.2f}"
            predicted_score_text = "Predicted Exam Score"
            show_result = True
        else:
            predicted_score_text = "Model not loaded"
            prediction_display = "N/A"

    if show_result:
        st.markdown(
            f"""
            <div class='result-box'>
                <p class='result-title'>Result</p>
                <p class='result-value'>{prediction_display}</p>
                <p class='small-note'>{predicted_score_text}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"""
            <div class='result-box'>
                <p class='result-title'>Result</p>
                <p class='result-value-placeholder'>{prediction_display}</p>
                <p class='small-note'>{predicted_score_text}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
