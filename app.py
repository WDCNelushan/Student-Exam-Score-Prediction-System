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
    html, body, .stApp, .main, .block-container { width: 100% !important; max-width: 100% !important; min-width: 0 !important; margin: 0 auto !important; padding: 0 !important; overflow-x: hidden !important; box-sizing: border-box !important; }
    .stApp { background: #030416; color: #e6eef8; }
    header, footer, #MainMenu, [data-testid="collapsedControl"] { visibility: hidden !important; height: 0 !important; padding: 0 !important; margin: 0 !important; }
    .block-container { padding: 1.5rem 2rem 2.75rem; background: linear-gradient(180deg,#05081a,#081024); }
    .page-title { color: #f8fafc; font-size: 3rem; font-weight: 800; margin: 0; letter-spacing: -0.02em; }
    .page-subtitle { color: #b9c6d8; margin: 0.75rem 0 0; font-size: 1.05rem; max-width: 900px; white-space: normal; line-height: 1.7; }
    .hero { background: rgba(9, 15, 34, 0.9); border-radius: 26px; padding: 2.25rem 2.5rem; border: 1px solid rgba(56, 189, 248, 0.14); box-shadow: 0 28px 90px rgba(0, 0, 0, 0.28); margin-bottom: 2rem; }
    .section-card { background: rgba(12, 19, 37, 0.92); border-radius: 24px; padding: 1.75rem 1.85rem; border: 1px solid rgba(113, 105, 255, 0.18); box-shadow: 0 18px 50px rgba(0, 0, 0, 0.25); margin-bottom: 1.5rem; }
    .section-card h2 { margin: 0 0 0.75rem 0; font-size: 2.75rem; color: #ffffff; }
    .section-card p { color: #b9c6d8; margin: 0 0 1.25rem 0; font-size: 1rem; line-height: 1.6; }
    .card { background: linear-gradient(145deg, #111128 0%, #0d0d22 100%); border-radius: 20px; padding: 2rem; border: 1px solid rgba(99, 102, 241, 0.15); box-shadow: 0 4px 24px rgba(0, 0, 0, 0.4), 0 0 60px rgba(99, 102, 241, 0.03), inset 0 1px 0 rgba(255, 255, 255, 0.03); height: 100%; }
    .card h2 { color: #ffffff; font-size: 1.5rem; margin-bottom: 0.3rem; font-weight: 700; }
    .card p { color: #7a8ba8; margin-top: 0; font-size: 0.92rem; line-height: 1.5; }
    .habit-label { color: #e6eef8; font-weight: 600; font-size: 1.1rem; margin: 0 0 0.35rem 0; }
    .stSlider span { color: #66e0ff !important; }
    .stSlider [data-baseweb="slider"] div[role="slider"] { background: #6366f1 !important; }
    .stButton>button { background: linear-gradient(135deg, #7c3aed 0%, #06b6d4 100%) !important; color: #fff !important; border-radius: 999px !important; padding: 0.9rem 2rem !important; font-weight: 600 !important; font-size: 0.95rem !important; border: none !important; box-shadow: 0 8px 32px rgba(124, 58, 237, 0.3), 0 4px 12px rgba(6, 182, 212, 0.2) !important; transition: all 0.3s ease !important; width: auto !important; }
    .stButton>button:hover { transform: translateY(-2px) !important; box-shadow: 0 12px 40px rgba(124, 58, 237, 0.4), 0 6px 16px rgba(6, 182, 212, 0.3) !important; }
    .result-box { background: linear-gradient(180deg, rgba(15, 19, 36, 0.95) 0%, rgba(8, 16, 31, 0.98) 100%); border-radius: 24px; padding: 3rem 2rem; text-align: center; min-height: 260px; display: flex; flex-direction: column; align-items: center; justify-content: center; border: 1px solid rgba(56, 189, 248, 0.4); box-shadow: 0 0 75px rgba(56, 189, 248, 0.18), inset 0 0 28px rgba(255, 255, 255, 0.05); margin-top: 1.5rem; }
    .result-title { color: #a0c7f3; font-size: 0.95rem; letter-spacing: 0.3em; text-transform: uppercase; margin-bottom: 1rem; font-weight: 700; }
    .result-value { font-weight: 900; font-size: 60px !important; line-height: 0.9; background: linear-gradient(135deg, #38bdf8 0%, #22d3ee 45%, #818cf8 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; text-shadow: 0 0 24px rgba(56, 189, 248, 0.8); filter: drop-shadow(0 0 36px rgba(56, 189, 248, 0.55)); }
    .result-value-placeholder { font-weight: 700; font-size: 56px !important; color: #3a3a5c; line-height: 1; }
    .small-note { color: #7a8ba8; margin-top: 1rem; font-size: 0.88rem; }
    .divider { height: 1px; background: linear-gradient(90deg, transparent, rgba(99,102,241,0.3), transparent); margin: 1.5rem 0; }
    @media (max-width: 800px) { .page-title { font-size: 2rem; } .result-box { min-height: 160px; } }
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
