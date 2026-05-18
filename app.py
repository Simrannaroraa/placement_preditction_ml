import streamlit as st
import numpy as np
import pickle
import os

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Campus Placement Predictor",
    page_icon="🎓",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

/* Background */
.stApp {
    background: #0d0f14;
    color: #e8e6e1;
}

/* Hide default streamlit elements */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 2rem; padding-bottom: 3rem; }

/* Hero banner */
.hero {
    text-align: center;
    padding: 3rem 1rem 2rem;
    margin-bottom: 2rem;
}
.hero h1 {
    font-family: 'Syne', sans-serif;
    font-size: 3rem;
    font-weight: 800;
    letter-spacing: -0.03em;
    background: linear-gradient(135deg, #f5c842 0%, #ff6b35 50%, #e8e6e1 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin: 0;
}
.hero p {
    color: #888;
    font-size: 1.05rem;
    margin-top: 0.6rem;
    font-weight: 300;
}

/* Section headers */
.section-label {
    font-family: 'Syne', sans-serif;
    font-size: 0.7rem;
    font-weight: 700;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: #f5c842;
    margin-bottom: 0.8rem;
    margin-top: 2rem;
}

/* Card */
.card {
    background: #161a22;
    border: 1px solid #252a35;
    border-radius: 16px;
    padding: 1.5rem 1.8rem;
    margin-bottom: 1.2rem;
}

/* Result boxes */
.result-placed {
    background: linear-gradient(135deg, #1a2e1a, #0d1f0d);
    border: 1px solid #2e7d32;
    border-radius: 16px;
    padding: 2rem;
    text-align: center;
    margin-top: 1.5rem;
}
.result-notplaced {
    background: linear-gradient(135deg, #2e1a1a, #1f0d0d);
    border: 1px solid #7d3232;
    border-radius: 16px;
    padding: 2rem;
    text-align: center;
    margin-top: 1.5rem;
}
.result-emoji { font-size: 3rem; margin-bottom: 0.5rem; }
.result-title {
    font-family: 'Syne', sans-serif;
    font-size: 1.6rem;
    font-weight: 700;
    margin: 0.3rem 0;
}
.result-placed .result-title { color: #66bb6a; }
.result-notplaced .result-title { color: #ef5350; }
.result-subtitle { color: #aaa; font-size: 0.95rem; margin-top: 0.3rem; }
.salary-chip {
    display: inline-block;
    background: #f5c842;
    color: #0d0f14;
    font-family: 'Syne', sans-serif;
    font-weight: 700;
    font-size: 1.1rem;
    padding: 0.4rem 1.2rem;
    border-radius: 100px;
    margin-top: 1rem;
}

/* Divider */
.divider {
    border: none;
    border-top: 1px solid #252a35;
    margin: 2rem 0;
}

/* Streamlit widget overrides */
div[data-testid="stSelectbox"] > div, 
div[data-testid="stNumberInput"] input,
div[data-testid="stTextInput"] input,
div[data-testid="stSlider"] {
    background: #1e2330 !important;
    border-color: #2e3448 !important;
    color: #e8e6e1 !important;
    border-radius: 8px !important;
}

/* Button */
div[data-testid="stButton"] button {
    background: linear-gradient(135deg, #f5c842, #ff6b35) !important;
    color: #0d0f14 !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important;
    font-size: 1rem !important;
    padding: 0.7rem 2.5rem !important;
    border: none !important;
    border-radius: 100px !important;
    width: 100% !important;
    cursor: pointer !important;
    letter-spacing: 0.03em !important;
    transition: opacity 0.2s !important;
}
div[data-testid="stButton"] button:hover {
    opacity: 0.88 !important;
}

label { color: #b0b8cc !important; font-size: 0.9rem !important; }
</style>
""", unsafe_allow_html=True)

# ── Load models ────────────────────────────────────────────────────────────────
@st.cache_resource
def load_models():
    errors = []
    placement_model = None
    salary_model = None

    for path in ["model.pkl", os.path.join(os.path.dirname(__file__), "model.pkl"),
                 "/mnt/user-data/uploads/model.pkl"]:
        if os.path.exists(path):
            try:
                placement_model = pickle.load(open(path, "rb"))
                break
            except Exception as e:
                errors.append(str(e))

    for path in ["model1.pkl", os.path.join(os.path.dirname(__file__), "model1.pkl"),
                 "/mnt/user-data/uploads/model1.pkl"]:
        if os.path.exists(path):
            try:
                salary_model = pickle.load(open(path, "rb"))
                break
            except Exception as e:
                errors.append(str(e))

    return placement_model, salary_model, errors

placement_model, salary_model, load_errors = load_models()

# ── Hero ───────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <h1>Placement Predictor</h1>
  <p>Find out your placement chances & expected salary — powered by ML</p>
</div>
""", unsafe_allow_html=True)

# Model status warning
if not placement_model or not salary_model:
    st.warning(
        "⚠️ Model files (`model.pkl` / `model1.pkl`) not found next to this script. "
        "Place them in the same directory and restart.\n\n"
        "Running in **demo mode** — prediction buttons are disabled.",
        icon="⚠️"
    )
    demo_mode = True
else:
    demo_mode = False

# ── Form ───────────────────────────────────────────────────────────────────────
st.markdown('<div class="section-label">Student Details</div>', unsafe_allow_html=True)

col1, col2 = st.columns(2)
with col1:
    name = st.text_input("Your Name", placeholder="e.g. Arjun Sharma")
with col2:
    cgpa = st.number_input("CGPA (out of 10)", min_value=0.0, max_value=10.0,
                           value=7.5, step=0.1, format="%.1f")

st.markdown('<div class="section-label">Academic & Project Profile</div>', unsafe_allow_html=True)
col3, col4, col5 = st.columns(3)
with col3:
    projects = st.number_input("Major Projects", min_value=0, max_value=10, value=1)
with col4:
    mini_projects = st.number_input("Mini Projects", min_value=0, max_value=10, value=1)
with col5:
    workshops = st.number_input("Workshops / Certifications", min_value=0, max_value=20, value=2)

col6, col7 = st.columns(2)
with col6:
    tw_percentage = st.number_input("12th Percentage (%)", min_value=0.0, max_value=100.0,
                                    value=75.0, step=0.5)
with col7:
    te_percentage = st.number_input("10th Percentage (%)", min_value=0.0, max_value=100.0,
                                    value=80.0, step=0.5)

backlogs = st.number_input("Number of Backlogs", min_value=0, max_value=20, value=0)

st.markdown('<div class="section-label">Skills & Activities</div>', unsafe_allow_html=True)

skills_raw = st.text_input(
    "Technical Skills (comma-separated)",
    placeholder="e.g. Python, SQL, Machine Learning, Java",
    value="Python, SQL, Machine Learning"
)
num_skills = max(1, skills_raw.count(",") + 1) if skills_raw.strip() else 1

col8, col9, col10 = st.columns(3)
with col8:
    communication_skills = st.slider("Communication Skill Rating", 1.0, 5.0, 4.0, 0.1)
with col9:
    internship = st.selectbox("Internship Done?", ["Yes", "No"])
with col10:
    hackathon = st.selectbox("Participated in Hackathon?", ["Yes", "No"])

internship_enc = 1 if internship == "Yes" else 0
hackathon_enc  = 1 if hackathon  == "Yes" else 0

st.markdown("<hr class='divider'>", unsafe_allow_html=True)

# ── Predict ────────────────────────────────────────────────────────────────────
predict_clicked = st.button("🔍  Predict My Placement", disabled=demo_mode)

if predict_clicked and not demo_mode:
    arr = np.array([cgpa, projects, workshops, mini_projects, num_skills,
                    communication_skills, internship_enc, hackathon_enc,
                    tw_percentage, te_percentage, backlogs], dtype=float)

    output = placement_model.predict([arr])[0]
    placed = (output == "Placed")

    p_flag = 1.0 if placed else 0.0
    arr1 = np.array([cgpa, projects, workshops, mini_projects, num_skills,
                     communication_skills, internship_enc, hackathon_enc,
                     tw_percentage, te_percentage, backlogs, p_flag], dtype=float)
    salary_raw = salary_model.predict([arr1])[0]

    # Format salary with commas (Indian style)
    s = str(int(salary_raw))
    if len(s) == 6:
        salary_fmt = f"{s[0]},{s[1:3]},{s[3:]}"
    elif len(s) == 7:
        salary_fmt = f"{s[:2]},{s[2:4]},{s[4:]}"
    else:
        salary_fmt = f"{salary_raw:,}"

    display_name = name.strip() if name.strip() else "You"

    if placed:
        st.markdown(f"""
        <div class="result-placed">
          <div class="result-emoji">🎉</div>
          <div class="result-title">High Placement Chances!</div>
          <div class="result-subtitle">Congratulations, {display_name}! The model predicts you are likely to be placed.</div>
          <div class="salary-chip">Expected Salary: ₹{salary_fmt} / year</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="result-notplaced">
          <div class="result-emoji">📚</div>
          <div class="result-title">Keep Improving!</div>
          <div class="result-subtitle">Sorry, {display_name} — the model predicts lower placement chances right now. Focus on upskilling!</div>
        </div>
        """, unsafe_allow_html=True)

    # Feature summary
    with st.expander("📊 Input Summary"):
        col_a, col_b = st.columns(2)
        with col_a:
            st.metric("CGPA", f"{cgpa:.1f}")
            st.metric("Skills Count", num_skills)
            st.metric("Backlogs", backlogs)
            st.metric("12th %", f"{tw_percentage:.1f}%")
        with col_b:
            st.metric("Projects", projects)
            st.metric("Workshops", workshops)
            st.metric("Communication", f"{communication_skills:.1f}/5")
            st.metric("10th %", f"{te_percentage:.1f}%")

elif predict_clicked and demo_mode:
    st.error("Models not loaded. Please add model.pkl and model1.pkl to the app directory.")

# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="text-align:center; color:#444; font-size:0.8rem; margin-top:3rem;">
  Placement Predictor · Random Forest · 88.7% Accuracy
</div>
""", unsafe_allow_html=True)