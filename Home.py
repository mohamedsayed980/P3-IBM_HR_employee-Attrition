# ─────────────────────────────────────────────
#  Repo_3 — HR Employee Attrition Analysis
#  Home.py  |  Multipage Launcher
#  Author : Mohamed · M3
# ─────────────────────────────────────────────
# =============================================================================
## path = streamlit run "E:\FINAL PROJECTS\P3-IBM_HR_employee Attrition\Home.py"
# ================================================================#

import streamlit as st
import pathlib

# ── Page config ───────────────────────────────
st.set_page_config(
    page_title  = "HR Attrition · M3",
    page_icon   = "🧑‍💼",
    layout      = "wide",
    initial_sidebar_state = "expanded",
)

# ── Logo ──────────────────────────────────────
LOGO = pathlib.Path(__file__).parent / "M3_logo.png"

# ── CLR palette ───────────────────────────────
CLR = {
    "primary"   : "#1565c0",
    "success"   : "#2e7d32",
    "warning"   : "#e65100",
    "danger"    : "#c62828",
    "teal"      : "#00695c",
    "secondary" : "#455a64",
    "light"     : "#e3f2fd",
    "dark"      : "#1a237e",
    "purple"    : "#6a1b9a",
    "amber"     : "#f57f17",
    "pink"      : "#ad1457",
    "grey"      : "#546e7a",
    "white"     : "#ffffff",
    "black"     : "#212121",
}

# ── Sidebar ───────────────────────────────────
with st.sidebar:
    if LOGO.exists():
        st.image(str(LOGO), width=70)
    st.markdown("### 🧑‍💼 HR Attrition")
    st.markdown("**Author:** Mohamed · M3")
    st.markdown("---")
    st.markdown("#### 📂 Navigation")
    st.markdown("""
- **🏠 Home** ← You are here
- **📊 EDA Dashboard** → Stage 1
- **🤖 ML Models** → Stage 2
""")
    st.markdown("---")
    st.markdown("#### 📁 Dataset Info")
    st.markdown("""
- **Source:** IBM HR Analytics (Kaggle)
- **Rows:** 1,470 employees
- **Columns:** 35 features
- **Target:** Attrition (Yes / No)
""")
    st.markdown("---")
    st.caption("© M3 · Data Analysis Portfolio")

# ── Main Header ───────────────────────────────
st.markdown(f"""
<div style='background: linear-gradient(135deg, {CLR["dark"]} 0%, {CLR["purple"]} 100%);
            padding: 2.5rem 2rem; border-radius: 16px; margin-bottom: 1.5rem;'>
    <h1 style='color: white; margin: 0; font-size: 2.4rem;'>
        🧑‍💼 HR Employee Attrition Analysis
    </h1>
    <p style='color: #e1bee7; margin: 0.5rem 0 0 0; font-size: 1.1rem;'>
        End-to-End Data Analysis & Machine Learning · IBM HR Analytics Dataset
    </p>
</div>
""", unsafe_allow_html=True)

# ── KPI Cards ─────────────────────────────────
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown(f"""
    <div style='background:white; padding:1.2rem; border-radius:12px;
                border-left:5px solid {CLR["primary"]}; text-align:center;'>
        <h2 style='color:{CLR["primary"]}; margin:0;'>1,470</h2>
        <p style='color:{CLR["secondary"]}; margin:0; font-size:0.85rem;'>Total Employees</p>
    </div>""", unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div style='background:white; padding:1.2rem; border-radius:12px;
                border-left:5px solid {CLR["danger"]}; text-align:center;'>
        <h2 style='color:{CLR["danger"]}; margin:0;'>16.1%</h2>
        <p style='color:{CLR["secondary"]}; margin:0; font-size:0.85rem;'>Attrition Rate</p>
    </div>""", unsafe_allow_html=True)

with col3:
    st.markdown(f"""
    <div style='background:white; padding:1.2rem; border-radius:12px;
                border-left:5px solid {CLR["success"]}; text-align:center;'>
        <h2 style='color:{CLR["success"]}; margin:0;'>35</h2>
        <p style='color:{CLR["secondary"]}; margin:0; font-size:0.85rem;'>Features</p>
    </div>""", unsafe_allow_html=True)

with col4:
    st.markdown(f"""
    <div style='background:white; padding:1.2rem; border-radius:12px;
                border-left:5px solid {CLR["purple"]}; text-align:center;'>
        <h2 style='color:{CLR["purple"]}; margin:0;'>9</h2>
        <p style='color:{CLR["secondary"]}; margin:0; font-size:0.85rem;'>Job Roles</p>
    </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── Two Stages ────────────────────────────────
col_a, col_b = st.columns(2, gap="large")

with col_a:
    st.markdown(f"""
    <div style='background:white; border:1px solid #e0e0e0; border-radius:14px;
                padding:1.5rem; height:100%;'>
        <h3 style='color:{CLR["primary"]}; margin-top:0;'>📊 Stage 1 — EDA Dashboard</h3>
        <p style='color:{CLR["secondary"]};'>Deep exploratory analysis across 12 tabs:</p>
        <ul style='color:{CLR["black"]}; line-height:1.9;'>
            <li>Data Overview & Correlation</li>
            <li>Variables Analysis</li>
            <li>IQR Cleaning & Outliers Lab</li>
            <li>Missing Values & Multicollinearity</li>
            <li>Insights & Recommendations</li>
            <li>📦 HR KPI Dashboard <b>(NEW)</b></li>
            <li>🔍 Attrition Deep Dive <b>(NEW)</b></li>
            <li>💰 Salary Analysis <b>(NEW)</b></li>
            <li>🧪 Statistical Tests</li>
        </ul>
    </div>""", unsafe_allow_html=True)

with col_b:
    st.markdown(f"""
    <div style='background:white; border:1px solid #e0e0e0; border-radius:14px;
                padding:1.5rem; height:100%;'>
        <h3 style='color:{CLR["success"]}; margin-top:0;'>🤖 Stage 2 — ML Models</h3>
        <p style='color:{CLR["secondary"]};'>Full machine learning pipeline across 5 tabs:</p>
        <ul style='color:{CLR["black"]}; line-height:1.9;'>
            <li>Regression Models (6) → predict <b>Monthly Income</b></li>
            <li>Classification Models (6) → predict <b>Attrition</b> (Yes/No)</li>
            <li>Model Comparison & Report</li>
            <li>Predict New Employee</li>
            <li>Final Insights & Report (PDF + Word)</li>
        </ul>
        <br>
        <p style='color:{CLR["secondary"]}; font-size:0.85rem;'>
            ✅ 12 models · Parallel training · class_weight=balanced · AUC evaluation
        </p>
    </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── Business Context ──────────────────────────
st.markdown(f"""
<div style='background:{CLR["light"]}; border-radius:12px; padding:1.2rem 1.5rem;
            border-left: 5px solid {CLR["purple"]}; margin-bottom:1rem;'>
    <h4 style='color:{CLR["dark"]}; margin-top:0;'>🎯 Why This Project Matters</h4>
    <p style='color:{CLR["black"]}; margin:0; line-height:1.8;'>
        Employee attrition costs companies <b>50–200% of an employee's annual salary</b> to replace.
        This project identifies <b>who is likely to leave, why, and when</b> —
        enabling HR teams to take proactive retention actions before it is too late.
    </p>
</div>
""", unsafe_allow_html=True)

# ── How to Use ────────────────────────────────
st.markdown(f"""
<div style='background:{CLR["light"]}; border-radius:12px; padding:1.2rem 1.5rem;'>
    <h4 style='color:{CLR["dark"]}; margin-top:0;'>🚀 How to Use</h4>
    <ol style='color:{CLR["black"]}; line-height:2;'>
        <li>Go to <b>📊 EDA Dashboard</b> → upload <code>hr_attrition_clean.csv</code></li>
        <li>Explore all 12 EDA tabs — insights auto-saved to session</li>
        <li>Go to <b>🤖 ML Models</b> → data flows automatically from Stage 1</li>
        <li>Train models, compare results, export the final report</li>
    </ol>
</div>
""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── Footer ────────────────────────────────────
st.markdown(f"""
<div style='text-align:center; color:{CLR["grey"]}; font-size:0.8rem; padding:1rem;'>
    Built with ❤️ by Mohamed · M3 · Data Analysis Portfolio · IBM HR Analytics Dataset
</div>
""", unsafe_allow_html=True)
