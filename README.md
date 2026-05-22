# 🧑‍💼 HR Employee Attrition — End-to-End Data Analysis & ML
> **Author:** Mohamed · M3 · Data Analysis Portfolio  
> **Dataset:** IBM HR Analytics Employee Attrition & Performance (Kaggle)  
> **Stack:** Python · Streamlit · Scikit-learn · Pandas · Matplotlib · Seaborn

---

## 📌 Project Overview

A full end-to-end data analysis and machine learning project built on the **IBM HR Analytics dataset** — a rich real-world HR dataset containing 1,470 employee records with 35 features covering demographics, job information, satisfaction scores, performance, and compensation.

This project answers the critical business question: **Who is likely to leave, why, and what can HR do about it?**

> Employee attrition costs companies **50–200% of an employee's annual salary** to replace. Early identification enables proactive retention strategies.

---

## 🗂️ Dataset

| Property | Value |
|----------|-------|
| Source | [Kaggle — IBM HR Analytics](https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset) |
| File | `WA_Fn-UseC_-HR-Employee-Attrition.csv` |
| Shape | 1,470 rows × 35 columns |
| Missing Values | None |
| Attrition Rate | 16.1% (237 left / 1,233 stayed) |

### Dataset Prep (Jupyter)
```python
df["Attrition_flag"] = (df["Attrition"] == "Yes").astype(int)
df["OverTime_flag"]  = (df["OverTime"] == "Yes").astype(int)
df["Gender_flag"]    = (df["Gender"] == "Male").astype(int)
df.drop(columns=["EmployeeCount","StandardHours","Over18"], inplace=True)
df.to_csv("hr_attrition_clean.csv", index=False)
```

---

## 🏗️ Architecture

```
📁 Repo_3_HR_Attrition/
├── Home.py                    ← Multipage launcher
├── M3_logo.png                ← Brand logo
├── requirements.txt
├── README.md
├── pages/
│   ├── EDA_dashboard.py       ← Stage 1 (Tabs 1–12)
│   └── ML_Models.py           ← Stage 2 (Tabs 9–13)
└── data/
    └── hr_attrition_clean.csv
```

**Run:**
```bash
streamlit run Home.py
```

---

## 📊 Stage 1 — EDA Dashboard (12 Tabs)

| Tab | Name | Description |
|-----|------|-------------|
| 1 | Data Overview & Correlation | Shape, dtypes, top correlations with target |
| 2 | Variables Analysis | Distributions, histograms, categorical counts |
| 3 | IQR Cleaning | Outlier detection and removal |
| 4 | Outliers Lab | Z-score, box plots, outlier summary |
| 5 | Dashboard Summary | Visual summary of cleaned data |
| 6 | Missing Values | Heatmap, imputation strategies |
| 7 | Multicollinearity | VIF analysis, correlation matrix |
| 8 | Insights | Auto-generated EDA insights |
| 9 | **HR KPI Dashboard** ⭐ | Attrition rate, headcount, income, tenure KPIs |
| 10 | **Attrition Deep Dive** ⭐ | Who leaves? By dept, role, travel, overtime, satisfaction |
| 11 | **Salary Analysis** ⭐ | Income by role, level, dept · hike vs performance |
| 12 | **Statistical Tests** ⭐ | T-Test, ANOVA, Chi-Square — p-value confirmed insights |

> ⭐ = New tabs — HR-specific business analysis

---

## 🤖 Stage 2 — ML Models (5 Tabs)

| Tab | Name | Description |
|-----|------|-------------|
| 9 | Regression Models (6) | Predict **Monthly Income** |
| 10 | Classification Models (6) | Predict **Attrition** (0=Stay, 1=Leave) |
| 11 | Comparison & Report | Side-by-side model comparison |
| 12 | Predict New Employee | Input features → get attrition prediction |
| 13 | Final Insights & Report | Auto-report → export PDF + Word |

### Models Included
**Regression (6):** Linear · Ridge · Lasso · Decision Tree · Random Forest · Gradient Boosting

**Classification (6):** Logistic Regression · KNN · Decision Tree · Random Forest · Gradient Boosting · SVM

### ML Targets
| Type | Target | Note |
|------|--------|------|
| Classification | `Attrition_flag` (0/1) | `class_weight="balanced"` — handles 84/16 imbalance |
| Regression | `MonthlyIncome` | Predict employee salary |

---

## 🔑 Key Business Insights

- **Overall Attrition Rate:** 16.1% — significantly above healthy 5–10% benchmark
- **OverTime is the #1 driver:** employees working overtime leave at 2–3x the rate
- **Low satisfaction = high attrition:** JobSatisfaction score 1 shows highest departure rate
- **Early tenure is critical:** most attrition happens in first 1–3 years
- **Sales Representatives:** highest attrition rate by job role
- **Statistical Tests:** OverTime vs Attrition (Chi² p < 0.05) · Income differs significantly (T-Test p < 0.05)

---

## ⚙️ Features

| Feature | Detail |
|---------|--------|
| Class imbalance handling | `class_weight="balanced"` on LR, DT, RF, SVM |
| AUC evaluation | Honest metric for 84/16 imbalanced data |
| Parallel ML training | ThreadPoolExecutor · CPU slider |
| Save/Load models | joblib |
| PDF + Word export | reportlab · python-docx |
| Session bridge | EDA → ML via `st.session_state` |

---

## 🚀 How to Run

```bash
# 1. Clone the repo
git clone https://github.com/your-username/Repo_3_HR_Attrition.git
cd Repo_3_HR_Attrition

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the app
streamlit run Home.py

# 4. Upload hr_attrition_clean.csv in EDA Dashboard
```

## 📥 Dataset Setup
1. Download from [Kaggle — IBM HR Analytics](https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset)
2. Run the prep script in Jupyter → produces `hr_attrition_clean.csv`
3. Place in `/data` folder or upload via app

---

## 🗺️ Portfolio Roadmap

| # | Project | Domain | Status |
|---|---------|--------|--------|
| P1 | House Prices Analysis | Real Estate | ✅ Complete |
| P2 | Olist E-Commerce | Retail / Business | ✅ Complete |
| P3 | HR Attrition | Human Resources | ✅ Complete |
| P4–P7 | Coming Soon... | Various Domains | 🔜 |

---

## 👤 Author

**Mohamed · M3**  
Mechanical Engineer → Data Analyst  
📊 Building a professional DA/DS portfolio — one dataset at a time.

---

*Built with ❤️ using Python & Streamlit*
