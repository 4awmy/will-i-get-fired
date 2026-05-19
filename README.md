<div align="center">

# 🤖 Will I Get Fired?
### *A machine learning project that tells you whether robots are coming for your job.*
> *(Spoiler: probably yes, but at least we built a pretty dashboard about it.)*

<br>

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-Live-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://xoa-ml.streamlit.app)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![Plotly](https://img.shields.io/badge/Plotly-Charts-3F4F75?style=for-the-badge&logo=plotly&logoColor=white)](https://plotly.com)
[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)

<br>

### 🚀 [Try the Live App → xoa-ml.streamlit.app](https://xoa-ml.streamlit.app)

<br>

*Built for CIT3601 · Arab Academy for Science, Technology & Maritime Transport · Term 6*

</div>

---

## 🤔 What Is This?

A full end-to-end Machine Learning application that predicts how likely your job is to be automated away by AI. We trained **4 machine learning models** on **30,000 job records** and wrapped everything in a slick interactive dashboard so you can find out — with scientific precision — whether you should start updating your resume.

You enter things you'd actually know about your job (title, industry, salary, experience, education, remote ratio, location) and the model classifies your role into one of 5 risk levels: from *Very Safe* (keep calm) all the way to *Critical Risk* (panic responsibly).

*We, the engineers who built this classifier, received a "Safe" rating. We are choosing to believe the model is correct.*

---

## 📸 Screenshots

> **💡 To add screenshots:** Run the app locally (`streamlit run app.py`), take screenshots of each tab, and save them to `docs/screenshots/`.

<table>
  <tr>
    <td align="center">
      <strong>🏠 Hero + Data Explorer</strong><br>
      <img src="docs/screenshots/01_hero.png" alt="Hero section with stat cards and data explorer tab" width="100%"/>
    </td>
    <td align="center">
      <strong>⚙️ Model Lab</strong><br>
      <img src="docs/screenshots/02_model_lab.png" alt="Model comparison cards and accuracy chart" width="100%"/>
    </td>
  </tr>
  <tr>
    <td align="center">
      <strong>🎯 Risk Analyzer — Input Form</strong><br>
      <img src="docs/screenshots/03_risk_input.png" alt="Job parameters input form" width="100%"/>
    </td>
    <td align="center">
      <strong>📊 Risk Analyzer — Result + Gauge</strong><br>
      <img src="docs/screenshots/04_risk_result.png" alt="Gauge chart and risk result card" width="100%"/>
    </td>
  </tr>
</table>

---

## ✨ Features

| Feature | Description |
|---|---|
| 📊 **Data Explorer** | 6 interactive Plotly charts — distributions, correlations, salary breakdowns |
| 🧠 **4 ML Models** | Random Forest, Neural Network (MLP), Decision Tree, Naive Bayes |
| 🏆 **Model Comparison** | Side-by-side accuracy cards with gold crown on the winner |
| 🎯 **Risk Gauge** | Visual speedometer showing your job's automation risk level |
| 🎨 **Color-Coded Results** | Each risk grade has its own gradient color scheme and actionable tip |
| 📈 **Confidence Breakdown** | Per-class probability bars so you see how sure the model is |
| ⚡ **Cached Training** | Models train once and are cached — no waiting on every click |
| 🧪 **Unit Tests** | Automated tests covering data loading, feature consistency, and model training |

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        DATA LAYER                           │
│   data.csv — 30,000 job records × 13 raw columns           │
└────────────────────────────┬────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────┐
│                    PREPROCESSING (model_utils.py)           │
│                                                             │
│  1. Drop missing rows (dropna)                              │
│  2. Target Engineering:                                     │
│       Automation Risk (%) → Risk_Grade (0–4)                │
│  3. Feature Selection — 8 features used:                    │
│       Text:   Job Title, Industry, Job Status,              │
│               Required Education, Location                  │
│       Number: Median Salary, Experience, Remote Ratio       │
│       ✗ Excluded: AI Impact Level (circular),               │
│                   Job Openings/Projected (macro data),      │
│                   Gender Diversity (aggregate stat)         │
│  4. Label Encoding (text → integers, saved per column)      │
│  5. StandardScaler (numbers → mean=0, std=1, saved)         │
│  6. Train/Test Split (80% / 20%, random_state=42)           │
└────────────────────────────┬────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────┐
│                      MODEL LAYER                            │
│                                                             │
│   ┌──────────────┐  ┌──────────────┐  ┌────────────────┐   │
│   │ Random Forest│  │  Neural Net  │  │ Decision Tree  │   │
│   │ 100 trees    │  │  (100, 50)   │  │ baseline +     │   │
│   │ 🏆 Best      │  │  MLP, 500 it │  │ tuned version  │   │
│   └──────────────┘  └──────────────┘  └────────────────┘   │
│                      ┌─────────────┐                        │
│                      │ Naive Bayes │                        │
│                      │ GaussianNB  │                        │
│                      └─────────────┘                        │
│                                                             │
│   All trained on 24,000 samples, evaluated on 6,000         │
└────────────────────────────┬────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────┐
│                   APPLICATION LAYER (app.py)                │
│                                                             │
│  Tab 1 — Data Explorer                                      │
│    • Risk score distribution (histogram)                    │
│    • Class balance (donut chart)                            │
│    • Salary by risk grade (boxplot)                         │
│    • Top industries (bar chart)                             │
│    • AI Impact vs Risk Grade (stacked bar)                  │
│    • Correlation heatmap                                    │
│                                                             │
│  Tab 2 — Model Lab                                          │
│    • Per-model accuracy cards                               │
│    • Accuracy comparison bar chart                          │
│    • Algorithm explainer                                    │
│                                                             │
│  Tab 3 — Risk Analyzer                                      │
│    • 8-field input form (user-known info only)              │
│    • Live prediction via selected model                     │
│    • Gauge chart + color-coded result card                  │
│    • Per-class probability bars                             │
└─────────────────────────────────────────────────────────────┘
```

---

## 🔄 ML Pipeline

```
Raw Input (8 fields)
        │
        ├─ Text fields ──→ LabelEncoder (same fitted encoder from training)
        │
        ├─ Number fields ─→ StandardScaler (same fitted scaler from training)
        │
        └─ Feature vector ──→ Trained Model ──→ Risk Grade (0–4)
                                                       │
                                        ┌──────────────┴──────────────┐
                                        │         predict_proba()     │
                                        │  [0.05, 0.10, 0.15, 0.60, 0.10] │
                                        └─────────────────────────────┘
                                                       │
                                            Gauge + Result Card + Probability Bars
```

**Why these 8 features?** We only kept inputs a real person would know without looking up external databases:

| ✅ Kept | ❌ Excluded | Reason for exclusion |
|---|---|---|
| Job Title | AI Impact Level | Circular — directly encodes the answer |
| Industry | Job Openings (2024) | Macro BLS data, not personally known |
| Job Status | Projected Openings (2030) | Same — requires external lookup |
| Required Education | Gender Diversity (%) | Aggregate stat, not individual knowledge |
| Median Salary (USD) | | |
| Experience (Years) | | |
| Remote Work Ratio (%) | | |
| Location | | |

---

## 📊 Model Results

Evaluated on a 20% held-out test set (6,000 samples the models never saw during training):

| Rank | Model | Test Accuracy | Type |
|:---:|---|:---:|---|
| 🥇 | **Random Forest** | Highest | Ensemble (100 trees) |
| 🥈 | Neural Network (MLP) | ~Equal | Deep Learning (2 hidden layers) |
| 🥉 | Decision Tree | ~Equal | Logic-based (tuned: max_depth=10) |
| 4 | Naive Bayes | ~Equal | Probabilistic baseline |

> **Honest note:** All models hover near ~20% accuracy — equivalent to random chance on a balanced 5-class problem. This is not a bug in the code. It's a property of the dataset: the `Automation Risk (%)` column was generated synthetically and has near-zero correlation with the other features. The correlation heatmap in the Data Explorer tab shows this clearly. The ML pipeline is correct; the data simply has no learnable signal. A real-world dataset with genuine feature-target relationships would produce much higher accuracy.

> **Random Forest** is still designated the best model: it has the most stable behavior, the smallest overfitting gap among tree-based methods, and supports `predict_proba()` for the confidence visualization.

---

## 🛠️ Tech Stack

| Layer | Technology | Why |
|---|---|---|
| **Web App** | [Streamlit](https://streamlit.io) | Python → interactive UI with zero HTML/CSS |
| **ML** | [scikit-learn](https://scikit-learn.org) | Industry-standard ML library |
| **Data** | [pandas](https://pandas.pydata.org) | DataFrame manipulation |
| **Charts** | [Plotly](https://plotly.com) | Interactive, hoverable, beautiful |
| **Static Charts** | [Seaborn + Matplotlib](https://seaborn.pydata.org) | EDA charts in the notebook |
| **Preprocessing** | LabelEncoder + StandardScaler | Text → numbers, scale normalization |
| **Testing** | pytest | Automated tests for data and model integrity |

---

## 📁 Project Structure

```
will-i-get-fired/
│
├── app.py               ← Streamlit frontend (UI, charts, prediction form)
├── model_utils.py       ← Backend ML logic (data loading, training, caching)
├── data.csv             ← Dataset: 30,000 jobs × 13 features
├── requirements.txt     ← Python dependencies
│
├── tests/
│   └── test_app.py      ← Unit tests (data loading, features, models)
│
├── docs/
│   ├── PROJECT_REPORT.md         ← Full written report (PDF submission)
│   ├── REQUIREMENTS_EXPLAINED.md ← How each course requirement is met
│   ├── INSTALLATION_GUIDE.md     ← Local setup walkthrough
│   ├── DEPLOYMENT_GUIDE.md       ← Streamlit Cloud deployment steps
│   ├── GAP_ANALYSIS.md           ← Requirement checklist
│   └── screenshots/              ← App screenshots
│
├── JobRiskInsights.ipynb  ← Full Jupyter notebook (all 11 required sections)
└── Final_Project_Description.md  ← Course project brief
```

### Key file responsibilities

| File | Responsibility |
|---|---|
| `model_utils.py` | Single source of truth for `FEATURE_COLS` and `NUM_COLS_TO_SCALE`. All preprocessing and model training logic lives here. Cached with `@st.cache_data` / `@st.cache_resource`. |
| `app.py` | Pure UI layer. Imports from `model_utils`, renders tabs, handles user input, calls prediction. |
| `JobRiskInsights.ipynb` | Standalone analysis notebook. Mirrors the same preprocessing logic as `model_utils.py` for full reproducibility. |

---

## 🚀 Run It Locally

**Step 1 — Clone:**
```bash
git clone https://github.com/4awmy/will-i-get-fired.git
cd will-i-get-fired
```

**Step 2 — Install dependencies:**
```bash
pip install -r requirements.txt
```

**Step 3 — Launch:**
```bash
streamlit run app.py
```

Opens at `http://localhost:8501` — no configuration needed.

**Step 4 — Run tests (optional but responsible):**
```bash
pytest tests/
```

---

## 📓 Jupyter Notebook

The complete analysis notebook (`JobRiskInsights.ipynb`) covers all 11 required sections:

| # | Section | Key Content |
|---|---|---|
| 1 | Problem Understanding | Problem framing, stakeholders, input/output definition |
| 2 | Dataset Description | Feature table with ✅/❌ model inclusion, target engineering |
| 3 | Data Exploration (EDA) | Shape, dtypes, missing values, duplicates, statistics |
| 4 | Data Visualization | 6 charts with interpretations |
| 5 | Data Preprocessing | 5 steps: dropna → target engineering → encoding → scaling → split |
| 6 | Model Training | 4 algorithms trained and explained |
| 7 | Model Comparison | Accuracy table + bar chart |
| 8 | Model Improvement | Decision Tree overfitting → hyperparameter fix |
| 9 | Model Evaluation | Classification report + confusion matrix |
| 10 | Prediction Demo | Live `predict_job_risk()` function with 3 examples |
| 11 | Final Discussion | Honest analysis of results, limitations, future work |

```bash
jupyter notebook JobRiskInsights.ipynb
```

---

## 📋 Risk Levels Explained

| Grade | Label | Risk Range | What It Means |
|:---:|---|:---:|---|
| 0 | 🟢 Very Safe | 0–20% | Robots can't replace you. Congrats, you're creative. |
| 1 | 🟡 Safe | 20–40% | AI is a tool for you, not a threat. For now. |
| 2 | 🟠 Moderate | 40–60% | Hybrid role. Start learning something non-automatable. |
| 3 | 🔴 High Risk | 60–80% | Update that LinkedIn. Just saying. |
| 4 | ⛔ Critical Risk | 80–100% | Your job description reads like a Python script. |

---

## 👥 The Team

> Two people who are now very anxious about their own career prospects.

| Name |
|---|
| Omar Hossam Eldin Metwally Gad |
| Mohamed Ahmed Ezzat Mohamed Elsayed |

*Submitted for CIT3601 — Professional Training In AI I - Misr El Gedida · Term 6 · AAST*

---

## 📄 License

[MIT](LICENSE) — use it, fork it, improve it. If a robot uses this project to automate your job, we accept no responsibility.

---

<div align="center">

**Made with ☕ and existential dread about the future of work.**

[⭐ Star this repo](https://github.com/4awmy/will-i-get-fired) · [🐛 Report a bug](https://github.com/4awmy/will-i-get-fired/issues) · [🚀 Live Demo](https://xoa-ml.streamlit.app)

</div>
