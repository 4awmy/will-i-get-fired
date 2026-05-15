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

*Built for CAI3101 · Arab Academy for Science, Technology & Maritime Transport · Term 5*

</div>

---

## 🤔 What Is This?

A full end-to-end Machine Learning application that predicts how likely your job is to be automated away by AI. We trained **4 machine learning models** on **30,000 job records** and wrapped everything in a slick interactive dashboard so you can find out — with scientific precision — whether you should start updating your resume.

The models look at your job's title, industry, salary, education requirements, AI impact level, and a bunch of other features, then classify your role into one of 5 risk levels: from *Very Safe* (keep calm) all the way to *Critical Risk* (panic responsibly).

*We, the engineers who built this classifier, received a "Safe" rating. We are choosing to believe the model is correct.*

---

## 📸 Screenshots

> **💡 To add screenshots:** Run the app locally (`streamlit run app.py`), take screenshots of each tab, and save them to `docs/screenshots/`. Then replace the placeholder paths below.

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
| 🧪 **Unit Tests** | 4 automated tests covering data loading, feature consistency, and model training |

---

## 🧠 How It Works (The 30-Second Version)

```
Your job details  →  Label Encode text  →  StandardScale numbers
       ↓
  Random Forest (or whichever model you pick)
       ↓
  Risk Grade: 0 (Very Safe) → 4 (Critical Risk)
       ↓
  "You might want to learn prompt engineering."
```

More formally:

1. **Data** — 30,000 job records with 12 features (title, industry, salary, AI impact, etc.)
2. **Preprocessing** — Text columns encoded with `LabelEncoder`, numerical columns normalized with `StandardScaler`
3. **Target Engineering** — Continuous `Automation Risk (%)` converted to 5 discrete classes
4. **Training** — 80/20 train-test split, 4 algorithms trained in parallel
5. **Evaluation** — Accuracy, Precision, Recall, F1-score, Confusion Matrix
6. **Deployment** — Streamlit app with live prediction on new inputs

---

## 📊 Model Results

Evaluated on a 20% held-out test set (6,000 samples the models never saw during training):

| Rank | Model | Test Accuracy | Type |
|:---:|---|:---:|---|
| 🥇 | **Random Forest** | Highest | Ensemble (100 trees) |
| 🥈 | Neural Network (MLP) | Second | Deep Learning (2 hidden layers) |
| 🥉 | Decision Tree | Third | Logic-based |
| 4 | Naive Bayes | Fourth | Probabilistic baseline |

> **Random Forest wins** because averaging 100 trees cancels out the noise that kills a single tree's generalization. Also it's just really good at everything. We love Random Forest.

---

## 🛠️ Tech Stack

| Layer | Technology | Why |
|---|---|---|
| **Web App** | [Streamlit](https://streamlit.io) | Python → interactive UI, no HTML/CSS needed |
| **ML** | [scikit-learn](https://scikit-learn.org) | Industry-standard ML library |
| **Data** | [pandas](https://pandas.pydata.org) | DataFrame manipulation |
| **Charts** | [Plotly](https://plotly.com) | Interactive, hoverable, beautiful |
| **Preprocessing** | LabelEncoder + StandardScaler | Text → numbers, scale normalization |
| **Testing** | pytest | 4 automated tests for data and model integrity |

---

## 📁 Project Structure

```
will-i-get-fired/
│
├── app.py               ← Streamlit frontend (UI, charts, prediction form)
├── model_utils.py       ← Backend ML logic (loading, training, caching)
├── data.csv             ← Dataset: 30,000 jobs × 13 features
├── requirements.txt     ← Python dependencies
│
├── tests/
│   └── test_app.py      ← 4 unit tests (data loading, features, models)
│
├── docs/
│   ├── PROJECT_REPORT.md         ← Full written report (use for PDF submission)
│   ├── REQUIREMENTS_EXPLAINED.md ← How we meet each course requirement
│   ├── INSTALLATION_GUIDE.md     ← Local setup walkthrough
│   ├── DEPLOYMENT_GUIDE.md       ← Streamlit Cloud deployment steps
│   ├── GAP_ANALYSIS.md           ← Requirement checklist
│   └── screenshots/              ← Add your screenshots here
│
├── JobRiskInsights.ipynb  ← Full Jupyter notebook (all 11 required sections)
└── Final_Project_Description.md  ← Course project brief
```

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

The app opens at `http://localhost:8501` — no configuration needed.

**Step 4 — Run tests (optional but responsible):**
```bash
pytest tests/
```

---

## 📓 Jupyter Notebook

The complete analysis notebook (`JobRiskInsights.ipynb`) contains all 11 required sections:

1. Problem Understanding
2. Dataset Description
3. Data Exploration (EDA)
4. Data Visualization (6 charts with explanations)
5. Data Preprocessing (5 steps with reasoning)
6. Model Training (4 algorithms)
7. Model Comparison
8. Model Improvement (overfitting experiment)
9. Model Evaluation (Precision, Recall, F1, Confusion Matrix)
10. Prediction Demo
11. Final Discussion

Run it with `jupyter notebook JobRiskInsights.ipynb` or open in VS Code.

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

> Three people who are now very anxious about their own career prospects.

| Name |
|---|
| Omar Hossam Eldin Metwally Gad |
| Mohamed Ahmed Ezzat Mohamed Elsayed |
| Belal Ashraf Sobhy Mohamed Hassan |

*Submitted for CAI3101 — Introduction to Artificial Intelligence · Term 5 · AAST*

---

## 📄 License

[MIT](LICENSE) — use it, fork it, improve it. If a robot uses this project to automate your job, we accept no responsibility.

---

<div align="center">

**Made with ☕ and existential dread about the future of work.**

[⭐ Star this repo](https://github.com/4awmy/will-i-get-fired) · [🐛 Report a bug](https://github.com/4awmy/will-i-get-fired/issues) · [🚀 Live Demo](https://xoa-ml.streamlit.app)

</div>
