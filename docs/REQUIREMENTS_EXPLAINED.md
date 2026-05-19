# 📋 Project Requirements Explained

This document details how the **AI Job Market Risk Analyzer** satisfies the CIT3601 course objectives. You can use this content for your **Written Report** and **PowerPoint Presentation**.

---

## 1. Project Overview
*   **Goal:** Apply machine learning to predict the automation risk of various jobs.
*   **Why?** To demonstrate the end-to-end ML workflow: Data -> Processing -> Modeling -> Evaluation -> Insight.

## 2. Dataset Understanding
*   **Source:** Frey & Osborne (2013) "The Future of Employment" — Oxford University academic research dataset, hosted via Plotly open datasets. This is real-world occupational data, not synthetic.
*   **Size:** 702 real U.S. occupations (Exceeds the >200 requirement).
*   **Features:** 4 dataset columns, of which 3 are used as model inputs: `occupation` (text), `education` (text), and `average_ann_wage` (numerical). The fourth column, `probability`, is the target variable — automation probability (0.0–1.0) derived by Frey & Osborne from a Gaussian process classifier trained on O*NET task and skill features.
*   **Problem Type:** **Classification**. We classify occupations into 5 discrete risk categories (0: Very Safe to 4: Critical Risk) derived from the continuous automation probability using equal-width bins.

## 3. Data Preprocessing (How we handled the data)
*   **Cleaning:** Used `dropna()` to remove incomplete rows, ensuring model stability.
*   **Encoding:** Converted text data (e.g., "Accountants", "Bachelor's degree") into numbers using `LabelEncoder` so the math models can understand them.
*   **Scaling:** Used `StandardScaler` to normalize the `average_ann_wage` column (values range from ~$20K to ~$190K) so large numbers don't bias the model. This is crucial for models like Neural Networks and Naive Bayes.
*   **Target Engineering:** Converted the continuous `probability` (0.0–1.0) into 5 "Risk Grades" using equal-width bins to turn this into a classification problem. Class imbalance in the resulting grades is handled with `class_weight='balanced'` across all models.

## 4. Model Development
We selected three distinct algorithms to compare performance:

1.  **Random Forest Classifier (Ensemble):**
    *   *Why?* It builds multiple decision trees and averages them. It is very accurate and resists overfitting.
2.  **Naive Bayes (Probabilistic):**
    *   *Why?* A simple, fast baseline model based on probability theory. Good for checking if complex models are actually necessary.
3.  **Decision Tree (Logic-based):**
    *   *Why?* Easy to interpret (if-then rules), but prone to overfitting.
4.  **Artificial Neural Network (MLP):**
    *   *Why?* To capture complex, non-linear relationships in the data. Uses `MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500)` — fulfills the "Advanced Model" course requirement.

## 5. Model Evaluation
*   **Metric:** **Accuracy Score** (Correct Predictions / Total Predictions).
*   **Results:** The application dynamically calculates accuracy on a 20% test split (141 samples). Observed results: Naive Bayes ~58.2%, Neural Network ~56.0%, Random Forest ~54.6%, Decision Tree ~48.9%.
*   **Insight:** Naive Bayes outperforms the more complex models because it generalizes efficiently with the small 561-sample training set. The strong salary-automation correlation (r = −0.550) and education-risk relationship provide genuine learnable signal, enabling all models to substantially outperform random chance (~20% baseline).

## 6. Actionable Insights (Result Interpretation)
The application doesn't just output a number; it translates predictions into advice:
*   **Safe Roles:** Suggests the user is in a creative/socially complex field.
*   **High Risk Roles:** Suggests routine/repetitive tasks.
*   **Advice:** Upskilling in "Soft Skills" (Leadership, Empathy) is recommended for High Risk categories.
