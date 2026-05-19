# 📋 Project Requirements Explained

This document details how the **AI Job Market Risk Analyzer** satisfies the CIT3601 course objectives. You can use this content for your **Written Report** and **PowerPoint Presentation**.

---

## 1. Project Overview
*   **Goal:** Apply machine learning to predict the automation risk of various jobs.
*   **Why?** To demonstrate the end-to-end ML workflow: Data -> Processing -> Modeling -> Evaluation -> Insight.

## 2. Dataset Understanding
*   **Source:** Synthetic/Public dataset representing Job Market trends.
*   **Size:** 30,000 Samples (Exceeds the >200 requirement).
*   **Features:** 13 dataset columns, of which 8 are used as model inputs: `Job Title`, `Industry`, `Job Status`, `Median Salary`, `Required Education`, `Experience`, `Remote Work Ratio`, `Location`. Four columns are excluded — `AI Impact Level` (circular), `Job Openings (2024)`, `Projected Openings (2030)`, and `Gender Diversity (%)` (macro/aggregate data).
*   **Problem Type:** **Classification**. We are classifying jobs into 5 discrete risk categories (0: Very Safe to 4: Critical Risk).

## 3. Data Preprocessing (How we handled the data)
*   **Cleaning:** Used `dropna()` to remove incomplete rows, ensuring model stability.
*   **Encoding:** Converted text data (e.g., "Software Engineer", "IT") into numbers using `LabelEncoder` so the math models can understand them.
*   **Scaling:** Used `StandardScaler` to normalize numerical values (like Salary: $50k vs $150k) so large numbers don't bias the model. This is crucial for models like Neural Networks and Naive Bayes.
*   **Target Engineering:** Converted the continuous "Automation Risk %" (0-100) into 5 "Risk Grades" to turn this into a classification problem.

## 4. Model Development
We selected three distinct algorithms to compare performance:

1.  **Random Forest Classifier (Ensemble):**
    *   *Why?* It builds multiple decision trees and averages them. It is very accurate and resists overfitting.
2.  **Naive Bayes (Probabilistic):**
    *   *Why?* A simple, fast baseline model based on probability theory. Good for checking if complex models are actually necessary.
3.  **Decision Tree (Logic-based):**
    *   *Why?* Easy to interpret (if-then rules), but prone to overfitting.
4.  *(Planned)* **Artificial Neural Network (MLP):**
    *   *Why?* To capture complex, non-linear relationships in the data, fulfilling the "Advanced Model" suggestion.

## 5. Model Evaluation
*   **Metric:** **Accuracy Score** (Correct Predictions / Total Predictions).
*   **Results:** The application dynamically calculates accuracy on a 20% test split.
*   **Insight:** We expect Random Forest (or the Neural Network) to outperform Naive Bayes due to the complexity of the data.

## 6. Actionable Insights (Result Interpretation)
The application doesn't just output a number; it translates predictions into advice:
*   **Safe Roles:** Suggests the user is in a creative/socially complex field.
*   **High Risk Roles:** Suggests routine/repetitive tasks.
*   **Advice:** Upskilling in "Soft Skills" (Leadership, Empathy) is recommended for High Risk categories.
