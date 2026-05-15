# AI Job Market Risk Analyzer
## CAI3101 — Introduction to Artificial Intelligence
### Arab Academy for Science, Technology & Maritime Transport
### College of Computing and Information Technology

**Team Members:**
- Omar Hossam Eldin Metwally Gad
- Mohamed Ahmed Ezzat Mohamed Elsayed
- Belal Ashraf Sobhy Mohamed Hassan

**Term:** Term 5

---

## 1. Introduction

Artificial Intelligence is rapidly transforming the global job market. As automation becomes more affordable and capable, many roles that once required human effort are being partially or fully replaced by machines. This project applies a complete Machine Learning workflow to predict which jobs are most at risk of automation based on their characteristics.

The goal is not only to predict risk, but to understand which features (salary, industry, AI impact level, education) are the strongest indicators of automation vulnerability — providing actionable insight to workers, companies, and policymakers.

---

## 2. Problem Statement

**What is the problem?**
Given a set of job characteristics (title, industry, salary, required education, AI impact level, etc.), can we accurately classify how likely that job is to be automated?

**Why is it important?**
- Over 85 million jobs may be displaced by automation by 2025 (World Economic Forum).
- Workers need advance warning to upskill before their role is eliminated.
- Companies need to plan which roles to retain, retrain, or automate.

**Who benefits?**
- Job seekers evaluating career security
- HR departments planning workforce transitions
- Educators designing future-proof curricula
- Policymakers designing job protection programs

**Input / Output:**
- **Input:** Job features — title, industry, salary, education, experience, remote ratio, AI impact level, location
- **Output:** Risk grade from 0 (Very Safe) to 4 (Critical Risk)

**Problem Type:** Multi-class Classification (5 classes)

---

## 3. Dataset Description

| Property | Value |
|---|---|
| Source | Kaggle — AI in the Job Market Dataset |
| Total Rows | 30,000 |
| Total Columns | 13 |
| Target Column | `Automation Risk (%)` → converted to `Risk_Grade` (0–4) |
| Missing Values | None |
| Duplicate Rows | None |

### Feature List

| Column | Type | Description |
|---|---|---|
| Job Title | Categorical | Name of the job role |
| Industry | Categorical | Sector (IT, Healthcare, Finance, etc.) |
| Job Status | Categorical | Employment trend (Increasing / Decreasing / Stable) |
| AI Impact Level | Categorical | Qualitative AI impact (Low / Moderate / High / Very High) |
| Median Salary (USD) | Numerical | Annual median salary |
| Required Education | Categorical | Minimum education level |
| Experience Required (Years) | Numerical | Years of experience needed |
| Job Openings (2024) | Numerical | Current job openings |
| Projected Openings (2030) | Numerical | Projected openings by 2030 |
| Remote Work Ratio (%) | Numerical | Percentage of remote work allowed |
| Automation Risk (%) | Numerical | **Target** — raw automation probability (0–100) |
| Location | Categorical | Country (USA, UK, Canada, Australia) |
| Gender Diversity (%) | Numerical | Proportion of gender diversity in the role |

---

## 4. Data Exploration (EDA)

### 4.1 Dataset Shape
- 30,000 rows and 13 columns
- 6 categorical columns (text) and 7 numerical columns

### 4.2 Data Types
Six columns are `object` type — these require Label Encoding before model training. Seven columns are `float64` — these require StandardScaler to normalize their scales.

### 4.3 Missing Values
No missing values were found in any column. The dataset is complete and does not require imputation.

### 4.4 Duplicate Rows
No duplicate rows were found. Every record is unique.

### 4.5 Basic Statistics
- `Automation Risk (%)` ranges from near 0% to near 100%, with a mean of approximately 50% — confirming the dataset covers the full risk spectrum.
- `Median Salary (USD)` shows high variance (large standard deviation) — scaling is essential to prevent salary from dominating smaller-scale features like experience.
- `Experience Required (Years)` spans 0–20+ years.

### 4.6 Target Column Analysis
The target `Automation Risk (%)` is a continuous value between 0 and 100. We convert it to 5 discrete classes:

| Grade | Label | Risk Range |
|---|---|---|
| 0 | Very Safe | 0–20% |
| 1 | Safe | 20–40% |
| 2 | Moderate | 40–60% |
| 3 | High Risk | 60–80% |
| 4 | Critical Risk | 80–100% |

Each class contains approximately 20% of the dataset (6,000 jobs), confirming balanced class distribution.

### 4.7 Problems Identified
1. Text columns cannot be directly used by ML algorithms — encoding required.
2. Numerical columns have very different scales — normalization required.
3. The target is continuous but classification requires discrete labels — target engineering required.

---

## 5. Data Visualization

Six visualizations were produced, each with an interpretation:

### Chart 1: Distribution of Automation Risk (%)
A histogram with KDE curve showing the spread of raw automation risk scores.
- **Finding:** The distribution is roughly uniform across 0–100%, confirming balanced classes after discretization.

### Chart 2: Class Distribution of Risk Grades (Pie Chart)
Shows the proportion of jobs in each of the 5 risk categories after target engineering.
- **Finding:** All 5 classes are approximately equal (~20% each), confirming no class imbalance issue.

### Chart 3: Median Salary by Risk Grade (Boxplot)
Compares salary distributions across risk categories.
- **Finding:** Higher-salary jobs tend to fall in lower-risk categories, suggesting that expertise (which drives higher pay) protects against automation.

### Chart 4: Top 10 Industries by Job Count (Bar Chart)
Shows which industries have the most job records.
- **Finding:** The dataset is diverse across many sectors — the model learns across industries rather than being biased to one.

### Chart 5: Remote Work Ratio vs Automation Risk (Scatter Plot)
Explores whether remote-friendly jobs tend to be safer from automation.
- **Finding:** No strong linear relationship — remote work and automation risk are largely independent. Non-linear patterns may still exist.

### Chart 6: Correlation Heatmap (Numerical Features)
Shows pairwise correlations between all numerical columns.
- **Finding:** No single feature is perfectly correlated with the target, meaning the model must learn from combinations of features. Some numerical features are weakly correlated with each other, which is acceptable.

---

## 6. Data Preprocessing

Five preprocessing steps were applied:

### Step 1: Remove Missing Values
Used `df.dropna()` to remove any rows with missing values. This is a safety measure — the EDA confirmed no rows were actually removed.

### Step 2: Target Engineering
Converted `Automation Risk (%)` from a continuous value to 5 discrete risk grades (0–4). The original column was then dropped to prevent data leakage — if kept, the model would simply compute the grade directly rather than learning from features.

### Step 3: Label Encoding
Applied `LabelEncoder` to all 6 categorical text columns. Each unique text value is converted to a unique integer (e.g., `"IT" → 0`, `"Healthcare" → 1`). The fitted encoders are saved for use in the prediction demo.

### Step 4: Feature Scaling (StandardScaler)
Applied `StandardScaler` to all 6 numerical columns. After scaling, each column has mean ≈ 0 and standard deviation ≈ 1. This is critical for:
- **Neural Networks:** They are extremely sensitive to feature scale and fail to converge without normalization.
- **Naive Bayes:** Assumes features follow a normal distribution centered around zero.

### Step 5: Train-Test Split
Split the dataset into 80% training (24,000 samples) and 20% testing (6,000 samples) using `random_state=42` for reproducibility. The model is trained on the training set and evaluated only on the test set, which it never sees during training.

---

## 7. Algorithms Used

### 7.1 Decision Tree Classifier
A tree-based model that recursively partitions the feature space into if-then rules. At each node, it finds the feature and threshold that best separates the classes. Easy to visualize and interpret, but prone to overfitting without constraints.

### 7.2 Naive Bayes (GaussianNB)
A probabilistic model based on Bayes' Theorem. It assumes all features are independent of each other and normally distributed. Very fast and requires no hyperparameter tuning, making it a good baseline. The independence assumption is violated in real data, which limits its accuracy.

### 7.3 Random Forest Classifier
An ensemble method that builds many decision trees (n=100 in our case), each on a random subset of data and features. The final prediction is the majority vote across all trees. This "wisdom of the crowd" approach dramatically reduces variance and overfitting compared to a single tree.

### 7.4 Neural Network (MLPClassifier)
A Multi-Layer Perceptron with two hidden layers (100 and 50 neurons). It learns non-linear feature combinations through forward propagation and adjusts weights through backpropagation. Requires scaled input features to converge correctly. Captures complex patterns that simpler models miss.

---

## 8. Model Comparison

All four models were trained on the same 80% training set and evaluated on the same 20% test set:

| Rank | Model | Test Accuracy (%) |
|---|---|---|
| 1 | Random Forest | Highest |
| 2 | Neural Network (MLP) | Second |
| 3 | Decision Tree | Third |
| 4 | Naive Bayes | Lowest |

**Advantages and Disadvantages:**

| Model | Advantages | Disadvantages |
|---|---|---|
| Random Forest | High accuracy, resistant to overfitting, handles mixed data well | Slower to train, less interpretable than single trees |
| Neural Network | Captures complex non-linear patterns | Requires scaling, longer training, sensitive to hyperparameters |
| Decision Tree | Easy to interpret and visualize | Overfits easily without constraints |
| Naive Bayes | Very fast, simple, no tuning needed | Assumes feature independence — unrealistic in practice |

**Best Model:** Random Forest — it achieves the highest test accuracy and generalizes well due to the ensemble averaging effect.

---

## 9. Model Improvement Experiments

### Experiment: Decision Tree Hyperparameter Tuning

**Observation from Baseline:**
The unconstrained Decision Tree achieved very high training accuracy (often ~100%) but significantly lower test accuracy. This large gap confirmed overfitting — the tree memorized training samples including noise.

**Change Applied:**
- Added `max_depth=10` — limits the tree to 10 levels, forcing it to learn general rules rather than memorizing details.
- Added `min_samples_split=10` — a node needs at least 10 samples before it can split, preventing decisions on tiny subsets.

**Result:**
- Training accuracy decreased slightly (expected — the model can no longer memorize everything).
- Test accuracy increased.
- Overfitting gap decreased significantly.

**Conclusion:** Hyperparameter tuning directly improves generalization. The model that scores highest on test data — not training data — is the correct goal for real-world deployment.

---

## 10. Evaluation Results

The best model (Random Forest) was evaluated with full classification metrics:

**Metrics Used:**
- **Accuracy:** Overall percentage of correct predictions
- **Precision:** Of predictions for class X, what fraction were correct?
- **Recall:** Of actual class X samples, what fraction were identified?
- **F1-score:** Harmonic mean of precision and recall — best metric when both matter
- **Confusion Matrix:** Full breakdown of correct vs. incorrect predictions per class

**Key findings from confusion matrix:**
- The diagonal (correct predictions) dominates — the model performs well overall.
- Most errors occur between **neighboring classes** (e.g., Moderate vs Safe, or Moderate vs High Risk). This is expected: jobs at class boundaries have genuinely ambiguous features.
- Errors between distant classes (Very Safe vs Critical) are rare, confirming the model has learned the correct ordering of risk.

---

## 11. Prediction Demo

A `predict_job_risk()` function was built that accepts all job features as inputs and returns the risk grade and label. It uses the same `LabelEncoder` objects and `StandardScaler` that were fitted during preprocessing — ensuring consistency between training and inference.

**Example predictions:**
- A Financial Planner with high salary and growing job openings → predicted lower risk.
- A Legal Secretary with high AI impact and declining projected openings → predicted higher risk.

The live interactive prediction demo is available in the Streamlit web application at: `https://xoa-ml.streamlit.app/`

---

## 12. Discussion

### Which model gave the best result?
**Random Forest** consistently achieved the highest test accuracy across all experiments.

### Why did it perform better?
The ensemble averaging of 100 trees corrects the weaknesses of any single tree. Individual trees overfit to noise; when averaged, the noise cancels out and only the true signal remains.

### What problems did we face?
1. Designing meaningful class boundaries for the continuous target variable.
2. High-cardinality categorical columns (Job Title has hundreds of unique values) — LabelEncoder introduces a false ordinal relationship.
3. Neural Network convergence failure without feature scaling.
4. Decision Tree overfitting requiring hyperparameter correction.

### What are the limitations?
- The dataset appears synthetic — real-world automation risk depends on geopolitical, regulatory, and company-level factors not captured here.
- LabelEncoder for categorical features is an approximation; One-Hot Encoding would be more theoretically correct.
- Hyperparameters were not exhaustively tuned — GridSearchCV could yield further improvements.

### How can it be improved in the future?
- Apply GridSearchCV for systematic hyperparameter tuning.
- Replace LabelEncoder with One-Hot Encoding for categorical columns.
- Use 5-fold cross-validation for more reliable performance estimates.
- Try XGBoost or LightGBM — gradient boosting methods frequently outperform Random Forest on tabular data.
- Collect real-world job market data to validate and extend findings.

---

## 13. Conclusion

This project successfully applied the full data science workflow to a real-world problem: predicting job automation risk. We collected a dataset of 30,000 jobs, performed thorough exploratory analysis, applied appropriate preprocessing, trained and compared 4 machine learning algorithms, improved the best-performing ones through hyperparameter tuning, and deployed the final model as an interactive Streamlit web application.

**Random Forest** emerged as the best model, balancing high accuracy with strong generalization. The project demonstrates that combining multiple algorithms and interpreting their results is far more valuable than relying on any single model.

---

## 14. References

1. World Economic Forum. *The Future of Jobs Report 2020.* https://www.weforum.org/reports/the-future-of-jobs-report-2020
2. Kaggle. *AI in the Job Market Dataset.* https://www.kaggle.com/
3. Scikit-learn Documentation. *RandomForestClassifier, DecisionTreeClassifier, GaussianNB, MLPClassifier.* https://scikit-learn.org/
4. Pedregosa, F. et al. *Scikit-learn: Machine Learning in Python.* JMLR 12, pp. 2825–2830, 2011.
5. Streamlit. *Open-source Python framework for data apps.* https://streamlit.io/
6. Pandas Documentation. https://pandas.pydata.org/docs/
7. Seaborn Documentation. https://seaborn.pydata.org/
