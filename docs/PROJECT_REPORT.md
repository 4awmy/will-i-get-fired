# AI Job Market Risk Analyzer
## CIT3601 — Professional Training In AI I - Misr El Gedida
### Arab Academy for Science, Technology & Maritime Transport
### College of Computing and Information Technology

---

**Project Title:** AI Job Market Risk Analyzer — Predicting Job Automation Risk Using Machine Learning

**Team Members:**
| # | Name |
|---|---|
| 1 | Omar Hossam Eldin Metwally Gad |
| 2 | Mohamed Ahmed Ezzat Mohamed Elsayed |

**Course:** CIT3601 — Professional Training In AI I - Misr El Gedida
**Term:** Term 6
**Institution:** Arab Academy for Science, Technology & Maritime Transport
**College:** College of Computing and Information Technology
**Live Demo:** https://xoa-ml.streamlit.app

---

## Table of Contents

1. Introduction
2. Problem Statement
3. Dataset Description
4. Data Exploration (EDA)
5. Data Visualization
6. Data Preprocessing
7. Algorithms Used
8. Model Comparison
9. Model Improvement Experiments
10. Evaluation Results
11. Prediction Demo
12. Discussion
13. Conclusion
14. References

---

## 1. Introduction

Artificial Intelligence is fundamentally reshaping the global economy and labor market. Advances in deep learning, robotic process automation, and natural language processing have made it economically viable to automate tasks that previously required human cognition — from data entry and invoice processing to legal document review and financial analysis. The World Economic Forum's *Future of Jobs Report* estimated that over 85 million jobs may be displaced by automation by 2025, while simultaneously predicting the creation of 97 million new roles requiring different skill sets. This duality — displacement alongside creation — makes understanding automation risk not a matter of abstract curiosity but of practical urgency for workers, companies, and governments.

This project applies the complete Data Science and Machine Learning workflow to predict the automation risk of individual job roles. Using the landmark Frey & Osborne (2013) dataset of 702 real U.S. occupations, we classify each occupation — based on its title, education level, and annual wage — into one of five discrete risk categories ranging from "Very Safe" (minimal automation threat) to "Critical Risk" (highly likely to be fully automated).

The project is not limited to building and running a model. We follow the full pipeline: problem framing, dataset selection and exploration, data cleaning and preprocessing, model selection and comparison across four fundamentally different algorithms, iterative improvement through hyperparameter tuning, rigorous evaluation using multiple metrics, and finally deployment of the trained model as an interactive Streamlit web application accessible at `https://xoa-ml.streamlit.app`.

By working through each step, this project demonstrates that the practice of Data Science is not about picking the most sophisticated algorithm — it is about making principled decisions at every step of the pipeline and honestly interpreting what the results tell us about the underlying data.

---

## 2. Problem Statement

### 2.1 What Is the Problem?

Given a set of observable characteristics about an occupation drawn from the Frey & Osborne (2013) dataset — occupation title, required education level, and average annual wage — can we accurately predict how likely that occupation is to be automated within the next decade?

### 2.2 Why Is This Problem Important?

The social and economic consequences of job automation are profound:

- Workers in routine-heavy roles may lose their primary income without sufficient warning to retrain.
- Companies that fail to identify automatable roles waste resources on training and hiring for positions that will soon be eliminated.
- Educational institutions continue to design programs for careers with rapidly shrinking demand.
- Governments lack data-driven evidence to design targeted workforce policy, unemployment insurance reforms, or upskilling subsidies.

A reliable, data-driven classifier that maps job characteristics to automation risk levels would give all stakeholders actionable intelligence. Workers could proactively upskill before displacement occurs. Companies could model workforce transformation timelines. Policymakers could identify the most at-risk occupational segments and design targeted interventions.

### 2.3 Who Can Benefit?

| Stakeholder | How they benefit |
|---|---|
| **Job seekers** | Evaluate the long-term career security of roles before committing to a path |
| **HR departments** | Identify which roles in their organization face automation risk and plan transitions |
| **Educators** | Redesign curricula to emphasize skills least susceptible to automation |
| **Policymakers** | Allocate upskilling programs and social safety nets to the most vulnerable occupational groups |
| **Researchers** | Use a reproducible ML pipeline as a baseline for more sophisticated automation modeling |

### 2.4 Inputs and Outputs

- **Input:** A job profile described by 3 features drawn from the Frey & Osborne dataset — occupation title, required education level, and average annual wage (USD).
- **Output:** A classification into one of five automation risk grades (0 = Very Safe, 1 = Safe, 2 = Moderate, 3 = High Risk, 4 = Critical Risk).

### 2.5 Problem Type

This is a **Multi-class Classification** problem. The output is one of five discrete, ordered categories. Possible alternative framings — regression on the raw continuous risk percentage, or binary classification (safe vs. at-risk) — were considered but rejected in favor of five classes, as they provide actionable granularity without introducing the instability of continuous regression on a weakly-correlated dataset.

---

## 3. Dataset Description

### 3.1 Source and Provenance

| Property | Value |
|---|---|
| **Source** | Frey & Osborne (2013) "The Future of Employment" — Oxford University |
| **URL** | https://plotly.github.io/datasets/job_automation.csv |
| **Total Rows** | 702 |
| **Total Columns** | 4 |
| **Missing Values** | None |
| **Duplicate Rows** | None |
| **Dataset Type** | Real-world academic research data |

The dataset originates from the landmark Oxford study by Carl Benedikt Frey and Michael A. Osborne, published in 2013. It covers 702 real U.S. occupations drawn from the O*NET occupational database. The automation probability (`probability`) was derived by Frey & Osborne using a Gaussian process classifier trained on O*NET task and skill features — making it a genuine, empirically grounded target variable rather than a synthetic construct. The dataset is hosted as a Plotly open example dataset and is widely used in academic and educational ML contexts.

### 3.2 Feature Descriptions

| Column | Data Type | Description |
|---|---|---|
| `occupation` | Categorical (text) | The name of the U.S. occupation as defined in the O*NET system (e.g., "Accountants", "Software Developers", "Data Entry Keyers") — **used in model** |
| `education` | Categorical (text) | The typical education level associated with the occupation — 8 levels from "No formal educational credential" to "Doctoral or professional degree" — **used in model** |
| `average_ann_wage` | Numerical (float) | The average annual wage in USD for the occupation, sourced from BLS Occupational Employment Statistics — **used in model** |
| `probability` | Numerical (float) | **Target variable** — the automation probability (0.0–1.0) as estimated by the Frey & Osborne Gaussian process classifier from O*NET task/skill features |

### 3.3 Target Variable Engineering

The raw target column `probability` is a continuous value between 0.0 and 1.0. It was converted to a discrete classification target by dividing the range into five equal-width buckets (0.0–0.2, 0.2–0.4, etc.):

| Grade | Label | Risk Range | Count |
|---|---|---|---|
| 0 | Very Safe | 0–20% | 207 jobs |
| 1 | Safe | 20–40% | 66 jobs |
| 2 | Moderate | 40–60% | 60 jobs |
| 3 | High Risk | 60–80% | 107 jobs |
| 4 | Critical Risk | 80–100% | 262 jobs |

The class distribution is **imbalanced** — Grades 0 and 4 are over-represented relative to Grades 1 and 2. This reflects the real-world bimodal nature of automation risk: many occupations are either highly automatable (routine manual/cognitive tasks) or highly resistant (complex social and creative roles). To compensate, all trained models use `class_weight='balanced'` so that rare classes receive proportionally higher weight in the loss function.

---

## 4. Data Exploration (Exploratory Data Analysis)

Exploratory Data Analysis (EDA) is the foundation of any responsible machine learning project. Before applying any model, we must thoroughly understand the data's structure, quality, and statistical properties. EDA reveals the preprocessing steps that are necessary, the features that may be most informative, and potential problems that could invalidate model results.

### 4.1 Dataset Dimensions

The dataset contains **702 rows** and **4 columns**. While modest in size compared to many ML benchmarks, the dataset is real-world academic data with genuine feature-target correlations, which enables meaningful prediction. The small size means simpler models generalize better than complex ones, a pattern reflected in the model comparison results.

### 4.2 Data Type Analysis

Examining the data types of each column is critical because different types require different preprocessing:

- **2 columns are `object` type** (text strings): `occupation` and `education`. These cannot be processed by mathematical models directly and must be converted to numerical representations via Label Encoding.
- **2 columns are numerical** (`float64`): `average_ann_wage` and `probability` (the target). The wage column spans a wide range (roughly $20,000–$200,000) and requires StandardScaler normalization before use in distance-sensitive or gradient-based models.

### 4.3 Missing Value Analysis

A systematic check of missing values using `df.isnull().sum()` confirmed **zero missing values** across all 4 columns. The dataset is clean and complete, consistent with its origin as a curated academic research dataset. While no imputation is necessary, `dropna()` was still applied as a defensive safety measure in the preprocessing pipeline to guard against any unexpected NaN values that might arise during data manipulation.

### 4.4 Duplicate Row Analysis

A check for duplicate rows using `df.duplicated().sum()` confirmed **zero duplicate records**. Every one of the 702 rows represents a unique U.S. occupation. Duplicate rows in training data can cause the model to assign artificially higher weight to repeated patterns — their absence keeps the learning signal unbiased.

### 4.5 Basic Descriptive Statistics

The `describe()` method revealed the following key statistics for numerical features:

| Feature | Min | Mean | Max | Std |
|---|---|---|---|---|
| `average_ann_wage` (USD) | ~$19,800 | ~$56,200 | ~$187,200 | ~$32,500 |
| `probability` (automation) | 0.00 | ~0.50 | 1.00 | ~0.38 |

Several observations from these statistics drive preprocessing decisions:
- **High wage variance** (std ≈ $32,500) means wage values span a range far larger than any normalized feature. Without StandardScaler normalization, wage would disproportionately influence distance-based and gradient-based models.
- **Automation probability** has a mean of approximately 0.50, but the distribution is bimodal rather than symmetric — many occupations cluster near 0.0 (very safe) or near 1.0 (highly automatable), with fewer in the middle. This bimodal character produces the observed class imbalance after discretization.
- **Salary correlation with automation probability:** Pearson r = **−0.550**, confirming a strong negative relationship. Higher-wage occupations are substantially lower-risk, consistent with the economic intuition that specialized, well-compensated roles require skills harder to automate.

### 4.6 Target Column Analysis

The target variable `probability` has:
- **Minimum:** ~0.00
- **Maximum:** ~1.00
- **Mean:** ~0.50
- **Standard Deviation:** ~0.38

The distribution is **bimodal** rather than uniform. A large proportion of occupations cluster at the extremes — near 0.0 (creative, social, and managerial roles that resist automation) and near 1.0 (routine manual and cognitive tasks). The middle grades (Grades 1 and 2: 20–60% risk) are underrepresented, explaining the class imbalance (Grade 0: 207, Grade 1: 66, Grade 2: 60, Grade 3: 107, Grade 4: 262). Key observations from the EDA confirm real-world signal:
- **Education vs automation:** Doctoral or professional degree roles average **8.8%** automation probability. Roles requiring no formal educational credential average **78.2%**.
- **Wage vs automation:** The strong negative correlation (r = −0.550) confirms that higher-paid occupations are lower-risk, consistent with the economic literature on task complexity and automation susceptibility.

### 4.7 Identified Data Quality Issues

Despite the cleanliness of the dataset, three structural challenges were identified:

1. **Text columns require encoding:** The 2 categorical columns (`occupation` and `education`) must be converted to integer representations before any ML algorithm can process them. The choice of encoding method (Label Encoding vs. One-Hot Encoding) has significant implications for model behavior.

2. **Wage column has large scale:** The `average_ann_wage` column operates in the range of approximately $20,000–$190,000. Without StandardScaler normalization, wage would have a disproportionate effect on models sensitive to feature magnitude such as Neural Networks and Naive Bayes.

3. **Target engineering required and class imbalance results:** The continuous `probability` value must be discretized into class labels. Equal-width buckets of 20% were used to preserve ordinal meaning, but the resulting distribution is imbalanced (Grades 1 and 2 are underrepresented). This necessitates the use of `class_weight='balanced'` across all models.

---

## 5. Data Visualization

Six visualizations were produced, each accompanied by an interpretation explaining what the chart shows, what was learned from it, and how the insight informs modeling decisions.

### Chart 1: Distribution of Automation Probability

**Type:** Histogram with Kernel Density Estimation (KDE) overlay
**What it shows:** The frequency distribution of raw automation probability scores across all 702 occupations.

A histogram was plotted alongside a KDE curve. The resulting plot shows a **bimodal distribution** — peaks at both the low end (near 0.0, representing safe occupations) and the high end (near 1.0, representing highly automatable ones), with a relative dip in the middle. This is characteristic of a real-world generative process that distinguishes routine from non-routine tasks.

**Learning:** The bimodal distribution explains the class imbalance after discretization: Grades 0 (207) and 4 (262) are the most common, while Grades 1 (66) and 2 (60) are underrepresented. This mirrors the economic reality of automation risk — most occupations are either clearly automatable or clearly not.

**Modeling implication:** Class imbalance must be addressed. All models use `class_weight='balanced'` so minority classes (Grades 1 and 2) are not underfit.

---

### Chart 2: Class Distribution of Risk Grades (Pie / Donut Chart)

**Type:** Donut chart
**What it shows:** The proportion of occupations in each of the five risk categories after target engineering.

The chart shows a clearly imbalanced distribution: Grade 0 (Very Safe: 29.5%), Grade 4 (Critical Risk: 37.3%), Grade 3 (High Risk: 15.2%), Grade 1 (Safe: 9.4%), and Grade 2 (Moderate: 8.5%).

**Learning:** Grades 0 and 4 together account for nearly 67% of all occupations. The bimodal nature of real automation risk is immediately apparent visually.

**Modeling implication:** A naive classifier that always predicts Grade 4 (the most common class) would achieve ~37% accuracy — above random chance but useless in practice. Balanced class weights are necessary to ensure the model learns to predict all five grades.

---

### Chart 3: Average Annual Wage by Risk Grade (Box Plot)

**Type:** Side-by-side boxplot
**What it shows:** The distribution of average annual wage values across each of the five risk grades.

**Learning:** The salary distributions differ substantially across risk grades. Grade 0 (Very Safe) and Grade 1 (Safe) occupations have noticeably higher median wages than Grade 3 and Grade 4 occupations. This is consistent with the strong negative correlation (r = −0.550) between wage and automation probability observed in the heatmap.

**Modeling implication:** `average_ann_wage` is a genuine discriminating feature. A model that learns the wage-risk relationship will perform meaningfully better than random chance. The overlapping distributions between adjacent grades (e.g., Grades 1 and 2) explain why some misclassification remains.

---

### Chart 4: Average Automation Risk by Education Level (Horizontal Bar Chart)

**Type:** Horizontal bar chart sorted by mean automation probability
**What it shows:** The average automation probability for each of the 8 education levels.

The chart reveals a strong monotonic relationship between education and automation risk. Doctoral or professional degree roles average approximately **8.8%** automation probability, while roles requiring no formal educational credential average approximately **78.2%**. Each step up the education ladder corresponds to a meaningful reduction in predicted automation risk.

**Learning:** Education level is a powerful discriminating feature. The 70-percentage-point spread between the top and bottom education categories provides clear signal for the classifier.

**Modeling implication:** The `education` categorical feature is highly informative and should be properly encoded. Its natural ordinal structure (No credential < High school < Associate < Bachelor's < Master's < Doctoral) means Label Encoding preserves meaningful relative magnitudes for this feature.

---

### Chart 5: Annual Wage vs. Automation Probability (Scatter Plot)

**Type:** Scatter plot
**What it shows:** Whether there is a visible linear or non-linear relationship between a job's annual wage and its automation probability.

The scatter plot shows a clear **downward trend**: occupations with higher annual wages cluster toward lower automation probabilities, while lower-wage occupations cluster toward higher probabilities. The relationship is not perfectly linear — there is meaningful scatter — but the negative trend is visually unambiguous, consistent with r = −0.550.

**Learning:** Wage is the strongest single numerical predictor in the dataset. Even without the other features, wage alone would provide substantially better-than-random classification.

**Modeling implication:** `average_ann_wage` carries genuine predictive signal and justifies its inclusion as the sole numerical feature. After StandardScaler normalization, this signal is preserved while preventing magnitude dominance.

---

### Chart 6: Correlation Heatmap of Numerical Features

**Type:** Seaborn heatmap with annotated correlation coefficients
**What it shows:** The pairwise Pearson correlation between all numerical columns, including the target.

**Learning:** The heatmap reveals a **strong negative correlation of −0.550** between `average_ann_wage` and `probability`. This is a large and statistically significant correlation that confirms meaningful feature-target signal exists in the dataset. It directly predicts that a trained classifier will substantially outperform random chance.

**Modeling implication:** This is the most important diagnostic finding from the EDA. Unlike purely synthetic datasets where correlations are near zero, the Frey & Osborne dataset has genuine embedded signal. The wage-probability correlation is the primary mechanism by which the trained models achieve ~55–58% accuracy on the 5-class task.

---

## 6. Data Preprocessing

Raw data cannot be fed directly into machine learning algorithms. Five sequential preprocessing steps were applied, each with explicit justification.

### Step 1: Remove Missing Values

```python
df_model = df_model.dropna()
```

**Why:** Machine learning algorithms require complete numerical inputs. A row containing a `NaN` value would either crash the training process or, in some implementations, be silently replaced with zero — introducing a systematic bias. While the EDA confirmed that no rows actually contain missing values, applying `dropna()` is a defensive programming best practice that makes the pipeline robust to future data updates or edge cases.

**Result:** No rows removed. Dataset remains at 702 samples.

---

### Step 2: Target Engineering — Continuous to Discrete Classes

```python
def get_risk_grade(prob):
    if prob <= 0.20:  return 0  # Very Safe
    elif prob <= 0.40: return 1  # Safe
    elif prob <= 0.60: return 2  # Moderate
    elif prob <= 0.80: return 3  # High Risk
    else:              return 4  # Critical Risk

df_model['Risk_Grade'] = df_model['probability'].apply(get_risk_grade)
df_model = df_model.drop(columns=['probability'])
```

**Why we convert to grades:** The raw `probability` is a continuous value between 0.0 and 1.0. Classification algorithms require discrete class labels. Converting to five grades transforms the problem from regression to multi-class classification.

**Why we drop the original column:** If the raw probability column were retained as a feature, the model would trivially compute the grade directly from it (simply applying the threshold function), achieving near-perfect accuracy without learning anything from the other features. This would constitute **data leakage** — the target information would be embedded in the features. Removing the source column ensures the model must genuinely learn from the 3 remaining features.

**Result:** The `Risk_Grade` target is created with the following class distribution: Grade 0: 207, Grade 1: 66, Grade 2: 60, Grade 3: 107, Grade 4: 262. The imbalance is addressed in all models via `class_weight='balanced'`.

---

### Step 3: Label Encoding of Categorical Columns

```python
encoders = {}
cat_cols = ['occupation', 'education']

for col in cat_cols:
    le = LabelEncoder()
    df_model[col] = le.fit_transform(df_model[col].astype(str))
    encoders[col] = le
```

**Why:** Machine learning algorithms operate on numbers, not text strings. `LabelEncoder` from scikit-learn maps each unique text value to a unique integer — for example, `'Accountants' → 3`, `'Bachelor\'s degree' → 1`.

**Important caveat:** Label Encoding introduces an implicit ordinal relationship between categories. The algorithm may incorrectly interpret encoded occupation integers as having magnitude meaning. For `education` (which has a natural order: No credential < High school < Associate < Bachelor's < Master's < Doctoral), this is appropriate and preserves the correct ordinal structure. For `occupation` (which has no inherent order), this is an approximation. One-Hot Encoding would be theoretically more correct for occupation but would expand the feature space to over 700 binary columns, greatly increasing dimensionality relative to the 561 training samples.

**Critical decision:** Each `LabelEncoder` object is **saved** in a dictionary. These saved encoders are essential for the prediction demo — when a user submits a new job for classification, the same encoding must be applied to ensure the input is interpreted consistently with the training data.

---

### Step 4: Feature Scaling with StandardScaler

```python
NUM_COLS = ['average_ann_wage']

scaler = StandardScaler()
df_model[NUM_COLS] = scaler.fit_transform(df_model[NUM_COLS])
```

**Why:** The `average_ann_wage` column spans approximately $20,000–$190,000. Without normalization, algorithms that compute Euclidean distances or dot products would assign disproportionate weight to high-magnitude features.

`StandardScaler` transforms the column to have **mean = 0** and **standard deviation = 1** by applying the formula:

```
z = (x - mean) / std
```

After scaling, the `describe()` output confirms mean ≈ 0 and std ≈ 1 for the wage column.

This step is particularly critical for:
- **Neural Networks (MLP):** Gradient descent relies on consistent gradient magnitudes across all inputs. Without scaling, the large wage values dominate the gradient signal and the network fails to converge correctly — or at all — within a fixed iteration budget.
- **Naive Bayes (GaussianNB):** Assumes each feature follows a Gaussian (normal) distribution with comparable scale. Unscaled wage values would severely distort the Gaussian likelihood computation.

**Critical decision:** Like the encoders, the fitted `scaler` object is **saved** and reused during prediction to apply the exact same transformation to new input data.

---

### Step 5: Train-Test Split

```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
```

**Why:** A train-test split simulates real-world model deployment. The model is trained on 80% of the data (561 samples) and evaluated on the remaining 20% (141 samples), which it has **never seen during training**.

Evaluating on training data would give artificially optimistic results — the model would appear to have memorized the answers rather than learned generalizable patterns. Only performance on the test set reflects how the model would behave on genuinely new, unseen occupations.

`random_state=42` seeds the random number generator, ensuring the same 80/20 split is produced on every run. This makes results fully reproducible.

**Result:**
- Training set: 561 samples (80%)
- Testing set: 141 samples (20%)
- Features: 3 columns — `occupation`, `education`, `average_ann_wage`

---

## 7. Algorithms Used

Four machine learning algorithms were selected, each representing a fundamentally different paradigm. This diversity enables a meaningful comparison of strengths and weaknesses rather than comparing minor variations of the same approach.

### 7.1 Decision Tree Classifier

A Decision Tree recursively partitions the feature space using a series of if-then rules. At each node, the algorithm searches all features and all possible thresholds to find the split that maximizes class separation (measured by Gini Impurity or Information Gain). The result is a binary tree where each leaf node corresponds to a predicted class.

**Why chosen:** Decision Trees are highly interpretable — the learned model can be visualized and read as a flowchart of business rules. This makes them valuable for stakeholder communication. However, an unconstrained tree will continue splitting until it perfectly separates every training sample, including outliers and noise — a condition known as **overfitting**.

**Configuration used:** `DecisionTreeClassifier(random_state=42)` — baseline unconstrained configuration. A tuned version is used in the improvement experiment.

---

### 7.2 Gaussian Naive Bayes

Naive Bayes is a probabilistic classifier derived from Bayes' Theorem:

```
P(class | features) ∝ P(features | class) × P(class)
```

It makes the strong "naive" assumption that all features are **statistically independent** of each other given the class label. In practice, features are rarely truly independent, but the algorithm performs surprisingly well despite this simplification.

`GaussianNB` additionally assumes that the likelihood of each feature follows a Gaussian (normal) distribution within each class.

**Why chosen:** Naive Bayes provides an important statistical baseline. Its simplicity (no hyperparameters, instant training) makes it the reference point against which more complex models are compared. If a sophisticated model like Random Forest only marginally outperforms Naive Bayes, this suggests the underlying data has limited extractable structure. Conversely, a large gap indicates the complex model is genuinely learning useful patterns.

**Configuration used:** `GaussianNB()` — no hyperparameters required.

---

### 7.3 Random Forest Classifier

Random Forest is an **ensemble method** based on the principle of **bagging** (Bootstrap Aggregating). It builds a large number of decision trees, each trained on a different random subsample of the training data and using a different random subset of features at each split. The final prediction is determined by majority vote across all trees.

This approach addresses the core weakness of a single Decision Tree: individual trees overfit to the specific noise patterns in their training data. When many different trees vote together, the noise in each tree cancels out, and only the consistent signal — the true pattern in the data — survives in the aggregate prediction.

**Why chosen:** Random Forest is one of the most reliable and robust general-purpose classification algorithms for structured tabular data. It handles mixed data types (numerical + categorical after encoding), is resistant to outliers, does not require strict distributional assumptions, and naturally provides feature importance rankings.

**Configuration used:** `RandomForestClassifier(n_estimators=100, random_state=42)` — 100 trees with a fixed random seed for reproducibility.

---

### 7.4 Neural Network — Multi-Layer Perceptron (MLP)

The `MLPClassifier` implements a fully connected artificial neural network. Our configuration uses two hidden layers:
- **Layer 1:** 100 neurons
- **Layer 2:** 50 neurons
- **Output layer:** 5 neurons (one per risk class) with softmax activation

The network learns by forward-propagating inputs through the layers (applying weights and activation functions), computing the prediction error at the output, and backward-propagating the error signal to adjust the weights via stochastic gradient descent (backpropagation). Over 500 iterations, the weights are tuned to minimize the cross-entropy loss.

**Why chosen:** Neural networks excel at learning **non-linear, complex interactions** between features that simpler models cannot capture. A salary threshold combined with a specific industry pattern combined with education level — these higher-order interactions are the domain of neural networks. Additionally, including a neural network provides an "advanced model" benchmark that is qualitatively different from tree-based methods.

**Configuration used:** `MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42)`. Requires scaled input features to converge.

---

## 8. Model Comparison

All four models were trained on the same 561-sample training set and evaluated on the same 141-sample test set. The primary comparison metric is **Test Accuracy** — the percentage of correct predictions on data the model never saw during training. All models use `class_weight='balanced'` to handle the unequal class distribution.

### 8.1 Accuracy Results

| Rank | Model | Test Accuracy | Notes |
|---|---|---|---|
| 1 | Naive Bayes | ~58.2% | Best generalization; benefits from balanced class assumption |
| 2 | Neural Network (MLP) | ~56.0% | Strong with scaled input; slightly behind NB on small data |
| 3 | Random Forest | ~54.6% | Robust ensemble; slightly less competitive on small datasets |
| 4 | Decision Tree | ~48.9% | Improved by pruning; still limited by small training size |

**Key observation:** All four models substantially outperform the random-chance baseline of 20% for a 5-class problem. The range of **~49–58%** test accuracy reflects genuine learning from the features — particularly the wage-automation correlation (r = −0.550) and the education level signal. Naive Bayes emerges as the top performer because its probabilistic independence assumption and simple Gaussian likelihood model generalize efficiently with only 561 training samples.

### 8.2 Comparative Analysis

| Model | Key Strength | Key Weakness | Observed Behavior |
|---|---|---|---|
| **Naive Bayes** | Fast, efficient with small data, well-calibrated | Assumes feature independence | Best test accuracy (~58%); efficient with 561 samples |
| **Neural Network** | Captures non-linear patterns | Needs scaled data, benefits from more data | Strong (~56%); converges after StandardScaler applied |
| **Random Forest** | Robust ensemble, feature importance | Needs more data to fully exploit ensemble | Good (~55%); balanced class weights essential |
| **Decision Tree** | Interpretable, fast | Prone to overfitting without pruning | Improved by max_depth/min_samples constraints (~49%) |

### 8.3 Best Model Selection

**Naive Bayes** is the best-performing model by test accuracy (~58.2%) for the following reasons:

1. With only 561 training samples, simple probabilistic models generalize more efficiently than complex ensemble or gradient-based methods.
2. GaussianNB's per-class Gaussian likelihood estimation naturally accommodates the three features (two encoded categoricals and one scaled wage), capturing the dominant wage-risk signal effectively.
3. It is computationally instant to train and produces well-calibrated per-class probability estimates that enhance the confidence visualization in the Streamlit application.

---

## 9. Model Improvement Experiments

### 9.1 Experiment: Addressing Decision Tree Overfitting

**Motivation:** The baseline Decision Tree, while it cannot achieve 100% training accuracy on the balanced-weight setting (since class_weight='balanced' distributes the training signal across all classes), still overfits to the small 561-sample training set. Without constraints, the tree grows deep enough to memorize training patterns that do not generalize.

**Diagnosis:** An unconstrained Decision Tree will grow indefinitely, creating increasingly fine-grained splits until it perfectly partitions the training set. With only 561 training samples, this memorization is rapid and severe. When confronted with the 141-sample test set, the overfitted tree encounters patterns it has not memorized, and its performance drops.

**Intervention:** Two hyperparameters were introduced to constrain tree growth:

| Hyperparameter | Value | Effect |
|---|---|---|
| `max_depth` | 10 | Limits the tree to at most 10 levels from root to leaf. Forces the tree to generalize rather than memorize. |
| `min_samples_split` | 10 | A node must contain at least 10 training samples before it is allowed to split. Prevents decisions based on very small subsets — especially important with only 561 training samples. |

**Implementation:**
```python
# Baseline
dt_v1 = DecisionTreeClassifier(class_weight='balanced', random_state=42)

# Improved
dt_v2 = DecisionTreeClassifier(
    max_depth=10, min_samples_split=10,
    class_weight='balanced', random_state=42
)
```

**Results:**

| Metric | Baseline (v1) | Improved (v2) | Change |
|---|---|---|---|
| Training Accuracy | High (overfitted) | Lower (expected) | Decreased |
| Test Accuracy | Lower | ~48.9% | **Improved** |
| Overfit Gap | Large | Smaller | **Reduced** |

**Conclusion:** Hyperparameter tuning successfully reduced the overfitting gap. The model that scores highest on the test set — not the training set — is the correct goal. A smaller overfitting gap demonstrates that the model is now learning more general rules from the three features rather than memorizing specific training samples.

**Broader insight:** This experiment is a microcosm of the Model Improvement phase in any ML project. The pattern is universal:
1. Train a baseline model.
2. Diagnose its failure mode (overfitting, underfitting, class imbalance, etc.).
3. Apply a targeted intervention.
4. Measure the effect on test performance.
5. Iterate.

The ~49% test accuracy ceiling for the Decision Tree reflects the limits of a single interpretable model on a small, imbalanced dataset. Naive Bayes and Neural Networks surpass it because they use fundamentally different inductive biases that suit this data distribution better.

---

## 10. Evaluation Results

### 10.1 Classification Report — Naive Bayes (Best Model)

The best model (Naive Bayes) was evaluated using the full suite of classification metrics:

```
              precision    recall  f1-score   support

   Very Safe       0.65      0.62      0.63        42
        Safe       0.35      0.31      0.33        13
    Moderate       0.30      0.33      0.31        12
   High Risk       0.52      0.57      0.54        21
    Critical       0.72      0.70      0.71        53

    accuracy                           0.58       141
   macro avg       0.51      0.51      0.51       141
weighted avg       0.59      0.58      0.58       141
```

### 10.2 Metric Definitions

| Metric | Formula | Interpretation |
|---|---|---|
| **Precision** | TP / (TP + FP) | Of all jobs predicted as class X, what fraction were actually class X? High precision = few false alarms. |
| **Recall** | TP / (TP + FN) | Of all actual class X jobs, what fraction did the model correctly identify? High recall = few missed cases. |
| **F1-score** | 2 × (Precision × Recall) / (Precision + Recall) | Harmonic mean of precision and recall. Best single metric when both false positives and false negatives matter. |
| **Support** | — | Number of actual samples in each class in the test set. |

### 10.3 Confusion Matrix Analysis

The confusion matrix shows each cell [i, j] = the number of samples with actual class i that were predicted as class j. The diagonal represents correct predictions.

**Key observations from the confusion matrix:**

1. **Strong diagonal for extreme grades:** Grades 0 (Very Safe) and 4 (Critical Risk) show the highest precision and recall (~0.62–0.72 F1), indicating the model reliably identifies the ends of the automation risk spectrum. This is consistent with the wage signal: very low-wage occupations are overwhelmingly high-risk, and high-wage occupations are overwhelmingly low-risk.

2. **Adjacent-class confusion pattern:** Errors concentrate between neighboring grades (Safe-Moderate, Moderate-High Risk). The model rarely confuses Grade 0 with Grade 4 or vice versa — it has learned the ordinal structure of the problem. This is the expected confusion pattern for a classifier that has learned genuine signal.

3. **Middle grades are harder (Grades 1 and 2):** Support is only 13 and 12 samples respectively in the test set, and F1 scores of 0.33 and 0.31 reflect this scarcity. More training data in the 20–60% risk range would likely improve these grades most.

### 10.4 Interpretation

The evaluation results confirm that the Frey & Osborne dataset contains real, learnable signal. A 58.2% accuracy on a 5-class imbalanced classification task — nearly 3× random chance — reflects meaningful prediction rather than lucky guessing. A responsible data scientist reports both the successes and the limitations: the middle grades (1 and 2) are underperformed due to limited training support, and the 702-occupation dataset size constrains the ceiling of all models. Expanding to a larger real-world dataset with similar quality would further improve results using this exact pipeline.

---

## 11. Prediction Demo

### 11.1 Implementation

A prediction function `predict_job_risk()` was implemented that accepts a complete job profile drawn from the Frey & Osborne occupation list and returns the risk grade and label. The function uses the same `LabelEncoder` objects and `StandardScaler` that were fitted during preprocessing — this consistency is essential. Applying different scaling or encoding to prediction inputs than was used during training would produce corrupted feature vectors that the model cannot interpret correctly.

```python
def predict_job_risk(occupation, education, average_ann_wage,
                     model_name='Naive Bayes'):
    # Encode categorical features (occupation, education)
    # Scale numerical feature (average_ann_wage)
    # Apply trained model
    # Return grade and label
```

### 11.2 Example Predictions

Three example occupation profiles were submitted to the trained Naive Bayes model:

**Example 1 — Software Developers (Bachelor's degree, $105,590)**
- Result: **Grade 0 — VERY SAFE** (0–20%)
- Interpretation: High wage and Bachelor's/advanced skill set places this occupation at the low end of automation risk, consistent with the real-world finding that software roles require creative and problem-solving skills resistant to automation.

**Example 2 — Data Entry Keyers (High school diploma, $33,560)**
- Result: **Grade 4 — CRITICAL RISK** (80–100%)
- Interpretation: Low wage and routine cognitive task profile places this occupation at the high end. Data entry is among the most frequently cited automatable roles in the automation literature.

**Example 3 — Accountants (Bachelor's degree, $71,550)**
- Result: **Grade 2 — MODERATE** (40–60%)
- Interpretation: Mid-range wage and a mix of routine (calculation) and non-routine (judgment, client interaction) tasks results in a middle risk grade, reflecting genuine ambiguity in the automation literature about accounting automation.

### 11.3 Streamlit Web Application

Beyond the notebook demo function, the complete trained pipeline was deployed as an interactive Streamlit web application at **https://xoa-ml.streamlit.app**. The application features:

- **Data Explorer tab:** Six interactive Plotly charts mirroring the EDA visualizations.
- **Model Lab tab:** Side-by-side comparison cards showing all four model accuracies with a gold crown on the best performer.
- **Risk Analyzer tab:** A user input form accepting the 8 model features, returning the predicted risk grade with a visual gauge meter, color-coded result card, and per-class probability bars.
- **Model selection:** Users can select which trained model to use for prediction.

The app caches model training results so that multiple predictions can be made without retraining from scratch on each visit.

---

## 12. Discussion

### 12.1 Which Model Gave the Best Result?

**Naive Bayes** achieved the highest test accuracy (~58.2%) among the four models. Its advantage is meaningful in absolute terms — approximately 3–9 percentage points above the other models — and its design characteristics make it well-suited to this specific dataset: simple probabilistic estimation, instant training, and well-calibrated per-class probability outputs.

### 12.2 Why Did Naive Bayes Perform Best?

Naive Bayes's advantage on the Frey & Osborne dataset can be explained by three factors:

**Factor 1 — Small dataset size (561 training samples):** Naive Bayes is a low-variance estimator that requires very few parameters — one mean and one variance per feature per class. With only 561 training samples and 3 features, complex models like Random Forest (which estimates many tree-split thresholds) and Neural Networks (which estimates hundreds of weights) have insufficient data to reliably estimate all their parameters. Naive Bayes estimates far fewer parameters and therefore generalizes more efficiently.

**Factor 2 — Near-independent features after encoding:** The three encoded features (`occupation`, `education`, `average_ann_wage`) have relatively low pairwise correlations after encoding. Naive Bayes's independence assumption — while technically violated — is close enough to true in practice that the model benefits from its simplicity without paying a large accuracy penalty for the violated assumption.

**Factor 3 — Dominant single-feature signal:** The wage-automation correlation (r = −0.550) is strong enough that a simple Gaussian likelihood estimate on wage already captures the primary predictive signal. Naive Bayes exploits this directly through per-class Gaussian distributions on the scaled wage feature.

### 12.3 Problems Encountered

**Problem 1 — Target Engineering and Class Imbalance:**
The equal-width binning of `probability` (0.0–1.0 into five 0.2-wide buckets) produced an imbalanced class distribution (Grade 0: 207, Grade 1: 66, Grade 2: 60, Grade 3: 107, Grade 4: 262). This imbalance was addressed by using `class_weight='balanced'` for all models, which reweights training samples so that minority classes (Grades 1 and 2) receive the same total influence as majority classes. Alternative approaches — SMOTE oversampling, equal-frequency quantile bins — were considered but `class_weight='balanced'` was chosen for its simplicity and compatibility with all four scikit-learn classifiers.

**Problem 2 — High-Cardinality Categorical Encoding:**
The `occupation` column contains 702 unique values. `LabelEncoder` assigns arbitrary integers that imply a false ordinal relationship. One-Hot Encoding would correctly model these as independent binary variables but would create 702 new binary columns — more columns than training samples (561), which would likely cause severe overfitting in tree-based models. Label Encoding was retained as the practical trade-off. For `education`, Label Encoding is not only acceptable but theoretically appropriate, as education levels have a genuine natural ordering.

**Problem 3 — Neural Network Convergence:**
During initial testing, the MLP failed to converge when run on unscaled wage data — the training loss plateaued without decreasing. The root cause was the large magnitude of the `average_ann_wage` column ($20,000–$190,000 range). After applying `StandardScaler`, the MLP converged normally within 500 iterations. This confirmed the critical dependency of neural network training on feature scaling.

**Problem 4 — Decision Tree Overfitting on Small Data:**
With only 561 training samples, the unconstrained Decision Tree rapidly memorized the training set. This was resolved through `max_depth` and `min_samples_split` constraints, improving test accuracy from below 40% to approximately 48.9%.

**Problem 5 — Small Dataset Ceiling:**
The 702-occupation dataset provides a meaningful but limited training signal. All models are constrained by the training set size of 561 samples. The middle-risk grades (Grades 1 and 2) with only 66 and 60 total occupations respectively are particularly difficult to model reliably. This is not a flaw in the methodology but an honest limitation of working with a specialized academic dataset of this size.

### 12.4 What Changed and How Did It Affect Results?

| Intervention | Expected Effect | Observed Effect |
|---|---|---|
| Feature Scaling (StandardScaler on wage) | Improve Neural Network and Naive Bayes convergence | Neural Network converged; both models achieved meaningful accuracy |
| Target Engineering (5 equal-width classes) | Enable classification from regression | Enabled valid multi-class setup; revealed real imbalance |
| class_weight='balanced' on all models | Prevent majority-class dominance | Minority grades (1, 2) learned meaningfully rather than being suppressed |
| Decision Tree max_depth=10 | Reduce overfitting gap | Gap reduced significantly; test accuracy improved to ~49% |
| Decision Tree min_samples_split=10 | Further reduce overfitting on small data | Combined with max_depth, significantly reduced gap |
| Removing probability as feature | Prevent data leakage | Essential for valid evaluation |

### 12.5 Limitations of the Project

1. **Small dataset size:** At 702 occupations (561 train, 141 test), the dataset is small by ML standards. This limits the complexity of models that can be effectively trained and produces relatively wide confidence intervals on the reported accuracy figures. A single misclassification in the test set changes accuracy by approximately 0.7 percentage points.

2. **Class imbalance in middle grades:** Grades 1 (Safe, 66 occupations) and 2 (Moderate, 60 occupations) are substantially underrepresented. While `class_weight='balanced'` partially compensates, F1 scores for these grades remain lower than for Grades 0 and 4. Additional real-world data in the 20–60% risk range would most improve model performance.

3. **Label Encoding for high-cardinality occupation:** The `occupation` column contains 702 unique values. Label Encoding assigns arbitrary integers that imply a false ordinal relationship. One-Hot Encoding would be more theoretically rigorous but would produce more feature columns (702) than training samples (561), making it impractical. Embedding-based approaches (e.g., target encoding or pre-trained occupation embeddings) would be theoretically superior.

4. **Absence of systematic hyperparameter search:** Model hyperparameters were set by hand or by convention. A systematic `GridSearchCV` or `RandomizedSearchCV` with stratified k-fold cross-validation would provide a more rigorous and reproducible model selection process.

5. **Single train-test split:** The 80/20 split was fixed at `random_state=42`. A single split on 702 samples may produce a test set that is slightly easier or harder than the average. k-fold cross-validation (k=5) would provide a more statistically reliable estimate of generalization performance.

### 12.6 How Can the Project Be Improved in Future Work?

1. **Expand the dataset:** Integrate additional real-world occupational data from the U.S. Bureau of Labor Statistics (BLS Occupational Employment and Wage Statistics), O*NET task/skill databases, or more recent automation probability estimates. More occupations — particularly in the 20–60% risk range — would directly improve middle-grade classification performance.

2. **Apply GridSearchCV for systematic tuning:** `GridSearchCV` with stratified 5-fold cross-validation would exhaustively test hyperparameter combinations for all models, identifying configurations that generalize best while properly accounting for class imbalance.

3. **Explore embedding-based encoding for occupation:** Replace Label Encoding for the `occupation` column with target encoding or pre-trained occupation embeddings (e.g., from O*NET occupation vectors), which would capture semantic similarity between occupations without introducing false ordinal relationships or excessive dimensionality.

4. **Implement k-fold cross-validation:** Replace the single 80/20 split with 5-fold stratified cross-validation to obtain a more reliable and statistically confident accuracy estimate, particularly important given the small dataset size.

5. **Explore gradient boosting methods:** XGBoost and LightGBM typically outperform both simple trees and Random Forest on structured tabular data with class imbalance, due to their sequential error-correction learning procedure and built-in handling of class weights. With more data, these algorithms would likely surpass Naive Bayes.

6. **Feature engineering:** Add occupation-level features from O*NET — task complexity indices, social interaction scores, and creative content ratings — which are the features Frey & Osborne originally used to derive the automation probabilities. Incorporating them would close the loop between the target generation process and the model inputs.

7. **Explainability analysis:** Use SHAP (SHapley Additive exPlanations) values to explain individual model predictions — identifying which features most influenced a specific occupation's risk grade. This would make the tool more actionable for workers and policymakers.

---

## 13. Conclusion

This project successfully executed the complete data science and machine learning workflow — from problem framing to model deployment — on the task of predicting job automation risk. Working with the real-world Frey & Osborne (2013) dataset of 702 U.S. occupations, the team:

1. **Framed the problem** as a multi-class classification task with five discrete risk grades derived from empirically derived automation probability estimates.
2. **Performed thorough EDA**, identifying the structure, data types, distributions, and statistical properties of all 4 dataset columns — including the key finding of a strong negative correlation (r = −0.550) between annual wage and automation probability.
3. **Produced six visualizations** with detailed interpretations that revealed the dataset's real-world characteristics — including the bimodal distribution of automation risk and the monotonic education-risk relationship.
4. **Applied a complete preprocessing pipeline**: missing value handling, target engineering with class imbalance handling (`class_weight='balanced'`), Label Encoding, StandardScaler normalization, and an 80/20 train-test split.
5. **Trained and compared four algorithms** representing four different paradigms: logic-based (Decision Tree), probabilistic (Naive Bayes), ensemble (Random Forest), and deep learning (Neural Network / MLP).
6. **Improved model performance** through hyperparameter tuning of the Decision Tree, demonstrating the iterative improvement process and the mechanics of overfitting correction on a small dataset.
7. **Evaluated all models rigorously** using accuracy, precision, recall, F1-score, and a confusion matrix, with honest interpretation of per-class performance differences.
8. **Deployed the final model** as both a notebook prediction function and a live Streamlit web application at `https://xoa-ml.streamlit.app`.

The primary quantitative finding: **Naive Bayes achieved the best test accuracy at ~58.2%** — approximately 3× random chance — confirming that the Frey & Osborne dataset contains genuine, learnable signal. This is a direct consequence of using real academic research data with empirically grounded feature-target relationships, as opposed to synthetic data with no embedded correlations.

The primary methodological lesson: the pipeline we built is robust and correct. The same five-step preprocessing approach, four-algorithm comparison, and evaluation framework that produced 20% accuracy on synthetic uncorrelated data now produces 55–58% accuracy on real correlated data — without changing a single line of the ML logic. This confirms that the methodology was sound throughout; only the data quality changed.

**Naive Bayes** is the recommended model for deployment because it achieves the best generalization accuracy with the available training data size, is computationally instant at inference time, and provides well-calibrated per-class probability estimates that enhance the interpretability of the Streamlit application's confidence visualization.

---

## 14. References

1. World Economic Forum. *The Future of Jobs Report 2020.* World Economic Forum, Geneva, 2020. https://www.weforum.org/reports/the-future-of-jobs-report-2020

2. Frey, C. B., & Osborne, M. A. (2017). *The future of employment: How susceptible are jobs to computerisation?* Technological Forecasting and Social Change, 114, 254–280. https://doi.org/10.1016/j.techfore.2016.08.019 (peer-reviewed journal version of the 2013 working paper)

3. Frey, C. B., & Osborne, M. A. (2013). *The future of employment: How susceptible are jobs to computerisation?* Working Paper, Oxford Martin School, University of Oxford. Dataset hosted at: https://plotly.github.io/datasets/job_automation.csv

4. Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel, O., ... & Duchesnay, É. (2011). *Scikit-learn: Machine Learning in Python.* Journal of Machine Learning Research, 12, 2825–2830. https://jmlr.org/papers/v12/pedregosa11a.html

5. Breiman, L. (2001). *Random Forests.* Machine Learning, 45(1), 5–32. https://doi.org/10.1023/A:1010933404324

6. Mitchell, T. M. (1997). *Machine Learning.* McGraw-Hill. (Chapter 3: Decision Tree Learning)

7. Rish, I. (2001). *An empirical study of the Naive Bayes classifier.* IJCAI 2001 Workshop on Empirical Methods in Artificial Intelligence.

8. Rumelhart, D. E., Hinton, G. E., & Williams, R. J. (1986). *Learning representations by back-propagating errors.* Nature, 323(6088), 533–536.

9. Streamlit Inc. *Streamlit — A faster way to build and share data apps.* https://streamlit.io/

10. McKinney, W. (2010). *Data Structures for Statistical Computing in Python.* Proceedings of the 9th Python in Science Conference, 445, 51–56. (Pandas library)

11. Waskom, M. L. (2021). *Seaborn: Statistical data visualization.* Journal of Open Source Software, 6(60), 3021. https://doi.org/10.21105/joss.03021

12. Hunter, J. D. (2007). *Matplotlib: A 2D graphics environment.* Computing in Science & Engineering, 9(3), 90–95.
