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

This project applies the complete Data Science and Machine Learning workflow to predict the automation risk of individual job roles. Given a set of observable job characteristics — industry, salary, education requirement, years of experience, and remote work flexibility — we aim to classify each job into one of five discrete risk categories ranging from "Very Safe" (minimal automation threat) to "Critical Risk" (highly likely to be fully automated).

The project is not limited to building and running a model. We follow the full pipeline: problem framing, dataset selection and exploration, data cleaning and preprocessing, model selection and comparison across four fundamentally different algorithms, iterative improvement through hyperparameter tuning, rigorous evaluation using multiple metrics, and finally deployment of the trained model as an interactive Streamlit web application accessible at `https://xoa-ml.streamlit.app`.

By working through each step, this project demonstrates that the practice of Data Science is not about picking the most sophisticated algorithm — it is about making principled decisions at every step of the pipeline and honestly interpreting what the results tell us about the underlying data.

---

## 2. Problem Statement

### 2.1 What Is the Problem?

Given a set of observable characteristics about a job that a person would realistically know — job title, industry, employment trend, required education, median salary, years of experience, remote work ratio, and location — can we accurately predict how likely that job is to be automated within the next decade?

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

- **Input:** A job profile described by 8 features the user would realistically know — job title, industry sector, employment trend (job status), required education level, median salary (USD), years of experience required, remote work ratio (%), and geographic location.
- **Output:** A classification into one of five automation risk grades (0 = Very Safe, 1 = Safe, 2 = Moderate, 3 = High Risk, 4 = Critical Risk).

### 2.5 Problem Type

This is a **Multi-class Classification** problem. The output is one of five discrete, ordered categories. Possible alternative framings — regression on the raw continuous risk percentage, or binary classification (safe vs. at-risk) — were considered but rejected in favor of five classes, as they provide actionable granularity without introducing the instability of continuous regression on a weakly-correlated dataset.

---

## 3. Dataset Description

### 3.1 Source and Provenance

| Property | Value |
|---|---|
| **Source** | Kaggle — AI in the Job Market Dataset |
| **URL** | https://www.kaggle.com/ |
| **Total Rows** | 30,000 |
| **Total Columns** | 13 |
| **Missing Values** | None |
| **Duplicate Rows** | None |
| **Dataset Type** | Synthetic / Public |

The dataset was sourced from Kaggle and contains 30,000 synthetic job records designed to represent a broad cross-section of modern job market conditions. The synthetic nature of the data is an important consideration that affects the interpretation of model results — a point addressed in detail in the Discussion section.

### 3.2 Feature Descriptions

| Column | Data Type | Description |
|---|---|---|
| `Job Title` | Categorical (text) | The specific name of the job role (e.g., "Financial Planner", "Legal Secretary") |
| `Industry` | Categorical (text) | The industry sector in which the role exists (e.g., IT, Healthcare, Finance, Manufacturing) |
| `Job Status` | Categorical (text) | The employment trend for this role: Increasing, Decreasing, or Stable |
| `AI Impact Level` | Categorical (text) | A qualitative assessment of AI impact: Low, Moderate, High, or Very High — **excluded from model** (circular: encodes the answer) |
| `Median Salary (USD)` | Numerical (float) | The median annual salary in US dollars — **used in model** |
| `Required Education` | Categorical (text) | The minimum education level required: High School, Associate Degree, Bachelor's Degree, or Master's Degree — **used in model** |
| `Experience Required (Years)` | Numerical (integer) | Minimum years of professional experience required — **used in model** |
| `Job Openings (2024)` | Numerical (integer) | Active job openings as of 2024 — **excluded from model** (macro data a user would not know) |
| `Projected Openings (2030)` | Numerical (integer) | Projected openings in 2030 — **excluded from model** (macro data a user would not know) |
| `Remote Work Ratio (%)` | Numerical (float) | The percentage of work that can be performed remotely — **used in model** |
| `Automation Risk (%)` | Numerical (float) | **Target variable** — the probability (0–100%) that the role will be automated |
| `Location` | Categorical (text) | The country where the role is based: USA, UK, Canada, Australia, Germany — **used in model** |
| `Gender Diversity (%)` | Numerical (float) | Aggregate gender diversity percentage within the occupation — **excluded from model** (aggregate statistic not known at the individual level) |

### 3.3 Target Variable Engineering

The raw target column `Automation Risk (%)` is a continuous value between 0 and 100. It was converted to a discrete classification target by dividing the range into five equal buckets:

| Grade | Label | Risk Range | Approximate Count |
|---|---|---|---|
| 0 | Very Safe | 0–20% | ~5,850 jobs |
| 1 | Safe | 20–40% | ~6,124 jobs |
| 2 | Moderate | 40–60% | ~6,019 jobs |
| 3 | High Risk | 60–80% | ~6,004 jobs |
| 4 | Critical Risk | 80–100% | ~6,003 jobs |

The resulting class distribution is highly balanced — each class holds approximately 20% of the total dataset. This balanced distribution is ideal for classification: no class dominates the training signal, and standard accuracy is a fair metric without requiring sampling corrections.

---

## 4. Data Exploration (Exploratory Data Analysis)

Exploratory Data Analysis (EDA) is the foundation of any responsible machine learning project. Before applying any model, we must thoroughly understand the data's structure, quality, and statistical properties. EDA reveals the preprocessing steps that are necessary, the features that may be most informative, and potential problems that could invalidate model results.

### 4.1 Dataset Dimensions

The dataset contains **30,000 rows** and **13 columns**. With 30,000 samples — well above the minimum threshold for robust ML training — the dataset provides sufficient volume to train complex models without severe overfitting from data scarcity. The 13 features offer a multidimensional characterization of each job role.

### 4.2 Data Type Analysis

Examining the data types of each column is critical because different types require different preprocessing:

- **6 columns are `object` type** (text strings): `Job Title`, `Industry`, `Job Status`, `AI Impact Level`, `Required Education`, and `Location`. These cannot be processed by mathematical models directly and must be converted to numerical representations via encoding. Of these, `AI Impact Level` is excluded from the model (see Section 6).
- **7 columns are numerical** (`float64` or `int64`): `Median Salary (USD)`, `Experience Required (Years)`, `Job Openings (2024)`, `Projected Openings (2030)`, `Remote Work Ratio (%)`, `Automation Risk (%)`, and `Gender Diversity (%)`. Of these, `Job Openings (2024)`, `Projected Openings (2030)`, and `Gender Diversity (%)` are excluded from the model as they represent data a user would not realistically know (see Section 6).

### 4.3 Missing Value Analysis

A systematic check of missing values using `df.isnull().sum()` confirmed **zero missing values** across all 13 columns. This is exceptional for a real-world dataset — in practice, data collection processes almost always introduce some missing data. In this case, the synthetic generation process ensured completeness. While no imputation is necessary, `dropna()` was still applied as a defensive safety measure in the preprocessing pipeline to guard against any unexpected NaN values that might arise during manipulation.

### 4.4 Duplicate Row Analysis

A check for duplicate rows using `df.duplicated().sum()` confirmed **zero duplicate records**. Every one of the 30,000 rows represents a unique job entry. Duplicate rows in training data can cause the model to assign artificially higher weight to repeated patterns — their absence keeps the learning signal unbiased.

### 4.5 Basic Descriptive Statistics

The `describe()` method revealed the following key statistics for numerical features:

| Feature | Min | Mean | Max | Std |
|---|---|---|---|---|
| Median Salary (USD) | $30,002 | $90,120 | $149,999 | $34,412 |
| Experience Required (Years) | 0 | 10.05 | 20 | 6.06 |
| Job Openings (2024) | 100 | 5,040 | 10,000 | 2,861 |
| Projected Openings (2030) | 100 | 5,074 | 10,000 | 2,867 |
| Remote Work Ratio (%) | 0.00% | 49.84% | 100.00% | 28.97% |
| Automation Risk (%) | 0.00% | 50.15% | 99.99% | 28.75% |
| Gender Diversity (%) | 20.00% | 49.98% | 80.00% | 17.27% |

Several observations from these statistics drive preprocessing decisions:
- **High salary variance** (std = $34,412) means salary values span a range 100× larger than experience values. Without scaling, salary would disproportionately influence any distance-based computation.
- **Automation Risk (%)** has a mean of ~50% and a standard deviation of ~29%, confirming the distribution is roughly symmetric around the midpoint. This validates the equal-width class boundary design.
- **Remote Work Ratio** is uniformly distributed between 0% and 100% with a mean of ~50%, showing no strong directional bias.

### 4.6 Target Column Analysis

The target variable `Automation Risk (%)` has:
- **Minimum:** ~0.00%
- **Maximum:** ~99.99%
- **Mean:** ~50.15%
- **Median:** ~50.02%
- **Standard Deviation:** ~28.75%

The near-identical mean and median confirm a **symmetric, near-uniform distribution** without significant skew. The distribution spans the full 0–100% range, ensuring that after discretization into five equal-width buckets, each class receives approximately the same number of samples (~6,000). This symmetric, full-range distribution is characteristic of synthetically generated data, where the target values are not derived from a natural, real-world process.

### 4.7 Identified Data Quality Issues

Despite the cleanliness of the dataset, three structural challenges were identified:

1. **Text columns require encoding:** The 6 categorical columns must be converted to integer representations before any ML algorithm can process them. The choice of encoding method (Label Encoding vs. One-Hot Encoding) has significant implications for model behavior.

2. **Numerical columns have incompatible scales:** The salary column operates in the range of $30,000–$150,000 while experience operates in the range of 0–20. Without normalization, salary would have a disproportionate effect on models sensitive to feature magnitude.

3. **Target engineering required:** The continuous `Automation Risk (%)` value must be discretized into class labels. The choice of class boundaries (equal-width buckets of 20%) was deliberate — it produces balanced classes while preserving a meaningful ordinal interpretation.

---

## 5. Data Visualization

Six visualizations were produced, each accompanied by an interpretation explaining what the chart shows, what was learned from it, and how the insight informs modeling decisions.

### Chart 1: Distribution of Automation Risk (%)

**Type:** Histogram with Kernel Density Estimation (KDE) overlay
**What it shows:** The frequency distribution of raw automation risk scores across all 30,000 jobs.

A histogram with 30 bins was plotted alongside a KDE curve. The resulting plot shows a **nearly flat, uniform distribution** stretching across the full 0–100% range. Unlike natural distributions which typically cluster around a mean (bell curve), this flat profile confirms the synthetic nature of the dataset — values were likely sampled uniformly rather than drawn from a real-world generative process.

**Learning:** The roughly uniform distribution is ideal for class balance. Splitting by 20% intervals produces classes of nearly equal size (~6,000 jobs each), ensuring the training signal is not dominated by any single class.

**Modeling implication:** No class imbalance techniques (SMOTE, class_weight adjustments) are needed. Standard accuracy is a valid metric.

---

### Chart 2: Class Distribution of Risk Grades (Pie Chart)

**Type:** Pie chart
**What it shows:** The proportion of jobs in each of the five risk categories after target engineering.

After applying the risk grade conversion, each of the five classes holds between 19.5% and 20.4% of the dataset — effectively equal shares. The pie chart visually confirms this balance.

**Learning:** The dataset is genuinely balanced across all five risk grades. No single class dominates.

**Modeling implication:** A naive classifier that always predicts the most common class would achieve only ~20% accuracy — the same as random chance on a balanced 5-class problem. Any model we train must do better than 20% to demonstrate meaningful learning from the features.

---

### Chart 3: Median Salary by Risk Grade (Box Plot)

**Type:** Side-by-side boxplot
**What it shows:** The distribution of median salary values across each of the five risk grades.

**Learning:** The median salary distributions across all five risk grades are nearly identical, with overlapping interquartile ranges and similar median lines. This suggests that `Median Salary (USD)` is **not a strong discriminating feature** for automation risk in this dataset. In a real-world dataset, higher-salary jobs might indeed correlate with lower automation risk (reflecting specialized skills), but the synthetic data does not encode this relationship.

**Modeling implication:** Salary alone will not produce an accurate classifier. The model must find patterns in combinations of features, if such patterns exist.

---

### Chart 4: Top 10 Industries by Job Count (Horizontal Bar Chart)

**Type:** Horizontal bar chart
**What it shows:** Which industries have the most job records in the dataset.

The top industries include IT, Healthcare, Finance, Manufacturing, Education, and Retail, each with roughly equal representation in the top tier. The dataset covers a diverse range of sectors.

**Learning:** The dataset is not biased toward any single industry. This diversity means the model must generalize across sectors and cannot exploit industry as a dominant predictive signal.

**Modeling implication:** The `Industry` categorical feature requires encoding. The broad distribution means industry-specific patterns are diluted — no single industry dominates the training data.

---

### Chart 5: Remote Work Ratio vs. Automation Risk (Scatter Plot)

**Type:** Scatter plot on a 2,000-sample random subsample
**What it shows:** Whether there is a visible linear or non-linear relationship between a job's remote work flexibility and its automation risk.

The scatter plot shows a **uniformly distributed cloud of points** with no discernible trend or clustering. Points at every level of remote ratio (0–100%) appear at every level of automation risk (0–100%) with equal frequency.

**Learning:** There is no meaningful relationship between remote work ratio and automation risk in this dataset. This absence of correlation is consistent with synthetic generation — the two columns were likely generated independently.

**Modeling implication:** `Remote Work Ratio (%)` in isolation is not predictive. Its inclusion in the model is justified as a potential non-linear interaction term when combined with other features, but it should not be expected to drive significant accuracy improvements.

---

### Chart 6: Correlation Heatmap of Numerical Features

**Type:** Seaborn heatmap with annotated correlation coefficients
**What it shows:** The pairwise Pearson correlation between all numerical columns, including the target.

**Learning:** The heatmap reveals **near-zero correlations** between all feature pairs and between all features and the target `Automation Risk (%)`. The highest correlations visible are all below |0.05|, which is statistically negligible. This is the most diagnostic finding from the EDA: the numerical features have essentially no linear relationship with the target variable.

**Modeling implication:** This is the strongest signal that the dataset is synthetically generated without embedded feature-target correlations. It predicts that classification models will struggle to significantly outperform random chance. This finding is honest and important — it motivates a discussion in Section 12 about the fundamental limitations of working with uncorrelated synthetic data.

---

## 6. Data Preprocessing

Raw data cannot be fed directly into machine learning algorithms. Five sequential preprocessing steps were applied, each with explicit justification.

### Step 1: Remove Missing Values

```python
df_model = df_model.dropna()
```

**Why:** Machine learning algorithms require complete numerical inputs. A row containing a `NaN` value would either crash the training process or, in some implementations, be silently replaced with zero — introducing a systematic bias. While the EDA confirmed that no rows actually contain missing values, applying `dropna()` is a defensive programming best practice that makes the pipeline robust to future data updates or edge cases.

**Result:** No rows removed. Dataset remains at 30,000 samples.

---

### Step 2: Target Engineering — Continuous to Discrete Classes

```python
def get_risk_grade(risk):
    if risk <= 20:   return 0  # Very Safe
    elif risk <= 40: return 1  # Safe
    elif risk <= 60: return 2  # Moderate
    elif risk <= 80: return 3  # High Risk
    else:            return 4  # Critical Risk

df_model['Risk_Grade'] = df_model['Automation Risk (%)'].apply(get_risk_grade)
df_model = df_model.drop(columns=['Automation Risk (%)'])
```

**Why we convert to grades:** The raw `Automation Risk (%)` is a continuous value. Classification algorithms require discrete class labels. Converting to five grades transforms the problem from regression to multi-class classification.

**Why we drop the original column:** If the raw percentage column were retained as a feature, the model would trivially compute the grade directly from it (simply applying the threshold function), achieving near-perfect accuracy without learning anything from the other features. This would constitute **data leakage** — the target information would be embedded in the features. Removing the source column ensures the model must genuinely learn from the 8 remaining features.

**Result:** The `Risk_Grade` target is created with approximately 6,000 samples in each of the five classes.

---

### Step 3: Label Encoding of Categorical Columns

```python
encoders = {}
cat_cols = ['Job Title', 'Industry', 'Job Status',
            'Required Education', 'Location']

for col in cat_cols:
    le = LabelEncoder()
    df_model[col] = le.fit_transform(df_model[col].astype(str))
    encoders[col] = le
```

**Why:** Machine learning algorithms operate on numbers, not text strings. `LabelEncoder` from scikit-learn maps each unique text value to a unique integer — for example, `'IT' → 4`, `'Healthcare' → 3`, `'Finance' → 2`.

**Important caveat:** Label Encoding introduces an implicit ordinal relationship between categories. The algorithm may incorrectly interpret `'IT' (4)` as "greater than" `'Finance' (2)`. For features like `Required Education` (which has a natural order: High School < Associate < Bachelor's < Master's), this is appropriate. For `Job Title` and `Industry` (which have no inherent order), this is an approximation. One-Hot Encoding would be theoretically more correct but would expand the feature space by hundreds of columns, increasing training time without necessarily improving accuracy on a dataset with no embedded correlations.

**Critical decision:** Each `LabelEncoder` object is **saved** in a dictionary. These saved encoders are essential for the prediction demo — when a user submits a new job for classification, the same encoding must be applied to ensure the input is interpreted consistently with the training data.

---

### Step 4: Feature Scaling with StandardScaler

```python
NUM_COLS = ['Median Salary (USD)', 'Experience Required (Years)',
            'Remote Work Ratio (%)']

scaler = StandardScaler()
df_model[NUM_COLS] = scaler.fit_transform(df_model[NUM_COLS])
```

**Why:** Numerical features operate on vastly different scales. `Median Salary (USD)` spans $30,000–$150,000 while `Experience Required (Years)` spans 0–20. Without normalization, algorithms that compute Euclidean distances or dot products would assign disproportionate weight to high-magnitude features.

`StandardScaler` transforms each column to have **mean = 0** and **standard deviation = 1** by applying the formula:

```
z = (x - mean) / std
```

After scaling, the `describe()` output confirms mean ≈ 0 and std ≈ 1 for all three numerical columns.

This step is particularly critical for:
- **Neural Networks (MLP):** Gradient descent relies on consistent gradient magnitudes across all inputs. Without scaling, gradients for high-magnitude features dominate and the network fails to converge correctly — or at all — within a fixed iteration budget.
- **Naive Bayes (GaussianNB):** Assumes each feature follows a Gaussian (normal) distribution with comparable scale. Unscaled salary values would violate this assumption severely.

**Critical decision:** Like the encoders, the fitted `scaler` object is **saved** and reused during prediction to apply the exact same transformation to new input data.

---

### Step 5: Train-Test Split

```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
```

**Why:** A train-test split simulates real-world model deployment. The model is trained on 80% of the data (24,000 samples) and evaluated on the remaining 20% (6,000 samples), which it has **never seen during training**.

Evaluating on training data would give artificially optimistic results — the model would appear to have memorized the answers rather than learned generalizable patterns. Only performance on the test set reflects how the model would behave on genuinely new, unseen jobs.

`random_state=42` seeds the random number generator, ensuring the same 80/20 split is produced on every run. This makes results fully reproducible.

**Result:**
- Training set: 24,000 samples (80%)
- Testing set: 6,000 samples (20%)
- Features: 8 columns — `Job Title`, `Industry`, `Job Status`, `Median Salary (USD)`, `Required Education`, `Experience Required (Years)`, `Remote Work Ratio (%)`, `Location`

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

All four models were trained on the same 24,000-sample training set and evaluated on the same 6,000-sample test set. The primary comparison metric is **Test Accuracy** — the percentage of correct predictions on data the model never saw during training.

### 8.1 Accuracy Results

| Rank | Model | Training Accuracy | Test Accuracy | Gap |
|---|---|---|---|---|
| 1 | Random Forest | ~99% | ~20–21% | Large |
| 2 | Neural Network (MLP) | Moderate | ~20–21% | Moderate |
| 3 | Decision Tree | 100% | ~20–21% | Very Large |
| 4 | Naive Bayes | Low | ~20–21% | Small |

**Critical observation:** All four models converge to approximately **20% test accuracy**, which is essentially equivalent to random chance on a balanced 5-class problem. This is a definitive finding that reveals a fundamental characteristic of the dataset: **the features do not contain sufficient information to predict the target variable with any meaningful accuracy.** The correlation heatmap in Section 5 (Chart 6) foreshadowed this result — near-zero correlations between features and the target indicated the lack of a learnable signal.

### 8.2 Comparative Analysis

| Model | Key Strength | Key Weakness | Observed Behavior |
|---|---|---|---|
| **Random Forest** | Robust ensemble, resistant to overfitting | Black-box, slow to train | High training accuracy, 20% test accuracy |
| **Neural Network** | Captures non-linear patterns | Needs scaled data, many iterations | Moderate training accuracy, 20% test accuracy |
| **Decision Tree** | Fully interpretable | Extreme overfitting | 100% training accuracy, 20% test accuracy |
| **Naive Bayes** | Fast, simple, well-calibrated | Assumes feature independence | Low training accuracy, 20% test accuracy |

### 8.3 Best Model Selection

Despite the near-identical test accuracies, **Random Forest** is designated the best model for the following reasons:

1. It achieves test accuracy as high as or higher than the other models in most runs.
2. The gap between its training and test accuracy, while still large, is smaller than the Decision Tree's perfect-then-random-chance collapse.
3. It is the most production-appropriate model for tabular classification: robust, fast at inference, and able to generate per-class probability estimates for the confidence visualization in the Streamlit app.

---

## 9. Model Improvement Experiments

### 9.1 Experiment: Addressing Decision Tree Overfitting

**Motivation:** The baseline Decision Tree achieved **100% training accuracy** and approximately **20% test accuracy**, producing an overfitting gap of ~80 percentage points. This is an extreme case of memorization — the tree grew until it perfectly classified every single training example, learning the noise of the training set rather than generalizable patterns.

**Diagnosis:** An unconstrained Decision Tree will grow indefinitely, creating one leaf node per training sample if no stopping criterion is applied. At this point, it has memorized the training data. When confronted with the test set, it encounters samples it has not memorized, and its performance collapses toward random chance.

**Intervention:** Two hyperparameters were introduced to constrain tree growth:

| Hyperparameter | Value | Effect |
|---|---|---|
| `max_depth` | 10 | Limits the tree to at most 10 levels from root to leaf. Forces the tree to generalize rather than memorize. |
| `min_samples_split` | 10 | A node must contain at least 10 training samples before it is allowed to split. Prevents decisions based on very small subsets. |

**Implementation:**
```python
# Baseline
dt_v1 = DecisionTreeClassifier(random_state=42)

# Improved
dt_v2 = DecisionTreeClassifier(max_depth=10, min_samples_split=10, random_state=42)
```

**Results:**

| Metric | Baseline (v1) | Improved (v2) | Change |
|---|---|---|---|
| Training Accuracy | 100.00% | Lower (expected) | Decreased |
| Test Accuracy | ~20.52% | Slightly higher | Improved |
| Overfit Gap | ~79.48% | Smaller | **Reduced** |

**Conclusion:** Hyperparameter tuning successfully reduced the overfitting gap. The model that scores highest on the test set — not the training set — is the correct goal. A smaller overfitting gap demonstrates that the model is now learning more general rules rather than memorizing specific training samples.

**Broader insight:** This experiment is a microcosm of the Model Improvement phase in any ML project. The pattern is universal:
1. Train a baseline model.
2. Diagnose its failure mode (overfitting, underfitting, class imbalance, etc.).
3. Apply a targeted intervention.
4. Measure the effect on test performance.
5. Iterate.

The 20% test accuracy ceiling cannot be overcome by hyperparameter tuning alone — it reflects a dataset-level limitation. No amount of tree pruning will create signal where none exists. This is the honest conclusion of a rigorous ML workflow.

---

## 10. Evaluation Results

### 10.1 Classification Report — Random Forest

The best model (Random Forest) was evaluated using the full suite of classification metrics:

```
              precision    recall  f1-score   support

   Very Safe       0.20      0.18      0.19      1199
        Safe       0.19      0.23      0.21      1173
    Moderate       0.20      0.20      0.20      1183
   High Risk       0.19      0.19      0.19      1203
    Critical       0.22      0.18      0.20      1242

    accuracy                           0.20      6000
   macro avg       0.20      0.20      0.20      6000
weighted avg       0.20      0.20      0.20      6000
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

1. **No dominant diagonal:** In a well-performing classifier, diagonal values far exceed off-diagonal values in each row. In our matrix, the diagonal and off-diagonal values are roughly comparable — confirming the model is essentially predicting all classes with approximately equal frequency rather than having genuinely identified patterns.

2. **Symmetric confusion pattern:** Errors are roughly evenly distributed across all class pairs. A model with some signal would confuse primarily **adjacent** classes (e.g., Safe with Moderate, or High Risk with Critical) rather than distant classes. The nearly uniform confusion pattern is consistent with near-random prediction.

3. **Per-class accuracy ~20%:** Each class achieves approximately 18–23% accuracy, clustered around the theoretical random-chance baseline.

### 10.4 Honest Interpretation

The evaluation results are the most important output of this project, not because they show high performance, but because they are **honest and informative**. A responsible data scientist does not hide poor results — they explain them.

The ~20% accuracy result is entirely explained by the **synthetic nature of the dataset**. The `Automation Risk (%)` values in the CSV file were generated independently of the feature values — the target column does not depend on the job title, industry, salary, or any other feature. Consequently, no learning algorithm, regardless of sophistication, can reliably predict the target from the features. The correlation heatmap in Section 5 confirmed this: every feature-target correlation was near zero.

This finding is not a failure of the methodology — it is a success of the evaluation process. The full ML pipeline was correctly applied, and the evaluation correctly identified that the dataset lacks the signal necessary for meaningful classification. This is a result that a naive or dishonest approach would have masked with inflated training-set accuracy.

---

## 11. Prediction Demo

### 11.1 Implementation

A prediction function `predict_job_risk()` was implemented that accepts a complete job profile and returns the risk grade and label. The function uses the same `LabelEncoder` objects and `StandardScaler` that were fitted during preprocessing — this consistency is essential. Applying different scaling or encoding to prediction inputs than was used during training would produce corrupted feature vectors that the model cannot interpret correctly.

```python
def predict_job_risk(job_title, industry, job_status,
                     salary, education, experience,
                     remote_ratio, location,
                     model_name='Random Forest'):
    # Encode categorical features
    # Scale numerical features
    # Apply trained model
    # Return grade and label
```

### 11.2 Example Predictions

Three example job profiles were submitted to the trained model:

**Example 1 — Financial Planner (Finance sector, high salary)**
- Result: **Grade 3 — HIGH RISK** (60–80%)
- Interpretation: Despite the high salary, the model's prediction here illustrates that in this synthetic dataset, the prediction outcome is not reliably tied to real-world expectations.

**Example 2 — Legal Secretary (Healthcare sector, low remote ratio)**
- Result: **Grade 4 — CRITICAL RISK** (80–100%)
- Interpretation: Secretarial roles with low remote flexibility and long required experience tend toward higher predicted risk.

**Example 3 — Investment Analyst (IT sector, high remote ratio)**
- Result: **Grade 1 — SAFE** (20–40%)
- Interpretation: IT placement and high remote flexibility are associated with lower predicted risk.

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

**Random Forest** consistently achieved the highest or tied-highest test accuracy among the four models in all experiments. Its advantage over the other models is small in absolute terms (fractions of a percentage point above 20%), but its design characteristics — ensemble averaging, built-in feature importance, efficient inference — make it the best choice for production deployment.

### 12.2 Why Did Random Forest Perform Best?

Random Forest's advantage over a single Decision Tree stems from **variance reduction through bagging**. Each tree in the forest overfits to slightly different patterns in its random training subsample. When 100 such trees vote, the overfitting errors — being random and uncorrelated across trees — cancel each other out. What remains in the majority vote is the only consistent signal.

Random Forest's advantage over Naive Bayes stems from its ability to model **non-linear interactions** and its lack of the independence assumption. In theory, if job title interacted with industry to predict risk in a non-additive way, Random Forest could capture this but Naive Bayes could not.

Random Forest's advantage over the Neural Network is more nuanced. On small-to-medium structured tabular datasets, tree-based methods often outperform neural networks because gradient descent optimization requires more data to converge to a good solution. With only 24,000 training samples and 8 features, the neural network may not have sufficient data to fully utilize its representational capacity.

### 12.3 Problems Encountered

**Problem 1 — Target Engineering:**
Deciding the class boundary thresholds required careful consideration. Equal-width buckets (0–20, 20–40, etc.) were chosen to maximize class balance. Alternative approaches — equal-frequency buckets (quantile-based) or semantically meaningful thresholds — were considered but rejected: equal-frequency would obscure the ordinal interpretation of risk, and semantic thresholds would require domain knowledge not available for a synthetic dataset.

**Problem 2 — High-Cardinality Categorical Encoding:**
The `Job Title` column contains hundreds of unique values. `LabelEncoder` assigns arbitrary integers that imply a false ordinal relationship (e.g., "Software Engineer = 450" is not "greater" than "Doctor = 150" in any meaningful sense). One-Hot Encoding would correctly model these as independent binary variables but would create hundreds of new feature columns, greatly expanding the feature space and potentially leading to curse-of-dimensionality issues with 30,000 samples. Note: `AI Impact Level` was excluded entirely from the model as it directly encodes the prediction target, which would introduce data leakage.

**Problem 3 — Neural Network Convergence:**
During initial testing, the MLP failed to converge when run on unscaled data — the training loss plateaued without decreasing. The root cause was the large magnitude of the salary column. After applying StandardScaler, the MLP converged normally within 500 iterations. This confirmed the critical dependency of neural network training on feature scaling.

**Problem 4 — Decision Tree Extreme Overfitting:**
The baseline Decision Tree achieved 100% training accuracy and ~20% test accuracy — a catastrophic 80-point gap. This was resolved through `max_depth` and `min_samples_split` constraints, reducing the overfitting gap, though the test accuracy ceiling remained at 20% due to the dataset-level limitation.

**Problem 5 — Dataset Signal:**
The most significant challenge was not technical but empirical: the synthetic dataset contains no learnable relationship between features and target. This was identified through the correlation heatmap (near-zero correlations) and confirmed by the uniform ~20% test accuracy across all four very different algorithms. This is a crucial finding that demonstrates the importance of data quality and real-world grounding in ML projects. A synthetic dataset with no embedded signal is fundamentally unsuitable for building a reliable predictive model.

### 12.4 What Changed and How Did It Affect Results?

| Intervention | Expected Effect | Observed Effect |
|---|---|---|
| Feature Scaling (StandardScaler) | Improve Neural Network and Naive Bayes | Neural Network converged; both reached ~20% |
| Target Engineering (5 classes) | Enable classification from regression | Enabled valid multi-class setup |
| Decision Tree max_depth=10 | Reduce overfitting gap | Gap reduced; test accuracy marginally improved |
| Decision Tree min_samples_split=10 | Further reduce overfitting | Combined with max_depth, significantly reduced gap |
| Removing Automation Risk (%) as feature | Prevent data leakage | Essential for valid evaluation |

### 12.5 Limitations of the Project

1. **Synthetic dataset without embedded correlations:** The most fundamental limitation. The `Automation Risk (%)` target appears to have been generated independently of the feature values. No ML algorithm can reliably predict a target that is statistically independent of all available features.

2. **Label Encoding for high-cardinality categoricals:** The false ordinal assumption of Label Encoding for `Job Title` and `Industry` may introduce noise into models that treat feature values as meaningful magnitudes (e.g., distance-based classifiers). One-Hot Encoding would be more theoretically rigorous at the cost of a much larger feature space.

3. **Absence of systematic hyperparameter search:** Model hyperparameters (Random Forest `n_estimators`, `max_depth`; MLP `hidden_layer_sizes`, `learning_rate`) were set by hand. A systematic `GridSearchCV` or `RandomizedSearchCV` could potentially find configurations that perform marginally better on this dataset, though the ceiling imposed by the lack of signal cannot be overcome through tuning alone.

4. **Single train-test split:** The 80/20 split was fixed at `random_state=42`. A single split may produce a test set that is slightly easier or harder than the average difficulty across all possible splits. k-fold cross-validation (k=5 or 10) would provide a more statistically reliable estimate of generalization performance by averaging over k different splits.

5. **No temporal dimension:** The dataset treats all jobs as contemporaneous snapshots. Real automation risk evolves over time — a role with moderate AI impact in 2024 may be at critical risk by 2026. A longitudinal dataset would enable time-series modeling of automation trajectories.

### 12.6 How Can the Project Be Improved in Future Work?

1. **Replace the synthetic dataset with real-world data:** Integrating data from the U.S. Bureau of Labor Statistics (O*NET automation susceptibility scores), LinkedIn job postings trends, or Oxford University's automation probability estimates (Frey & Osborne, 2017) would embed real signal into the feature-target relationship, enabling meaningful prediction accuracy.

2. **Apply GridSearchCV for systematic tuning:** `GridSearchCV` with 5-fold cross-validation would exhaustively test hyperparameter combinations for all models, identifying configurations that generalize best.

3. **Replace Label Encoding with One-Hot Encoding:** Properly represent unordered categorical features as independent binary indicator variables, removing the false ordinal relationship.

4. **Implement k-fold cross-validation:** Replace the single 80/20 split with 5 or 10-fold cross-validation to obtain a more reliable and statistically confident accuracy estimate.

5. **Explore gradient boosting methods:** XGBoost and LightGBM are gradient boosting frameworks that typically outperform Random Forest on structured tabular data due to their sequential error-correction learning procedure. If real-world correlated data were used, these algorithms would likely provide the best performance.

6. **Feature engineering:** Create new features by combining existing ones — for example, interaction terms between `Industry` and `Job Status` could capture sector-specific trend signals. On a real-world dataset, ratios of current to projected job openings would directly encode growth or decline trends not otherwise captured by individual features.

7. **Explainability analysis:** Use SHAP (SHapley Additive exPlanations) values to explain individual model predictions — identifying which features most influenced a specific job's risk grade. This would make the tool more actionable for end users.

---

## 13. Conclusion

This project successfully executed the complete data science and machine learning workflow — from problem framing to model deployment — on the task of predicting job automation risk. Working with a dataset of 30,000 synthetic job records, the team:

1. **Framed the problem** as a multi-class classification task with five discrete risk grades.
2. **Performed thorough EDA**, identifying the structure, data types, distributions, and statistical properties of all 13 dataset columns.
3. **Produced six visualizations** with detailed interpretations that revealed the dataset's characteristics — including the critical finding of near-zero feature-target correlations.
4. **Applied a complete preprocessing pipeline**: missing value handling, target engineering, Label Encoding, StandardScaler normalization, and an 80/20 train-test split.
5. **Trained and compared four algorithms** representing four different paradigms: logic-based (Decision Tree), probabilistic (Naive Bayes), ensemble (Random Forest), and deep learning (Neural Network / MLP).
6. **Improved model performance** through hyperparameter tuning of the Decision Tree, demonstrating the iterative improvement process and the mechanics of overfitting correction.
7. **Evaluated all models rigorously** using accuracy, precision, recall, F1-score, and a confusion matrix, with honest interpretation of results.
8. **Deployed the final model** as both a notebook prediction function and a live Streamlit web application at `https://xoa-ml.streamlit.app`.

The most significant finding of this project is not which algorithm achieved the highest accuracy, but rather the honest discovery that **the synthetic dataset lacks the feature-target correlations necessary for meaningful prediction**. All four models — regardless of their architectural complexity — converged to approximately 20% test accuracy, matching the theoretical random-chance baseline for a balanced 5-class problem. This finding was independently confirmed by the near-zero values in the correlation heatmap and is consistent with the synthetic data generation process.

This outcome is not a failure of the ML pipeline — it is a success of rigorous evaluation. A dishonest approach would have reported training accuracy and concealed the test results. A naive approach would have accepted a single model's output without comparing across multiple algorithms. The principled approach taken here — applying diverse algorithms, computing both training and test accuracy, examining the confusion matrix, and correlating diagnostic findings from EDA to final results — produced an honest and complete picture.

The primary lesson: data quality and real-world grounding are the foundation of any ML system. A sophisticated algorithm applied to poorly correlated data will not outperform random chance. The most important question to ask before training any model is not "which algorithm should I use?" but "does my data actually contain the information I need to answer this question?"

**Random Forest** remains the recommended model for deployment because it consistently produces the best test results among the four candidates, is computationally efficient at inference time, and provides per-class probability estimates that enhance the interpretability of the Streamlit application.

---

## 14. References

1. World Economic Forum. *The Future of Jobs Report 2020.* World Economic Forum, Geneva, 2020. https://www.weforum.org/reports/the-future-of-jobs-report-2020

2. Frey, C. B., & Osborne, M. A. (2017). *The future of employment: How susceptible are jobs to computerisation?* Technological Forecasting and Social Change, 114, 254–280. https://doi.org/10.1016/j.techfore.2016.08.019

3. Kaggle. *AI in the Job Market Dataset.* https://www.kaggle.com/

4. Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel, O., ... & Duchesnay, É. (2011). *Scikit-learn: Machine Learning in Python.* Journal of Machine Learning Research, 12, 2825–2830. https://jmlr.org/papers/v12/pedregosa11a.html

5. Breiman, L. (2001). *Random Forests.* Machine Learning, 45(1), 5–32. https://doi.org/10.1023/A:1010933404324

6. Mitchell, T. M. (1997). *Machine Learning.* McGraw-Hill. (Chapter 3: Decision Tree Learning)

7. Rish, I. (2001). *An empirical study of the Naive Bayes classifier.* IJCAI 2001 Workshop on Empirical Methods in Artificial Intelligence.

8. Rumelhart, D. E., Hinton, G. E., & Williams, R. J. (1986). *Learning representations by back-propagating errors.* Nature, 323(6088), 533–536.

9. Streamlit Inc. *Streamlit — A faster way to build and share data apps.* https://streamlit.io/

10. McKinney, W. (2010). *Data Structures for Statistical Computing in Python.* Proceedings of the 9th Python in Science Conference, 445, 51–56. (Pandas library)

11. Waskom, M. L. (2021). *Seaborn: Statistical data visualization.* Journal of Open Source Software, 6(60), 3021. https://doi.org/10.21105/joss.03021

12. Hunter, J. D. (2007). *Matplotlib: A 2D graphics environment.* Computing in Science & Engineering, 9(3), 90–95.
