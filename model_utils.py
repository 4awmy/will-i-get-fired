import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Define features and scaling columns globally or as constants so they can be imported
FEATURE_COLS = [
    'Job Title', 'Industry', 'Job Status',
    'Median Salary (USD)', 'Required Education', 'Experience Required (Years)',
    'Remote Work Ratio (%)', 'Location'
]

NUM_COLS_TO_SCALE = [
    'Median Salary (USD)', 'Experience Required (Years)',
    'Remote Work Ratio (%)'
]

@st.cache_data
def load_and_process_data(file_path):
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        return None, None, None, None, None

    # 1. Handle Missing Values
    df = df.dropna()

    # 2. Target Engineering
    if 'Automation Risk (%)' in df.columns:
        def get_risk_grade(risk):
            if risk <= 20: return 0  # Very Safe
            elif risk <= 40: return 1 # Safe
            elif risk <= 60: return 2 # Moderate
            elif risk <= 80: return 3 # High Risk
            else: return 4            # Critical Risk
        df['Risk_Grade'] = df['Automation Risk (%)'].apply(get_risk_grade)
    else:
        return None, None, None, None, None

    # 3. Encoding & Scaling
    encoders = {}
    cat_cols = ['Job Title', 'Industry', 'Job Status', 'Required Education', 'Location']

    # Store original DF for display, work on df_processed
    df_processed = df.copy()

    for col in cat_cols:
        if col in df_processed.columns:
            df_processed[col] = df_processed[col].astype(str)
            le = LabelEncoder()
            df_processed[col] = le.fit_transform(df_processed[col])
            encoders[col] = le

    final_features = [c for c in FEATURE_COLS if c in df_processed.columns]

    # Scaling Numerical Features
    scaler = StandardScaler()

    existing_num_cols = [c for c in NUM_COLS_TO_SCALE if c in final_features]
    if existing_num_cols:
        df_processed[existing_num_cols] = scaler.fit_transform(df_processed[existing_num_cols])

    return df, df_processed, final_features, encoders, scaler

@st.cache_resource
def train_models(X_train, y_train):
    """
    Trains multiple models and returns a dictionary of trained models.
    Cached to avoid retraining on every rerender.
    """
    models = {
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "Naive Bayes": GaussianNB(),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Neural Network (MLP)": MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42)
    }

    trained_models = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        trained_models[name] = model

    return trained_models
