import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler

FEATURE_COLS = ['occupation', 'education', 'average_ann_wage']

NUM_COLS_TO_SCALE = ['average_ann_wage']


def get_risk_grade(prob):
    if prob <= 0.2:   return 0  # Very Safe
    elif prob <= 0.4: return 1  # Safe
    elif prob <= 0.6: return 2  # Moderate
    elif prob <= 0.8: return 3  # High Risk
    else:             return 4  # Critical Risk


@st.cache_data
def load_and_process_data(file_path):
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        return None, None, None, None, None

    df = df.dropna()

    if 'probability' not in df.columns:
        return None, None, None, None, None

    df['Risk_Grade'] = df['probability'].apply(get_risk_grade)

    encoders = {}
    cat_cols = ['occupation', 'education']
    df_processed = df.copy()

    for col in cat_cols:
        le = LabelEncoder()
        df_processed[col] = le.fit_transform(df_processed[col].astype(str))
        encoders[col] = le

    scaler = StandardScaler()
    df_processed[NUM_COLS_TO_SCALE] = scaler.fit_transform(df_processed[NUM_COLS_TO_SCALE])

    return df, df_processed, FEATURE_COLS, encoders, scaler


@st.cache_resource
def train_models(X_train, y_train):
    models = {
        "Random Forest":        RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42),
        "Naive Bayes":          GaussianNB(),
        "Decision Tree":        DecisionTreeClassifier(class_weight='balanced', random_state=42),
        "Neural Network (MLP)": MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42),
    }
    return {name: model.fit(X_train, y_train) for name, model in models.items()}
