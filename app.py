import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import (
    accuracy_score, roc_auc_score,
    precision_score, recall_score,
    f1_score, matthews_corrcoef,
    confusion_matrix, classification_report
)
import seaborn as sns
import matplotlib.pyplot as plt


st.set_page_config(
    page_title="Forest Cover Type Classifier",
    layout="wide"
)

st.title("Forest Cover Type Classification App")
st.write("Upload test dataset and evaluate different ML models.")

@st.cache_resource
def load_models():
    models_dict = {
        "Logistic Regression": joblib.load("models/logistic_regression.pkl"),
        "Decision Tree": joblib.load("models/decision_tree.pkl"),
        "KNN": joblib.load("models/knn.pkl"),
        "Naive Bayes": joblib.load("models/naive_bayes.pkl"),
        "Random Forest": joblib.load("models/random_forest.pkl"),
        "XGBoost": joblib.load("models/xgboost.pkl")
    }
    scaler_model = joblib.load("models/scaler.pkl")
    return models_dict, scaler_model

try:
    models, scaler = load_models()
except Exception as e:
    st.error("Model files not found. Please ensure .pkl files exist in 'models/' folder.")
    st.stop()


uploaded_file = st.file_uploader("Upload CSV Test File", type=["csv"])

if uploaded_file is not None:

    data = pd.read_csv(uploaded_file)

    data = data.dropna(how="all")
    data.columns = data.columns.str.strip()

    st.subheader("Preview of Uploaded Data")
    st.dataframe(data.head())

    if "Cover_Type" not in data.columns:
        st.error("CSV must contain 'Cover_Type' column.")
        st.stop()

    data = data.dropna(subset=["Cover_Type"])

    X = data.drop("Cover_Type", axis=1)
    y = data["Cover_Type"]

    y = pd.to_numeric(y, errors="coerce")

    valid_index = y.notna()
    X = X[valid_index]
    y = y[valid_index]

    if len(y) == 0:
        st.error("No valid rows found after cleaning.")
        st.stop()

    if y.min() == 1:
        y = y - 1

    try:
        X_scaled = scaler.transform(X)
    except Exception:
        st.error("Feature mismatch! Ensure uploaded CSV matches training feature structure.")
        st.stop()

    model_name = st.selectbox("Select Model", list(models.keys()))
    model = models[model_name]

    y_pred = model.predict(X_scaled)
    y_prob = model.predict_proba(X_scaled)

    accuracy = accuracy_score(y, y_pred)
    auc = roc_auc_score(y, y_prob, multi_class='ovr')
    precision = precision_score(y, y_pred, average='weighted')
    recall = recall_score(y, y_pred, average='weighted')
    f1 = f1_score(y, y_pred, average='weighted')
    mcc = matthews_corrcoef(y, y_pred)

    st.subheader("Evaluation Metrics")

    col1, col2, col3 = st.columns(3)
    col1.metric("Accuracy", round(accuracy, 4))
    col2.metric("AUC", round(auc, 4))
    col3.metric("Precision", round(precision, 4))

    col4, col5, col6 = st.columns(3)
    col4.metric("Recall", round(recall, 4))
    col5.metric("F1 Score", round(f1, 4))
    col6.metric("MCC", round(mcc, 4))

    st.subheader("Confusion Matrix")

    cm = confusion_matrix(y, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    st.pyplot(fig)

    st.subheader("Classification Report")
    st.text(classification_report(y, y_pred))