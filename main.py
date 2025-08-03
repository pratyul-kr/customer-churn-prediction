import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt

st.set_page_config(page_title="Churn Predictor", layout="centered")
st.title("Customer Churn Prediction App")
st.markdown("Predict whether a customer is likely to churn based on their contract details.")

# Load dataset
@st.cache_data
def load_data():
    return pd.read_csv("customer-churn.csv")

try:
    df = load_data()
    st.subheader("Preview of Dataset")
    st.dataframe(df.head())
except FileNotFoundError:
    st.error("File 'customer-churn.csv' not found. Please ensure it's in the app directory.")

# Load models and encoders
try:
    rf_model = joblib.load("random_forest_model.pkl")
    log_model = joblib.load("logistic_model.pkl")
    contract_encoder = joblib.load("contract_encoder.pkl")
    internet_encoder = joblib.load("internet_encoder.pkl")
    churn_encoder = joblib.load("churn_encoder.pkl")
except FileNotFoundError:
    st.error("Trained models or encoders not found. Please run the training script first.")
    st.stop()

# Load test data
@st.cache_data
def load_test_data():
    try:
        X_test = pd.read_csv("X_test.csv")
        y_test = pd.read_csv("y_test.csv").squeeze()
        return X_test, y_test
    except:
        return None, None

X_test, y_test = load_test_data()

# Sidebar for model and threshold selection
st.sidebar.header("Model Configuration")
model_choice = st.sidebar.selectbox("Choose a model", ["Random Forest", "Logistic Regression"])
threshold = st.sidebar.slider("Classification Threshold", min_value=0.1, max_value=0.9, value=0.5, step=0.05)

# Select model
model = rf_model if model_choice == "Random Forest" else log_model

# Prediction UI
st.subheader("Make a Prediction")

contract = st.selectbox("Contract Type", contract_encoder.classes_)
internet = st.selectbox("Internet Service", internet_encoder.classes_)
tenure = st.number_input("Tenure (months)", min_value=0, max_value=100, value=12)
monthly_charges = st.number_input("Monthly Charges", min_value=0.0, max_value=200.0, value=70.0)

if st.button("Predict Churn"):
    contract_encoded = contract_encoder.transform([contract])[0]
    internet_encoded = internet_encoder.transform([internet])[0]

    input_data = np.array([[contract_encoded, tenure, monthly_charges, internet_encoded]])
    proba = model.predict_proba(input_data)[0][1]
    prediction = int(proba >= threshold)
    result = churn_encoder.inverse_transform([prediction])[0]

    st.success(f"Prediction using **{model_choice}** at threshold **{threshold:.2f}**: **{result}**")

    # Evaluate model
    if X_test is not None and y_test is not None:
        y_proba_test = model.predict_proba(X_test)[:, 1]
        y_pred_test = (y_proba_test >= threshold).astype(int)

        acc = accuracy_score(y_test, y_pred_test)
        prec = precision_score(y_test, y_pred_test)
        rec = recall_score(y_test, y_pred_test)
        f1 = f1_score(y_test, y_pred_test)

        st.subheader("Model Performance on Test Data")
        st.markdown(f"- **Accuracy:** {acc:.2f}")
        st.markdown(f"- **Precision:** {prec:.2f}")
        st.markdown(f"- **Recall:** {rec:.2f}")
        st.markdown(f"- **F1 Score:** {f1:.2f}")

        with st.expander("Full Classification Report"):
            report = classification_report(y_test, y_pred_test, target_names=churn_encoder.classes_)
            st.text(report)

        st.subheader("Performance Metrics Chart")
        metrics_df = pd.DataFrame({
            'Metric': ['Accuracy', 'Precision', 'Recall', 'F1 Score'],
            'Score': [acc, prec, rec, f1]
        })

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.barh(metrics_df['Metric'], metrics_df['Score'], color='skyblue')
        ax.set_xlim(0, 1)
        ax.set_title(f'{model_choice} Evaluation Metrics')
        ax.set_xlabel('Score')
        for i, v in enumerate(metrics_df['Score']):
            ax.text(v + 0.01, i, f"{v:.2f}", color='black', va='center')
        st.pyplot(fig)
    else:
        st.warning("Test data not available. To display metrics, please save `X_test.csv` and `y_test.csv` during training.")