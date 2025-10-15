import streamlit as st
import pandas as pd
import numpy as np
import pickle
import shap
import matplotlib.pyplot as plt

# Load trained model, scaler, and feature columns
model = pickle.load(open('logistic_model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))
feature_columns = pickle.load(open('feature_columns.pkl', 'rb'))
background = pd.read_csv("background_sample.csv")

# Page Configuration
st.set_page_config(page_title="Customer Churn Prediction", layout="centered")
st.title("Customer Churn Prediction Dashboard")
st.markdown("""
This interactive dashboard predicts the likelihood of a customer churning (leaving your service) based on their profile and service details. 
Provide customer data below to receive a prediction and interpretation of the model's reasoning.
""")

# ---------------------- INPUT SECTION ----------------------
st.markdown("---")
st.subheader("1. Customer Details")

col1, col2, col3 = st.columns(3)

with col1:
    gender = st.selectbox("Gender", ['Female', 'Male'])
    Partner = st.selectbox("Partner", ['No', 'Yes'])
    PhoneService = st.selectbox("Phone Service", ['No', 'Yes'])
    InternetService = st.selectbox("Internet Service", ['DSL', 'Fiber optic', 'No'])
    OnlineSecurity = st.selectbox("Online Security", ['No internet service', 'No', 'Yes'])
    DeviceProtection = st.selectbox("Device Protection", ['No internet service', 'No', 'Yes'])

with col2:
    SeniorCitizen = st.selectbox("Senior Citizen", ['No', 'Yes'])
    Dependents = st.selectbox("Dependents", ['No', 'Yes'])
    MultipleLines = st.selectbox("Multiple Lines", ['No phone service', 'No', 'Yes'])
    OnlineBackup = st.selectbox("Online Backup", ['No internet service', 'No', 'Yes'])
    TechSupport = st.selectbox("Tech Support", ['No internet service', 'No', 'Yes'])

with col3:
    StreamingTV = st.selectbox("Streaming TV", ['No internet service', 'No', 'Yes'])
    StreamingMovies = st.selectbox("Streaming Movies", ['No internet service', 'No', 'Yes'])
    Contract = st.selectbox(
        "Contract Type",
        ['Month-to-month', 'One year', 'Two year'],
        help="Longer contracts usually indicate lower churn likelihood"
    )
    PaperlessBilling = st.selectbox("Paperless Billing", ['No', 'Yes'])
    PaymentMethod = st.selectbox("Payment Method", [
        'Electronic check', 'Mailed check',
        'Bank transfer (automatic)', 'Credit card (automatic)'
    ])

st.markdown("**Billing Details**")
col4, col5, col6 = st.columns(3)
with col4:
    tenure = st.slider("Tenure (months)", 0, 72, 12)
with col5:
    MonthlyCharges = st.slider("Monthly Charges", 0.0, 150.0, 70.0)
with col6:
    TotalCharges = st.slider("Total Charges", 0.0, 10000.0, 1000.0)

# ---------------------- PREDICTION ----------------------
st.markdown("---")
st.subheader("2. Churn Prediction")

if st.button("🔍 Predict Churn"):
    with st.spinner("Analyzing customer profile..."):
        import time
        time.sleep(1.5)  # Simulated processing time

        input_dict = {
            'gender': gender,
            'SeniorCitizen': SeniorCitizen,
            'Partner': Partner,
            'Dependents': Dependents,
            'PhoneService': PhoneService,
            'MultipleLines': MultipleLines,
            'InternetService': InternetService,
            'OnlineSecurity': OnlineSecurity,
            'OnlineBackup': OnlineBackup,
            'DeviceProtection': DeviceProtection,
            'TechSupport': TechSupport,
            'StreamingTV': StreamingTV,
            'StreamingMovies': StreamingMovies,
            'Contract': Contract,
            'PaperlessBilling': PaperlessBilling,
            'PaymentMethod': PaymentMethod,
            'tenure': tenure,
            'MonthlyCharges': MonthlyCharges,
            'TotalCharges': TotalCharges
        }

        input_df = pd.DataFrame([input_dict])
        input_encoded = pd.get_dummies(input_df)

        for col in feature_columns:
            if col not in input_encoded.columns:
                input_encoded[col] = 0
        input_encoded = input_encoded[feature_columns]

        input_scaled = scaler.transform(input_encoded)

        churn_prob = model.predict_proba(input_scaled)[0][1]
        churn_label = "**High Risk of Churn**" if churn_prob > 0.5 else "**Low Risk of Churn**"
        result_color = "🔴" if churn_prob > 0.5 else "🟢"
        st.success("Prediction complete!")

    st.markdown(f"### {result_color} {churn_label}")
    st.markdown(f"**Predicted Probability:** `{churn_prob:.2%}`")

    # ---------------------- SHAP LOCAL ----------------------
    st.markdown("---")
    st.subheader("3. Why Did the Model Make This Prediction?")
    st.markdown("Below is a chart showing how individual features contributed to this specific prediction.")

    explainer = shap.Explainer(model, background)
    shap_values = explainer(input_encoded)

    fig_local, ax = plt.subplots()
    shap.plots.bar(shap_values[0], show=False)
    st.pyplot(fig_local)

    # ---------------------- SHAP GLOBAL ----------------------
    st.markdown("---")
    st.subheader("4. What Generally Drives Churn?")
    st.markdown("This chart highlights the most influential features for churn across all customers in the dataset.")

    global_explainer = shap.Explainer(model, background)
    global_shap_values = global_explainer(background)

    fig_global, ax = plt.subplots()
    shap.plots.bar(global_shap_values, show=False)
    st.pyplot(fig_global)

st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: gray; font-size: 13px;'>"
    "Created as part of a data analyst portfolio project by Yash Agarwal"
    "</div>",
    unsafe_allow_html=True
)
