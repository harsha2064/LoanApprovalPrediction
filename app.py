import streamlit as st
import pandas as pd
import pickle

# ==============================
# Load the trained model & scaler
# ==============================
with open("loan_model.pkl", "rb") as file:
    model = pickle.load(file)

with open("scaler.pkl", "rb") as file:
    scaler = pickle.load(file)

# ==============================
# Streamlit UI
# ==============================
st.title("🏦 Loan Approval Prediction App")
st.write("Fill in the details below to check if your loan will be approved.")

# --- Input fields ---
dependents = st.selectbox("Dependents", [0, 1, 2, 3])
applicant_income = st.number_input("Applicant Income", min_value=0, value=5000)
coapplicant_income = st.number_input("Coapplicant Income", min_value=0, value=0)
loan_amount = st.number_input("Loan Amount", min_value=0, value=100)
loan_amount_term = st.selectbox("Loan Amount Term (in days)", [360, 180, 120, 240])
credit_history = st.selectbox("Credit History", [0.0, 1.0])
gender = st.selectbox("Gender", ["Male", "Female"])
married = st.selectbox("Married", ["Yes", "No"])
education = st.selectbox("Education", ["Graduate", "Not Graduate"])
self_employed = st.selectbox("Self Employed", ["Yes", "No"])
property_area = st.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])

# ==============================
# Convert inputs to feature order
# ==============================
features = pd.DataFrame([[
    dependents,
    applicant_income,
    coapplicant_income,
    loan_amount,
    loan_amount_term,
    credit_history,
    1 if gender == "Male" else 0,
    1 if married == "Yes" else 0,
    1 if education == "Not Graduate" else 0,
    1 if self_employed == "Yes" else 0,
    1 if property_area == "Semiurban" else 0,
    1 if property_area == "Urban" else 0
]], columns=[
    "Dependents", "ApplicantIncome", "CoapplicantIncome", "LoanAmount",
    "Loan_Amount_Term", "Credit_History",
    "Gender_Male", "Married_Yes", "Education_Not Graduate",
    "Self_Employed_Yes", "Property_Area_Semiurban", "Property_Area_Urban"
])

# ==============================
# Prediction
# ==============================
if st.button("Predict Loan Status"):
    # scale numeric values
    X_scaled = scaler.transform(features)

    prediction = model.predict(X_scaled)[0]
    proba = model.predict_proba(X_scaled)[0]

    if prediction == 1:
        st.success(f"✅ Loan Approved! (Confidence: {round(proba[1]*100, 2)}%)")
    else:
        st.error(f"❌ Loan Rejected! (Confidence: {round(proba[0]*100, 2)}%)")
