import streamlit as st
import pickle
import numpy as np

# Load trained model
with open("loan_model.pkl", "rb") as f:
    model = pickle.load(f)

st.title("🏦 Loan Approval Prediction App")
st.write("Fill in the details below to check if your loan will be approved.")

# Input fields
gender = st.radio("Gender", ["Male", "Female"])
married = st.radio("Married", ["Yes", "No"])
dependents = st.selectbox("Dependents", ["0", "1", "2", "3+"])
education = st.radio("Education", ["Graduate", "Not Graduate"])
self_employed = st.radio("Self Employed", ["Yes", "No"])
applicant_income = st.number_input("Applicant Income", min_value=0)
coapplicant_income = st.number_input("Coapplicant Income", min_value=0)
loan_amount = st.number_input("Loan Amount", min_value=0)
loan_term = st.number_input("Loan Amount Term (in days)", min_value=0)
credit_history = st.selectbox("Credit History", [1.0, 0.0])
property_area = st.radio("Property Area", ["Rural", "Semiurban", "Urban"])

# Prepare features in the same order as training
features = [
    int(dependents.replace("3+", "3")),   # Dependents
    applicant_income,
    coapplicant_income,
    loan_amount,
    loan_term,
    credit_history,
    1 if gender == "Male" else 0,
    1 if married == "Yes" else 0,
    1 if education == "Not Graduate" else 0,
    1 if self_employed == "Yes" else 0,
    1 if property_area == "Semiurban" else 0,
    1 if property_area == "Urban" else 0
]

# Predict button
if st.button("Predict"):
    prediction = model.predict([features])[0]
    if prediction == 1:
        st.success("✅ Loan will be Approved!")
    else:
        st.error("❌ Loan will be Rejected.")
