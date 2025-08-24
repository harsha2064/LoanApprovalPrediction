import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.impute import SimpleImputer

# Title
st.title("🏦 Loan Approval Prediction")

# Load dataset
@st.cache_data
def load_data():
    df = pd.read_csv("LoanApprovalPrediction.csv")
    return df

df = load_data()
st.write("### Dataset Preview", df.head())

# Features and Target
X = df.drop(columns=['Loan_Status'])
y = df['Loan_Status']

# Handle missing values
imputer = SimpleImputer(strategy='most_frequent')
X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Models
models = {
    "Logistic Regression": LogisticRegression(max_iter=200),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier()
}

accuracies = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracies[name] = accuracy_score(y_test, y_pred)

# Show Accuracies
st.write("### Model Accuracies")
for name, acc in accuracies.items():
    st.write(f"✅ {name}: **{acc:.2f}**")

# Sidebar for user input
st.sidebar.header("Enter Applicant Details")

user_data = {}
for col in X.columns:
    user_data[col] = st.sidebar.text_input(f"{col}")

# Prediction Button
if st.sidebar.button("Predict Loan Approval"):
    user_df = pd.DataFrame([user_data])
    user_df = user_df.replace("", pd.NA)
    user_df = pd.DataFrame(imputer.transform(user_df), columns=X.columns)

    model = LogisticRegression(max_iter=200)
    model.fit(X_train, y_train)
    prediction = model.predict(user_df)[0]

    st.sidebar.success(f"🎯 Loan Status Prediction: **{prediction}**")
