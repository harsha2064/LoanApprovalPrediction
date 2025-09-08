

# 🏦 Loan Approval Prediction

This project predicts whether a loan application will be **Approved** or **Rejected** using Machine Learning models.


## ✨ Features

* Predict loan approval based on applicant details
* Machine Learning models: Logistic Regression, Decision Tree, Random Forest
* Data preprocessing: Missing values handling, categorical encoding, feature       scaling
* Interactive **Streamlit web app** for real-time prediction
* Deployed on **Streamlit Cloud** with sharable link

---

## 📊 Dataset

The dataset contains applicant details like income, loan amount, dependents, property area, etc.

* **Target variable:** Loan\_Status (Approved / Rejected)
* **Input features:**

  * Dependents
  * ApplicantIncome
  * CoapplicantIncome
  * LoanAmount
  * Loan_Amount_Term
  * Credit_History
  * Gender
  * Married
  * Education
  * Self_Employed
  * Property_Area

⚠️ **Note**: The model sometimes struggles with extremely large/unrealistic values (e.g., very high incomes or loan amounts). This is due to dataset limitations, not the ML pipeline.

---

## 🧠 Model Training

Steps followed in `LoanApprovalPrediction.ipynb`:

1. Data Cleaning – handled missing values
2. Encoding – converted categorical features using one-hot encoding
3. Scaling – standardized numerical features using `StandardScaler`
4. Model Training – trained Logistic Regression, Decision Tree, Random Forest
5. Evaluation – Logistic Regression performed best with this dataset

---

## 📂 Project Structure

* **Loan Approval Prediction.ipynb** → Jupyter Notebook with full data preprocessing, model training, and evaluation.
* **app.py** → Streamlit app code for deployment.
* **loan_model.pkl** → Trained machine learning model saved for predictions.
* **requirements.txt** → Python dependencies required to run the project.
* **scaler.pkl** → Scaler file used to standardize numerical features.


## ⚙️ Installation & Setup

1. Clone the repository

   ```bash
   git clone https://github.com/your-username/LoanApprovalPrediction.git
   cd LoanApprovalPrediction
   ```

2. Install dependencies

   ```bash
   pip install -r requirements.txt
   ```

3. Run Streamlit app

   ```bash
   streamlit run app.py
   ```

4. Ensure both `loan_model.pkl` and `scaler.pkl` are present in the folder:

   * `loan_model.pkl` → trained ML model
   * `scaler.pkl` → StandardScaler used during training (for consistent preprocessing)

---

## 🧪 Sample Test Cases

### ✅ Case 1: Loan Approved

* Dependents: `0`
* ApplicantIncome: `5000`
* CoapplicantIncome: `2000`
* LoanAmount: `150`
* Loan\_Amount\_Term: `360`
* Credit\_History: `1`
* Gender: `Male`
* Married: `Yes`
* Education: `Graduate`
* Self\_Employed: `No`
* Property\_Area: `Urban`

**Prediction → Approved**

---

### ❌ Case 2: Loan Rejected

* Dependents: `3+`
* ApplicantIncome: `1500`
* CoapplicantIncome: `0`
* LoanAmount: `250`
* Loan\_Amount\_Term: `360`
* Credit\_History: `0`
* Gender: `Male`
* Married: `No`
* Education: `Not Graduate`
* Self\_Employed: `Yes`
* Property\_Area: `Rural`

**Prediction → Rejected**

---

## 🚀 Deployment

The app is deployed on **Streamlit Cloud**
🔗 Live Demo: https://loanapprovalprediction-lrcmwobigfqm9ebtc5xwky.streamlit.app/

---

## 📌 Conclusion

* Logistic Regression gave the **best accuracy** with this dataset
* Demonstrates a complete **end-to-end ML workflow** (EDA → Preprocessing → Model Training → Deployment)

---


