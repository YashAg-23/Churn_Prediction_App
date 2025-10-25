# 🚀 Customer Churn Prediction App

### 📊 Subscription-Based Service | End-to-End Data Science Project

This project focuses on predicting **customer churn** for a subscription-based service using advanced **Machine Learning** techniques. It helps identify customers likely to cancel their subscription, enabling data-driven retention strategies and improved business outcomes.

---

## 📁 Project Overview

Customer churn directly impacts business revenue and growth.  
In this project, we built an **end-to-end churn prediction system** — from **data preprocessing and feature engineering** to **model training, SHAP-based interpretability**, and a **Streamlit web app** for real-time predictions.

The final solution allows business users to:
- Predict whether a customer is likely to churn.
- Understand **key factors influencing churn** through **SHAP visualizations** (both global and local explanations).
- Gain actionable insights for improving **customer retention strategies**.

---

## 🧠 Key Features

✅ **End-to-End ML Pipeline** — from data cleaning to deployment  
✅ **SHAP Explainability** — interactive feature importance insights  
✅ **Streamlit Web App** — clean, professional UI for business usability  
✅ **Comprehensive Model Evaluation** — precision, recall, F1, ROC-AUC, and confusion matrix  
✅ **Actionable Business Insights** — strategic recommendations based on data findings  

---

## 🧩 Tech Stack

| Layer | Tools / Libraries |
|-------|--------------------|
| **Programming Language** | Python |
| **Data Processing** | pandas, numpy |
| **Data Visualization** | matplotlib, seaborn |
| **Modeling** | scikit-learn |
| **Model Explainability** | SHAP |
| **App Development** | Streamlit |
| **Version Control** | Git & GitHub |

---

## ⚙️ Project Workflow

### 1️⃣ Data Understanding & Cleaning
- Imported and explored the **Telco Customer Churn dataset**.
- Handled **missing values**, **encoded categorical features**, and normalized numerical features.
- Performed **exploratory data analysis (EDA)** to uncover churn trends.

### 2️⃣ Feature Engineering
- Created dummy variables for categorical columns like:
  `InternetService`, `Contract`, `PaymentMethod`, `OnlineSecurity`, etc.
- Selected 30+ meaningful features contributing to churn.

### 3️⃣ Model Building
- Trained multiple models — Logistic Regression, Random Forest, XGBoost.
- Chose the **best-performing model** based on accuracy, recall, and AUC score.

### 4️⃣ Model Evaluation
- Generated **classification reports**, **ROC curves**, and **confusion matrix**.
- Evaluated business implications of precision vs recall trade-offs.

### 5️⃣ Model Explainability (SHAP)
- **Global Interpretability** — identified top drivers of churn across all customers.
- **Local Interpretability** — explained why individual customers churned.

### 6️⃣ Streamlit App Development
- Built a user-friendly interface for predictions.
- Integrated SHAP visualizations for transparency.
- Added clean UI polish and structured layout for professional appeal.

---

## 📈 Results & Key Insights

- High **recall rate** ensures minimal missed churn cases.  
- **Contract Type**, **Tenure**, and **Online Security** emerged as top predictors.  
- Identified customer segments most at risk of churn.  
- Provided actionable **retention recommendations** to reduce churn by ~20–25%.

---

## 💡 Business Impact & Recommendations

- **Offer discounts or loyalty programs** for month-to-month contract users.  
- **Encourage long-term plans** to stabilize customer retention.  
- **Enhance customer support and online security features** — critical churn factors.  
- Use **personalized outreach** for at-risk customers identified by the model.

---

## 🧾 Dataset

- **Source:** [Telco Customer Churn Dataset (Kaggle)](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)  
- **Rows:** 7,043  
- **Columns:** 21  
- **Target Variable:** `Churn` (Yes/No)

---

## 🖥️ App Demo

🎯 **Streamlit App Preview:**  
The app provides a smooth workflow for business users to:
- Input customer details
- Predict churn probability
- Explore SHAP-based interpretability

📌 **Local Execution:**
```bash
streamlit run app.py
