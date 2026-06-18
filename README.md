# 📊 Telco Customer Churn Analysis & Prediction

## 📌 Project Overview

This project focuses on analyzing customer churn behavior in a telecom company and building machine learning models to predict whether a customer is likely to leave the service.

The project covers the complete data analysis workflow including:

- Data Cleaning
- Data Preprocessing
- Feature Engineering
- Exploratory Data Analysis (EDA)
- Outlier Detection
- Machine Learning Modeling
- Hyperparameter Tuning
- Business Insights

---

## 🎯 Business Problem

Customer churn is one of the biggest challenges for telecom companies.

The goal of this project is to:

- Identify factors affecting customer churn.
- Analyze customer behavior patterns.
- Predict customers who are likely to churn.
- Provide business recommendations to improve customer retention.

---

## 📂 Dataset Information

Dataset: Telco Customer Churn Dataset

Total Records: **7043**

Total Features: **21**

### Features

- customerID
- gender
- SeniorCitizen
- Partner
- Dependents
- tenure
- PhoneService
- MultipleLines
- InternetService
- OnlineSecurity
- OnlineBackup
- DeviceProtection
- TechSupport
- StreamingTV
- StreamingMovies
- Contract
- PaperlessBilling
- PaymentMethod
- MonthlyCharges
- TotalCharges
- Churn (Target Variable)

---

## 🛠 Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-Learn
- XGBoost
- LightGBM

---

## 🔧 Data Preprocessing

Performed:

- Data Type Conversion
- Missing Value Handling
- Label Encoding
- One-Hot Encoding
- Feature Engineering
- Outlier Detection using IQR Method

---

## 📈 Exploratory Data Analysis (EDA)

Analysis Performed:

### Customer Churn Analysis
- Gender-wise Churn Analysis
- Senior Citizen-wise Churn Analysis
- Contract Type-wise Churn Analysis

### Distribution Analysis
- Churn Distribution
- Tenure Distribution
- Monthly Charges Distribution

### Relationship Analysis
- Monthly Charges vs Churn
- Tenure vs Churn

### Correlation Analysis
- Correlation Heatmap

### Outlier Detection
- Box Plot Analysis
- IQR Method

---

## ⚙️ Feature Engineering

### Tenure Group

Created customer tenure categories:

- 0–1 Year
- 1–2 Years
- 2–4 Years
- 4–6 Years

### Average Monthly Charge

Created new feature:

```python
AvgChargePerMonth = TotalCharges / (tenure + 1)
```

---

## 🤖 Machine Learning Models

The following models were trained and evaluated:

### 1. Logistic Regression

Used for baseline classification.

### 2. Decision Tree Classifier

Used for interpretable prediction.

### 3. Random Forest Classifier

Used for improved predictive performance.

### 4. XGBoost Classifier

Used for boosting-based prediction.

### 5. LightGBM Classifier

Used for efficient gradient boosting.

---

## 🔍 Hyperparameter Tuning

Applied:

### GridSearchCV

Optimized:

- n_estimators
- max_depth
- min_samples_split

### RandomizedSearchCV

Optimized:

- n_estimators
- max_depth
- min_samples_split
- min_samples_leaf

---

## 📊 Model Evaluation

Evaluation Techniques:

- Accuracy Score
- Confusion Matrix
- Classification Report
- Feature Importance Analysis

---

## 💡 Key Business Insights

### Insight 1
Customers with Month-to-Month contracts have the highest churn rate.

### Insight 2
Customers with shorter tenure are more likely to leave the service.

### Insight 3
Higher monthly charges are associated with increased churn.

### Insight 4
Customers using Electronic Check payment methods show higher churn behavior.

### Insight 5
Long-term customers are less likely to churn.

---

## 📁 Project Structure

```text
Telco-Customer-Churn/
│
├── data.csv
├── telco_churn.py
├── README.md
└── images/
```

---

## 👨‍💻 Author

**Jyoti.IO**

Aspiring Data Analyst | Machine Learning Enthusiast

---

⭐ If you found this project useful, please consider giving it a star.
