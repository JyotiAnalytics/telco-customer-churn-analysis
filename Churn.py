'''Telco Customer Churn(Data Analysis)'''

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df=pd.read_csv(r"C:\Users\USER\Downloads\data.csv")

print(df)
print(df.head(10))
print(df.tail(10))

print(df.describe())
print(df.info())


#Gender Wise Churn Analysis
print(df.groupby("gender")["Churn"].value_counts())


#Contract Type Wise Churn

print(df.groupby("Contract")["Churn"].value_counts())

#Internet Service Wise Average Monthly Charge
print(df.groupby("InternetService")["MonthlyCharges"].value_counts())

#Senior Citizen Wise Churn

print(df.groupby("SeniorCitizen")["Churn"].value_counts())

#Pivot Tabel(Contract vs Churn)
print(df.dtypes)
print(df["TotalCharges"].dtype)
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
print(pd.pivot_table(
    df,
    values="TotalCharges",
    index="Contract",
    columns="Churn",
    aggfunc="mean"
))

#Pivot Tabel(Gender vs Contract)
print(pd.pivot_table(
    df,
    values="MonthlyCharges",
    index="gender",
    columns="Contract",
    aggfunc="mean"
))

#Pivot Tabel(Payment Method Analysis)
print(pd.pivot_table(
    df,
    values="TotalCharges",
    index="PaymentMethod",
    aggfunc=["mean","max","min"]
))


#Missing Value Handelling

print(df.isnull().sum())
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

#Categorical Encoding
print(df.select_dtypes(include="object").columns)

#Label Encoding
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

df["gender"] = le.fit_transform(df["gender"])
df["Partner"] = le.fit_transform(df["Partner"])
df["Dependents"] = le.fit_transform(df["Dependents"])
df["PhoneService"] = le.fit_transform(df["PhoneService"])
df["Churn"] = le.fit_transform(df["Churn"])

#One Hot Coding

df = pd.get_dummies(
    df,
    columns=["InternetService","Contract","PaymentMethod"],
    drop_first=True
)
print(df)

#Feature Engineering

df["TenureGroup"] = pd.cut(
    df["tenure"],
    bins=[0,12,24,48,72],
    labels=["0-1 Year","1-2 Year","2-4 Year","4-6 Year"]
)
print(df)

#Average Charge Per Month

df["AvgChargePerMonth"] = (
    df["TotalCharges"] / (df["tenure"] + 1)
)
print(df)

#Senior Citizen Category

df["SeniorCitizen"] = df["SeniorCitizen"].replace(
    {0:"No",1:"Yes"}
)
print(df)


#Exploratory Data Analysis (EDA)
#A) Data Pattern Analysis
print(df.shape)
print(df.info())
print(df.describe())

'''Churn Distribution'''
print(df["Churn"].value_counts())

sns.countplot(x="Churn", data=df)
plt.show()

'''Tenure Distribution'''
sns.histplot(df["tenure"], kde=True)
plt.show()

'''Monthly Charges Distribution'''
sns.histplot(df["MonthlyCharges"], kde=True)
plt.show()



#B)Relationship Analysis
'''Monthly Charges vs Churn'''
sns.boxplot(x="Churn", y="MonthlyCharges", data=df)
plt.show()

'''Tenure vs Churn'''
sns.boxplot(x="Churn", y="tenure", data=df)
plt.show()


'''Correlation Analysis '''

num_df = df.select_dtypes(include="number")

sns.heatmap(num_df.corr(), annot=True)
plt.show()

#Outlier Check
'''Box Plot'''
import seaborn as sns
import matplotlib.pyplot as plt

sns.boxplot(x=df["MonthlyCharges"])
plt.show()

#IQR Method
q1 = df["MonthlyCharges"].quantile(0.25)
q3 = df["MonthlyCharges"].quantile(0.75)

iqr = q3 - q1

lower = q1 - 1.5 * iqr
upper = q3 + 1.5 * iqr

outliers = df[
    (df["MonthlyCharges"] < lower) |
    (df["MonthlyCharges"] > upper)
]

print("Outliers:", len(outliers))

#Machine Learning Working

df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"],
    errors="coerce"
)

df["TotalCharges"] = df["TotalCharges"].fillna(
    df["TotalCharges"].median()
)


X = df.drop(["customerID", "Churn"], axis=1)
X = X.fillna(X.median(numeric_only=True))

X = pd.get_dummies(X, drop_first=True)

y = df["Churn"]

print(X)
print(y)

#Train-Test Split
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)
print(X_train)
print(X_test)

print(y_train)
print(y_test)
print(X_train.select_dtypes(include=["object", "string"]).columns)
print(X_train.dtypes)
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
df["TenureGroup"] = le.fit_transform(df["TenureGroup"])
#Logistic Regression

from sklearn.linear_model import LogisticRegression

lr = LogisticRegression(max_iter=1000)

lr.fit(X_train, y_train)

y_pred = lr.predict(X_test)

#Evaluation
from sklearn.metrics import accuracy_score

print("Accuracy:", accuracy_score(y_test, y_pred))

#Random Forest
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(random_state=42)

rf.fit(X_train, y_train)

y_pred_rf = rf.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred_rf))
#Decision Tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn import tree
import matplotlib.pyplot as plt

# Model
dt = DecisionTreeClassifier(
    max_depth=4,
    random_state=42
)

# Train
dt.fit(X_train, y_train)

# Prediction
y_pred = dt.predict(X_test)

# Accuracy
print("Accuracy:", accuracy_score(y_test, y_pred))

# Classification Report
print(classification_report(y_test, y_pred))

#XGBoost
from xgboost import XGBClassifier

xgb = XGBClassifier()

xgb.fit(X_train, y_train)

y_pred_xgb = xgb.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred_xgb))

#Confusion Matrix
from sklearn.metrics import confusion_matrix

print(confusion_matrix(y_test, y_pred_rf))

#Classification Report
from sklearn.metrics import classification_report

print(classification_report(y_test, y_pred_rf))

#Feature Importance
import pandas as pd

importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": rf.feature_importances_
})

print(
    importance.sort_values(
        by="Importance",
        ascending=False
    ).head(10)
)

#Lightgbm
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score

lgbm = LGBMClassifier(random_state=42)

lgbm.fit(X_train, y_train)

y_pred = lgbm.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))

# Business Insights:
# 1. Month-to-Month contract customers have the highest churn rate.
# 2. Customers with shorter tenure are more likely to churn.
# 3. Higher monthly charges are associated with increased churn.
# 4. Electronic check users show higher churn behavior.
# 5. Long-term customers are less likely to churn.


#GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

rf = RandomForestClassifier(random_state=42)

param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [5, 10, 15],
    'min_samples_split': [2, 5, 10]
}

grid_search = GridSearchCV(
    estimator=rf,
    param_grid=param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1
)

grid_search.fit(X_train, y_train)

print("Best Parameters:", grid_search.best_params_)
print("Best Score:", grid_search.best_score_)


#RandomizedSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(random_state=42)

param_dist = {
    'n_estimators': [50,100,200,300],
    'max_depth': [5,10,15,20,None],
    'min_samples_split': [2,5,10],
    'min_samples_leaf': [1,2,4]
}

random_search = RandomizedSearchCV(
    estimator=rf,
    param_distributions=param_dist,
    n_iter=10,
    cv=5,
    scoring='accuracy',
    random_state=42,
    n_jobs=-1
)

random_search.fit(X_train, y_train)

print("Best Parameters:", random_search.best_params_)
print("Best Score:", random_search.best_score_)