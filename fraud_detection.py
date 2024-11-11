import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
import joblib
import matplotlib.pyplot as plt
import seaborn as sns


data = pd.read_csv('data/creditcard.csv')
print("Data loaded successfully.")
print(data.head())


# Standardize 'Amount' and 'Time' features
scaler = StandardScaler()
data['scaled_amount'] = scaler.fit_transform(data[['Amount']])
data['scaled_time'] = scaler.fit_transform(data[['Time']])


data.drop(['Amount', 'Time'], axis=1, inplace=True)

print("Missing values in target (y):", data['Class'].isnull().sum())
data_clean = data.dropna(subset=['Class'])  # Drop rows with NaN in the target 'Class'




data['Class'] = data['Class'].fillna(data['Class'].mode()[0])


X = data_clean.drop('Class', axis=1)
y = data_clean['Class']
X = X.fillna(X.median())  # Impute missing values in features with the median value


sm = SMOTE(random_state=42)
X_res, y_res = sm.fit_resample(X, y)

X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.2, random_state=42)
print("Data split into training and testing sets.")

model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)
print("Model training complete.")


y_pred = model.predict(X_test)
print("Classification Report:\n", classification_report(y_test, y_pred))
print("ROC-AUC Score:", roc_auc_score(y_test, y_pred))

joblib.dump(model, 'models/fraud_detection_model.pkl')
print("Model saved as 'models/fraud_detection_model.pkl'.")

