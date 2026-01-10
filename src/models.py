from sklearn.model_selection import train_test_split
from src.data_processing.py import clean_process_data  
import pandas as pd
import sys
import os
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, ConfusionMatrixDisplay

sys.path.append(os.path.abspath('..'))

df = pd.read_csv('../data/raw/heart_failure_clinical_records_dataset.csv')
df_processed = clean_process_data(df)

# I will exclude death_event to prevent data leakage

X = df_processed.drop('DEATH_EVENT', axis=1)
y = df_processed['DEATH_EVENT']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


# Model 1: Logistic regression. -> this is famous one for small medical data, I chose this linear classifier
#  model because it calculates death_event as a linear combination of its features  

model1 = LogisticRegression()
model1.fit(X_train, y_train)
y_pred_model1 = model1.predict(X_test)

print(classification_report(y_test, y_pred_log))
