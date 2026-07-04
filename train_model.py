import pandas as pd
import numpy as np
import joblib
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)

# ===============================
# Load Dataset
# ===============================

data = pd.read_csv("StudentsPerformance.csv")

print("\nMissing Values\n")
print(data.isnull().sum())

# ===============================
# Encode Categorical Columns
# ===============================

encoders = {}

categorical_columns = [
    "gender",
    "race/ethnicity",
    "parental level of education",
    "lunch",
    "test preparation course"
]

for col in categorical_columns:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])
    encoders[col] = le

# ===============================
# Create Target Column
# ===============================

average = (
    data["math score"] +
    data["reading score"] +
    data["writing score"]
) / 3

data["Pass"] = average.apply(lambda x: 1 if x >= 40 else 0)

# ===============================
# Features & Target
# ===============================

X = data.drop(
    ["Pass", "math score", "reading score", "writing score"],
    axis=1
)

y = data["Pass"]

# ===============================
# Train Test Split
# ===============================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

# ===============================
# Train Model
# ===============================

model = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)

model.fit(X_train, y_train)

# ===============================
# Prediction
# ===============================

y_pred = model.predict(X_test)

# ===============================
# Evaluation
# ===============================

print("\n==============================")
print("MODEL PERFORMANCE")
print("==============================")

print(f"\nAccuracy : {accuracy_score(y_test,y_pred):.2f}")

print("\nConfusion Matrix\n")
print(confusion_matrix(y_test,y_pred))

print("\nClassification Report\n")
print(classification_report(y_test,y_pred))

# ===============================
# Sample Prediction
# ===============================

sample = pd.DataFrame([{
    "gender":0,
    "race/ethnicity":2,
    "parental level of education":3,
    "lunch":1,
    "test preparation course":0
}])

prediction = model.predict(sample)

print("\n==============================")

if prediction[0] == 1:
    print("Prediction : PASS")
else:
    print("Prediction : FAIL")

print("==============================")

# ===============================
# Save Model
# ===============================

joblib.dump(model, "student_model.pkl")
joblib.dump(encoders, "label_encoders.pkl")

print("\nModel Saved : student_model.pkl")
print("Encoders Saved : label_encoders.pkl")