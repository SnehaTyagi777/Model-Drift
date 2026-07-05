# Import required libraries
import os
import pickle
import pandas as pd

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Create required folders
os.makedirs("data", exist_ok=True)
os.makedirs("models", exist_ok=True)

# Load Breast Cancer Dataset
dataset = load_breast_cancer()

# Features
X = pd.DataFrame(dataset.data, columns=dataset.feature_names)

# Target
y = pd.Series(dataset.target, name="target")

print("Dataset Loaded Successfully!")
print("Number of Records:", X.shape[0])
print("Number of Features:", X.shape[1])

#Save Reference Dataset
reference_df = X.copy()
reference_df["target"] = y

reference_df.to_csv("data/reference.csv", index=False)

print("\nreference.csv created successfully!")

#Split Dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("\nTraining Data :", X_train.shape)
print("Testing Data  :", X_test.shape)

#Train Logistic Regression
model = LogisticRegression(max_iter=10000)

model.fit(X_train, y_train)

print("\nModel Training Completed!")

#Save Model
with open("models/model.pkl", "wb") as file:
    pickle.dump(model, file)

print("model.pkl saved successfully!")

#Predictions
predictions = model.predict(X_test)

#Evaluation Metrics
accuracy = accuracy_score(y_test, predictions)
precision = precision_score(y_test, predictions)
recall = recall_score(y_test, predictions)
f1 = f1_score(y_test, predictions)

print("\nModel Evaluation")
print("-" * 40)
print(f"Accuracy  : {accuracy:.4f}")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1 Score  : {f1:.4f}")

