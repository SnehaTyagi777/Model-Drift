import os
import pandas as pd

from evidently import Report
from evidently.presets import DataDriftPreset

# Create reports folder if it doesn't exist
os.makedirs("reports", exist_ok=True)

# Load datasets
reference_data = pd.read_csv("data/reference.csv")
current_data = pd.read_csv("data/current.csv")

print("Reference Dataset Loaded :", reference_data.shape)
print("Current Dataset Loaded   :", current_data.shape)

# Create Data Drift Report
report = Report(
    metrics=[
        DataDriftPreset(),
    ]
)

# Run Report
my_eval = report.run(
    reference_data=reference_data,
    current_data=current_data,
)

# Save HTML Report
my_eval.save_html("reports/model_drift_report.html")

print("\nModel Drift Report Generated Successfully!")
print("Location: reports/model_drift_report.html")