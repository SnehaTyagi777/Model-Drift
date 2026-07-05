# Simulate Model Drift

import pandas as pd
import numpy as np

# Make results reproducible
np.random.seed(42)

# Load Reference Dataset

df = pd.read_csv("data/reference.csv")

print("Reference Dataset Loaded Successfully!")
print(df.shape)

# Add Gaussian Noise

feature_columns = df.columns[:-1]   # all columns except target

noise = np.random.normal(
    loc=0,
    scale=0.5,
    size=df[feature_columns].shape
)

df[feature_columns] = df[feature_columns] + noise

print("Gaussian Noise Added")

# Shift Important Features

df["mean radius"] = df["mean radius"] + 2

df["mean texture"] = df["mean texture"] + 3

print("Feature Shift Applied")

# Step 4: Flip 10% Target Labels

num_flip = int(0.10 * len(df))

random_index = np.random.choice(
    df.index,
    num_flip,
    replace=False
)

df.loc[random_index, "target"] = 1 - df.loc[random_index, "target"]

print("10% Labels Flipped")

# Save Current Dataset

df.to_csv("data/current.csv", index=False)

print("\ncurrent.csv created successfully!")

