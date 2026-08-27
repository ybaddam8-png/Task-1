"""
Task 1: Data Cleaning & Preprocessing
Dataset : Titanic (loaded via seaborn, same data as the classic Kaggle Titanic set)
Steps   : explore -> handle missing values -> encode categoricals -> scale
          numeric features -> detect/visualize/remove outliers
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder

pd.set_option("display.width", 120)

# ---------------------------------------------------------------------
# 1. Load data and explore basic info
# ---------------------------------------------------------------------
df = sns.load_dataset("titanic")

print("=" * 60)
print("STEP 1: Basic info")
print("=" * 60)
print(f"Shape: {df.shape}")
print("\nDtypes:\n", df.dtypes)
print("\nMissing values per column:\n", df.isnull().sum())

# 'deck' is ~77% missing and 'embark_town'/'alive'/'class'/'who'/'adult_male'/
# 'alone' duplicate information already in other columns -> drop them.
df = df.drop(columns=["deck", "embark_town", "alive", "class", "who", "adult_male", "alone"])

# ---------------------------------------------------------------------
# 2. Handle missing values
# ---------------------------------------------------------------------
print("\n" + "=" * 60)
print("STEP 2: Handling missing values")
print("=" * 60)

# age: numeric, right-skewed -> fill with median (robust to outliers)
median_age = df["age"].median()
df["age"] = df["age"].fillna(median_age)
print(f"Filled 'age' NaNs with median = {median_age}")

# embarked: categorical, only 2 missing -> fill with mode
mode_embarked = df["embarked"].mode()[0]
df["embarked"] = df["embarked"].fillna(mode_embarked)
print(f"Filled 'embarked' NaNs with mode = '{mode_embarked}'")

print("\nRemaining missing values:\n", df.isnull().sum().sum(), "total nulls")

# ---------------------------------------------------------------------
# 3. Encode categorical features
# ---------------------------------------------------------------------
print("\n" + "=" * 60)
print("STEP 3: Encoding categorical features")
print("=" * 60)

# sex: binary -> label encoding (0/1)
le = LabelEncoder()
df["sex"] = le.fit_transform(df["sex"])  # female=0, male=1
print(f"Label-encoded 'sex': {dict(zip(le.classes_, le.transform(le.classes_)))}")

# embarked: 3 unordered categories -> one-hot encoding
df = pd.get_dummies(df, columns=["embarked"], prefix="embarked", drop_first=True)
print("One-hot encoded 'embarked' ->", [c for c in df.columns if c.startswith("embarked_")])

# ---------------------------------------------------------------------
# 4. Detect & remove outliers (IQR method) — done BEFORE scaling so
#    the boxplots below are in real-world units (years, $) not z-scores.
# ---------------------------------------------------------------------
print("\n" + "=" * 60)
print("STEP 4: Outlier detection (IQR method)")
print("=" * 60)

numeric_cols = ["age", "fare"]

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
for ax, col in zip(axes, numeric_cols):
    sns.boxplot(y=df[col], ax=ax)
    ax.set_title(f"{col} — before outlier removal")
plt.tight_layout()
plt.savefig("plots/boxplots_before.png", dpi=120)
plt.close()

before_rows = len(df)
for col in numeric_cols:
    q1, q3 = df[col].quantile([0.25, 0.75])
    iqr = q3 - q1
    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    n_outliers = ((df[col] < lower) | (df[col] > upper)).sum()
    print(f"{col}: bounds=({lower:.2f}, {upper:.2f}), outliers found={n_outliers}")
    df = df[(df[col] >= lower) & (df[col] <= upper)]

print(f"Rows before: {before_rows}, after outlier removal: {len(df)}")

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
for ax, col in zip(axes, numeric_cols):
    sns.boxplot(y=df[col], ax=ax)
    ax.set_title(f"{col} — after outlier removal")
plt.tight_layout()
plt.savefig("plots/boxplots_after.png", dpi=120)
plt.close()

# ---------------------------------------------------------------------
# 5. Normalize / standardize numeric features
# ---------------------------------------------------------------------
print("\n" + "=" * 60)
print("STEP 5: Feature scaling (StandardScaler)")
print("=" * 60)

scale_cols = ["age", "fare", "sibsp", "parch"]
scaler = StandardScaler()
df[scale_cols] = scaler.fit_transform(df[scale_cols])
print("Scaled columns:", scale_cols)
print(df[scale_cols].describe().loc[["mean", "std"]].round(3))

# ---------------------------------------------------------------------
# Save cleaned dataset
# ---------------------------------------------------------------------
df.to_csv("data/titanic_cleaned.csv", index=False)
print("\nSaved cleaned dataset -> data/titanic_cleaned.csv")
print("Final shape:", df.shape)
