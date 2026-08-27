# Task 1: Data Cleaning & Preprocessing — AI & ML Internship (Elevate Labs)

## Objective
Clean and prepare raw data (Titanic dataset) for machine learning.

## Tools
Python, Pandas, NumPy, Matplotlib/Seaborn, scikit-learn (`StandardScaler`, `LabelEncoder`)

## Dataset
Titanic dataset (891 rows, 15 columns), loaded via `seaborn.load_dataset("titanic")` —
same underlying data as the classic Kaggle Titanic set. A raw copy is saved at
`data/titanic_raw.csv`.

## What was done

1. **Explored the data** — checked shape, dtypes, and null counts. Dropped `deck`
   (77% missing), plus `embark_town`, `class`, `who`, `adult_male`, `alive`, `alone`
   since they're redundant re-encodings of columns already kept (`embarked`, `pclass`,
   `sex`, `survived`).
2. **Handled missing values**
   - `age` (177 nulls, numeric, skewed) → filled with the **median** (28.0), which
     is robust to outliers unlike the mean.
   - `embarked` (2 nulls, categorical) → filled with the **mode** (`'S'`).
3. **Encoded categorical features**
   - `sex` → **label encoding** (binary: female=0, male=1).
   - `embarked` → **one-hot encoding** (`embarked_Q`, `embarked_S`, with `C` as the
     dropped baseline) since it has 3 unordered categories.
4. **Detected & removed outliers** using the **IQR method** (1.5×IQR bounds) on
   `age` and `fare`, with boxplots before/after saved to `plots/`.
   - `age`: bounds (2.5, 54.5) → 66 outliers removed
   - `fare`: bounds (-25.37, 63.33) → 107 outliers removed
   - Rows: 891 → 718 after removal
5. **Scaled numeric features** (`age`, `fare`, `sibsp`, `parch`) with
   `StandardScaler` (mean 0, std 1).

Final cleaned dataset: `data/titanic_cleaned.csv` (718 rows × 9 columns).

## Files
- `preprocess.py` — full pipeline, runs end-to-end (`python3 preprocess.py`)
- `data/titanic_raw.csv` — original data
- `data/titanic_cleaned.csv` — cleaned, encoded, scaled, outlier-free data
- `plots/boxplots_before.png`, `plots/boxplots_after.png` — outlier visualization

## Interview Questions

**1. What are the different types of missing data?**
- MCAR (Missing Completely At Random) — missingness unrelated to any value.
- MAR (Missing At Random) — missingness related to other observed variables (e.g.
  `age` missing more often for a certain ticket class).
- MNAR (Missing Not At Random) — missingness related to the value itself (e.g.
  people with very high fares declining to disclose it).

**2. How do you handle categorical variables?**
Encode them into numbers: label encoding for ordinal/binary categories, one-hot
encoding for unordered categories with few levels, and target/frequency encoding
for high-cardinality categories.

**3. What is the difference between normalization and standardization?**
Normalization (min-max scaling) rescales values into a fixed range, typically
[0, 1]. Standardization rescales to mean 0 and standard deviation 1, and doesn't
bound the range — better when data isn't uniformly distributed or has outliers.

**4. How do you detect outliers?**
Statistical methods like the IQR rule (values beyond Q1 − 1.5×IQR or Q3 + 1.5×IQR)
or z-score (|z| > 3), and visually with boxplots or scatter plots.

**5. Why is preprocessing important in ML?**
Most models assume clean, numeric, consistently-scaled input. Missing values,
unencoded categories, or wildly different feature scales can break training or
bias the model toward high-magnitude features, so preprocessing directly affects
model validity and accuracy.

**6. What is one-hot encoding vs label encoding?**
Label encoding assigns each category an integer (0, 1, 2…), which implies an
order — fine for binary or truly ordinal data. One-hot encoding creates a separate
binary column per category, avoiding a false sense of order — better for unordered
categories.

**7. How do you handle data imbalance?**
Resampling (oversample the minority class, e.g. SMOTE, or undersample the
majority), class-weighted loss functions, or choosing evaluation metrics
(precision/recall/F1, ROC-AUC) that aren't misleading under imbalance.

**8. Can preprocessing affect model accuracy?**
Yes — significantly. Poor handling of missing values, leaving categories
unencoded, unscaled features (which distort distance-based models like KNN/SVM),
or leaving in outliers can all degrade accuracy, sometimes more than the choice
of model itself.
