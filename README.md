# Task 1: Data Cleaning & Preprocessing — AI & ML Internship (Elevate Labs)

## Objective
Clean and prepare raw data (Titanic dataset) for machine learning.

## Tools
Python, Pandas, NumPy, Matplotlib/Seaborn, scikit-learn (`StandardScaler`, `LabelEncoder`)

## Dataset
Titanic dataset (891 rows, 15 columns), loaded via `seaborn.load_dataset("titanic")` —
same underlying data as the classic Kaggle Titanic set. A raw copy is saved at
`titanic_raw.csv`.

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

Final cleaned dataset: `/titanic_cleaned.csv` (718 rows × 9 columns).

## Files
- `preprocess.py` — full pipeline, runs end-to-end (`python3 preprocess.py`)
- `titanic_raw.csv` — original data
- `titanic_cleaned.csv` — cleaned, encoded, scaled, outlier-free data
- `boxplots_before.png`, `boxplots_after.png` — outlier visualization


