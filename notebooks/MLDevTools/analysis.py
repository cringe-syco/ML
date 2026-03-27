# =========================================
# END-TO-END DATA ANALYSIS SCRIPT (EDA ONLY)
# =========================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import os

from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.impute import SimpleImputer

# =========================
# LOGGING SETUP
# =========================
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()

report = []

# =========================
# LOAD DATA
# =========================
file_path = input("Enter dataset path (CSV/Excel): ")

if file_path.endswith('.csv'):
    df = pd.read_csv(file_path)
elif file_path.endswith('.xlsx'):
    df = pd.read_excel(file_path)
elif file_path.endswith('.parquet'):
    df = pd.read_parquet(file_path)
else:
    raise ValueError("Unsupported file format")

logger.info("Dataset Loaded Successfully")

print("\n===== BASIC INFO =====")
print(df.head())
print(df.info())

report.append(f"Dataset Shape: {df.shape}")
report.append(f"\nColumn Types:\n{df.dtypes}")

# =========================
# IDENTIFY FEATURE TYPES
# =========================
numerical_cols = df.select_dtypes(include=np.number).columns.tolist()
categorical_cols = df.select_dtypes(include='object').columns.tolist()
datetime_cols = df.select_dtypes(include=['datetime', 'datetime64']).columns.tolist()

# Try converting object → datetime
for col in df.columns:
    if col not in datetime_cols:
        try:
            df[col] = pd.to_datetime(df[col])
            datetime_cols.append(col)
            if col in categorical_cols:
                categorical_cols.remove(col)
        except:
            pass

print("\nNumerical:", numerical_cols)
print("Categorical:", categorical_cols)
print("Datetime:", datetime_cols)

# =========================
# MISSING VALUES
# =========================
print("\n===== MISSING VALUES =====")
missing = df.isnull().sum().sort_values(ascending=False)
missing_pct = (missing / len(df)) * 100

missing_df = pd.DataFrame({
    'Missing Count': missing,
    'Missing %': missing_pct
})

print(missing_df)
report.append("\nMissing Values:\n" + str(missing_df))

# Visualization
plt.figure(figsize=(10,5))
missing_pct.plot(kind='bar')
plt.title("Missing Value Percentage")
plt.show()

# =========================
# NUMERICAL ANALYSIS
# =========================
print("\n===== NUMERICAL ANALYSIS =====")

for col in numerical_cols:
    print(f"\n--- {col} ---")
    desc = df[col].describe()
    skew = df[col].skew()
    print(desc)
    print(f"Skewness: {skew}")

    report.append(f"\n{col} stats:\n{desc}")
    report.append(f"{col} skewness: {skew}")

    # Histogram
    sns.histplot(df[col], kde=True)
    plt.title(f"Distribution of {col}")
    plt.show()

    # Boxplot (Outliers)
    sns.boxplot(x=df[col])
    plt.title(f"Outliers in {col}")
    plt.show()

# =========================
# CORRELATION
# =========================
if len(numerical_cols) > 1:
    print("\n===== CORRELATION MATRIX =====")
    corr = df[numerical_cols].corr()

    plt.figure(figsize=(10,8))
    sns.heatmap(corr, annot=True, cmap='coolwarm')
    plt.title("Correlation Matrix")
    plt.show()

    report.append("\nCorrelation Matrix:\n" + str(corr))

    # High correlation detection
    high_corr = []
    threshold = 0.85

    for i in range(len(corr.columns)):
        for j in range(i):
            if abs(corr.iloc[i, j]) > threshold:
                high_corr.append((corr.columns[i], corr.columns[j], corr.iloc[i, j]))

    report.append("\nHighly Correlated Features (>0.85):\n" + str(high_corr))

# =========================
# CATEGORICAL ANALYSIS
# =========================
print("\n===== CATEGORICAL ANALYSIS =====")

for col in categorical_cols:
    print(f"\n--- {col} ---")
    vc = df[col].value_counts()
    print(vc)

    report.append(f"\n{col} value counts:\n{vc}")

    vc.plot(kind='bar')
    plt.title(f"Distribution of {col}")
    plt.show()

# =========================
# DATETIME ANALYSIS
# =========================
print("\n===== DATETIME ANALYSIS =====")

for col in datetime_cols:
    df[col] = pd.to_datetime(df[col], errors='coerce')

    df[f"{col}_year"] = df[col].dt.year
    df[f"{col}_month"] = df[col].dt.month
    df[f"{col}_day"] = df[col].dt.day
    df[f"{col}_weekday"] = df[col].dt.weekday

    print(f"Extracted features from {col}")

    report.append(f"\nDatetime features created from {col}")

# =========================
# OUTLIER CHECK (Z-SCORE)
# =========================
print("\n===== OUTLIER CHECK =====")

for col in numerical_cols:
    z = np.abs((df[col] - df[col].mean()) / df[col].std())
    outliers = (z > 3).sum()
    print(f"{col}: {outliers} outliers")

    report.append(f"{col}: {outliers} outliers")

# =========================
# SKEWNESS HANDLING SUGGESTION
# =========================
print("\n===== SKEWNESS CHECK =====")

for col in numerical_cols:
    skew = df[col].skew()
    if abs(skew) > 1:
        print(f"{col} is highly skewed. Consider log transform.")
        report.append(f"{col} skewed → recommend log transform")

# =========================
# SCALING (OPTIONAL)
# =========================
scale_choice = input("\nApply scaling? (standard/minmax/robust/none): ")

if scale_choice != "none":
    if scale_choice == "standard":
        scaler = StandardScaler()
    elif scale_choice == "minmax":
        scaler = MinMaxScaler()
    else:
        scaler = RobustScaler()

    df[numerical_cols] = scaler.fit_transform(df[numerical_cols])
    print("Scaling applied")

    report.append(f"Scaling applied: {scale_choice}")

# =========================
# SAVE CLEAN DATA (OPTIONAL)
# =========================
save_choice = input("\nSave processed dataset? (yes/no): ")

if save_choice == "yes":
    df.to_parquet("processed_data.parquet", index=False)
    print("Saved as processed_data.parquet")

# =========================
# SAVE REPORT
# =========================
with open("eda_report.txt", "w") as f:
    for line in report:
        f.write(line + "\n")

logger.info("EDA Report Saved")

print("\n✅ EDA Completed Successfully!")