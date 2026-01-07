import pandas as pd
import numpy as np
import lightgbm as lgb

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score

# Drop identifier
df = train.copy()
df = df.drop(columns=["id"])

# Ordinal encodings (EXPLICIT)
df["sleep_quality"] = df["sleep_quality"].map(
    {"poor": 0, "average": 1, "good": 2}
)

df["facility_rating"] = df["facility_rating"].map(
    {"low": 0, "medium": 1, "high": 2}
)

df["exam_difficulty"] = df["exam_difficulty"].map(
    {"easy": 0, "moderate": 1, "hard": 2}
)

# Explicit interaction
df["study_method_internet"] = (
    df["study_method"].astype(str)
    + "_"
    + df["internet_access"].astype(str)
).astype("category")

# Drop originals to avoid redundancy
df = df.drop(columns=["study_method", "internet_access"])

categorical_cols = [
    "gender",
    "course",
    "study_method_internet",
    # "internet_access",
    # "study_method",
    # "study_internet",
    # "sleep_quality",
    # "facility_rating",
    # "exam_difficulty"
]

for col in categorical_cols:
    df[col] = df[col].astype("category")

X = df.drop(columns=["exam_score"])
y = df["exam_score"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


train_data = lgb.Dataset(
    X_train,
    label=y_train,
    free_raw_data=False
)

valid_data = lgb.Dataset(
    X_test,
    label=y_test,
    reference=train_data,
    free_raw_data=False
)

params = {
    "objective": "regression",
    "metric": "rmse",

    # learning dynamics
    "learning_rate": 0.05,
    "num_boost_round": 600,

    # tree structure
    "num_leaves": 63,          # allow interactions, but not chaos
    "max_depth": -1,           # let leaves control depth
    "min_data_in_leaf": 100,   # stabilizes tails (important for you)

    # regularization
    "lambda_l1": 0.0,
    "lambda_l2": 1.0,          # gentle smoothing
    "min_gain_to_split": 0.0,

    # sampling
    "feature_fraction": 0.85,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,

    # misc
    "verbosity": -1,
    "seed": 42,
}


lgb_model = lgb.train(
    params,
    train_data,
    valid_sets=[valid_data],
    num_boost_round=500,
)

y_pred = lgb_model.predict(
    X_test, num_iteration=lgb_model.best_iteration
)

print("MAE:", mean_absolute_error(y_test, y_pred))
print("RMSE:", root_mean_squared_error(y_test, y_pred))
print("R2:", r2_score(y_test, y_pred))

# ------------------------------
MAE: 6.97643809562767
RMSE: 8.75253506060119
R2: 0.7845986709644195
# ------------------------------
