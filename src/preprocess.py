"""
Preprocessing pipeline for ESG & Financial Performance dataset.

Target: ProfitMargin (regression)
Features: Industry, Region, Year, Revenue, MarketCap, GrowthRate,
          ESG_Overall, ESG_Environmental, ESG_Social, ESG_Governance,
          CarbonEmissions, WaterUsage, EnergyConsumption

Design decisions:
- Split is GROUPED by CompanyID (GroupShuffleSplit), not row-wise, because each
  company appears 10 times (2016-2025) in this panel dataset. A row-wise random
  split would leak information about the same company across train and test
  (data leakage), inflating apparent performance.
- Categorical features (Industry, Region) -> one-hot encoded.
- Heavily right-skewed numeric features (Revenue, CarbonEmissions, WaterUsage,
  EnergyConsumption, MarketCap) -> log1p transformed before scaling, since their
  skew (7-16) would otherwise dominate distance/gradient-based models.
- All numeric features are standard-scaled AFTER the train/test split is defined,
  and the scaler is fit ONLY on training data (via Pipeline) to avoid leakage.
- CompanyID and CompanyName are dropped as features (identifiers, not predictive
  signal - including them would let models memorize per-company profit margin).
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import GroupShuffleSplit
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer
import joblib

TARGET = "ProfitMargin"
SKEWED_COLS = ["Revenue", "MarketCap", "CarbonEmissions", "WaterUsage", "EnergyConsumption"]
NUMERIC_COLS = ["Year", "Revenue", "MarketCap", "GrowthRate", "ESG_Overall",
                 "ESG_Environmental", "ESG_Social", "ESG_Governance",
                 "CarbonEmissions", "WaterUsage", "EnergyConsumption"]
CATEGORICAL_COLS = ["Industry", "Region"]


def load_and_split(path="data/processed/cleaned.csv", test_size=0.2, random_state=42):
    df = pd.read_csv(path)
    X = df[NUMERIC_COLS + CATEGORICAL_COLS].copy()
    y = df[TARGET].copy()
    groups = df["CompanyID"]

    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_idx, test_idx = next(gss.split(X, y, groups))

    X_train, X_test = X.iloc[train_idx].reset_index(drop=True), X.iloc[test_idx].reset_index(drop=True)
    y_train, y_test = y.iloc[train_idx].reset_index(drop=True), y.iloc[test_idx].reset_index(drop=True)

    # verify no company overlap between train/test
    overlap = set(groups.iloc[train_idx]) & set(groups.iloc[test_idx])
    assert len(overlap) == 0, f"Data leakage: {len(overlap)} companies appear in both train and test"

    return X_train, X_test, y_train, y_test


def build_preprocessor():
    log_skewed = Pipeline(steps=[
        ("log1p", FunctionTransformer(np.log1p, feature_names_out="one-to-one")),
        ("scale", StandardScaler()),
    ])
    numeric_unskewed = [c for c in NUMERIC_COLS if c not in SKEWED_COLS]

    preprocessor = ColumnTransformer(transformers=[
        ("log_skewed", log_skewed, SKEWED_COLS),
        ("scale_rest", StandardScaler(), numeric_unskewed),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL_COLS),
    ])
    return preprocessor


if __name__ == "__main__":
    X_train, X_test, y_train, y_test = load_and_split()
    print("Train shape:", X_train.shape, "Test shape:", X_test.shape)
    print("Train companies:", X_train.shape[0] // 10, "approx | Test companies:", X_test.shape[0] // 10, "approx")

    preprocessor = build_preprocessor()
    preprocessor.fit(X_train)
    feat_names = preprocessor.get_feature_names_out()
    print("Total engineered features after encoding:", len(feat_names))
    print(list(feat_names))

    joblib.dump((X_train, X_test, y_train, y_test), "data/processed/split.joblib")
    joblib.dump(preprocessor, "models/preprocessor.joblib")
    print("\nSaved split and fitted preprocessor.")
