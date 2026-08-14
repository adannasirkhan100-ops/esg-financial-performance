"""
Train and evaluate multiple regression models for ProfitMargin prediction.
Compares: Linear Regression (baseline), Random Forest, Gradient Boosting.
Final model chosen based on test-set RMSE/MAE/R2, then refit on train and saved.
"""
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score, KFold

from preprocess import build_preprocessor

X_train, X_test, y_train, y_test = joblib.load("data/processed/split.joblib")

models = {
    "LinearRegression": LinearRegression(),
    "Ridge": Ridge(alpha=1.0, random_state=42),
    "RandomForest": RandomForestRegressor(n_estimators=300, max_depth=10, min_samples_leaf=5, random_state=42, n_jobs=-1),
    "GradientBoosting": GradientBoostingRegressor(n_estimators=300, max_depth=3, learning_rate=0.05, random_state=42),
}

results = []
fitted_pipelines = {}

for name, model in models.items():
    pipe = Pipeline(steps=[
        ("preprocess", build_preprocessor()),
        ("model", model),
    ])
    pipe.fit(X_train, y_train)
    fitted_pipelines[name] = pipe

    preds = pipe.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)

    # 5-fold CV on training set only (grouped not strictly needed here since we
    # already isolated test companies; still uses train-only data)
    cv_scores = cross_val_score(pipe, X_train, y_train, cv=KFold(5, shuffle=True, random_state=42),
                                 scoring="neg_root_mean_squared_error")
    cv_rmse_mean = -cv_scores.mean()
    cv_rmse_std = cv_scores.std()

    results.append({
        "Model": name, "Test_MAE": mae, "Test_RMSE": rmse, "Test_R2": r2,
        "CV_RMSE_mean": cv_rmse_mean, "CV_RMSE_std": cv_rmse_std,
    })
    print(f"{name:20s} | MAE={mae:.3f}  RMSE={rmse:.3f}  R2={r2:.3f}  | CV RMSE={cv_rmse_mean:.3f}+/-{cv_rmse_std:.3f}")

results_df = pd.DataFrame(results).sort_values("Test_RMSE")
results_df.to_csv("models/model_comparison.csv", index=False)
print("\n=== Ranked by Test RMSE ===")
print(results_df.to_string(index=False))

best_name = results_df.iloc[0]["Model"]
best_pipe = fitted_pipelines[best_name]
joblib.dump(best_pipe, "models/final_model.joblib")
print(f"\nFinal selected model: {best_name} -> saved to models/final_model.joblib")

# Feature importance for the best tree-based model (if applicable) for interpretation
if best_name in ("RandomForest", "GradientBoosting"):
    feat_names = best_pipe.named_steps["preprocess"].get_feature_names_out()
    importances = best_pipe.named_steps["model"].feature_importances_
    imp_df = pd.DataFrame({"feature": feat_names, "importance": importances}).sort_values("importance", ascending=False)
    imp_df.to_csv("models/feature_importance.csv", index=False)
    print("\nTop 10 features:")
    print(imp_df.head(10).to_string(index=False))
