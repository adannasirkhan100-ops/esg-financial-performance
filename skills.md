# skills.md

Reusable workflow instructions and project skills used while building the
**Margin Ledger** ESG → Profit Margin regression project with Claude.

This file documents the repeatable process so it can be re-applied to a
similar tabular ML + static-deployment project in the future.

---

## 1. Dataset scouting skill

**When to use:** picking a dataset for a supervised ML assignment with a
"recent / real-world" requirement.

**Steps:**
1. Web-search `<domain> dataset <year> kaggle regression/classification` to
   surface 2025/2026-published candidates rather than defaulting to
   well-known legacy datasets (Titanic, Iris, Boston housing).
2. Fetch/skim the dataset's own description page for: size, target column,
   feature list, license, and whether it's real or synthetic data.
3. Prefer a dataset with (a) a clearly defined numeric or categorical
   target, (b) a mix of numeric and categorical features (so encoding work
   is real, not trivial), and (c) enough rows to support a train/test split
   without starving either side.
4. If the dataset is synthetic, say so explicitly in the report rather than
   presenting it as scraped real-world data — synthetic-but-realistic is
   an acceptable and honest choice for a course project.

## 2. Panel-data leakage check skill

**When to use:** any dataset where the same entity (company, user, patient)
appears in multiple rows (e.g. one row per year).

**Rule:** never use a plain random row-wise train/test split on panel data.
Use a **grouped split** (`GroupShuffleSplit` keyed on the entity ID) so the
same entity never appears in both train and test. Verify with an explicit
assertion that the train and test entity-ID sets do not intersect before
trusting any evaluation metric.

## 3. Encode-then-scale-then-split-safe pipeline skill

**When to use:** any tabular regression/classification pipeline.

**Steps:**
1. Identify skewed numeric columns via `.skew()` (|skew| > ~1 is a signal)
   and log-transform them before scaling.
2. Wrap categorical encoding (OneHotEncoder) and numeric scaling
   (StandardScaler) inside a single `sklearn.compose.ColumnTransformer`,
   itself inside a `sklearn.pipeline.Pipeline` with the model as the final
   step. This guarantees the scaler/encoder is fit only on training data
   when `.fit()` is called on the pipeline with `X_train`, preventing
   leakage from the test set into preprocessing statistics.
3. Never call `.fit()` or `.fit_transform()` on the full dataset before
   splitting.

## 4. Model comparison skill

**When to use:** "compare two or more models" requirements.

**Steps:**
1. Always include one simple, interpretable baseline (Linear/Ridge
   Regression or Logistic Regression) alongside at least one ensemble
   model (Random Forest, Gradient Boosting).
2. Evaluate on a held-out test set AND with k-fold cross-validation on the
   training set only, to sanity-check that test performance isn't a lucky
   split.
3. Select the final model by the primary metric (RMSE for regression) but
   report all models' metrics in the report/README for transparency —
   don't hide models that performed worse.

## 5. Deployable-model export skill (for static hosting)

**When to use:** deploying a trained scikit-learn model to a platform that
only serves static files (e.g. Netlify without serverless functions).

**Steps:**
1. Recognize that tree ensembles (Random Forest, Gradient Boosting) are not
   easily portable to client-side JavaScript, but a linear/ridge model is:
   its inference is just `intercept + sum(coefficient * scaled_feature)`.
2. Fit an interpretable linear model using the *exact same* preprocessing
   pipeline as the best model, export its intercept, coefficients, and the
   preprocessing statistics (means/scales for scaling, category lists for
   one-hot) to a small JSON file.
3. Re-implement the identical transform (log1p, standardize, one-hot) in
   JavaScript, reading from that JSON — never hand-derive new coefficients.
4. **Verify numerically**: run the same input row through the Python
   pipeline and the JS port, and confirm the predictions match to several
   decimal places before trusting the deployed app. (Done here via Node.js:
   sklearn output `4.953498973521278` vs JS output `4.953498973521279`.)
5. Be explicit in the UI and report about which model is actually deployed
   (the portable linear model) versus which model was selected as best in
   offline evaluation (the ensemble) — don't let the deployed demo imply
   it's running the winning model if it isn't.

## 6. Claude Artifact / Connector usage in this project

- **Claude Artifact:** the interactive "Margin Ledger" prediction UI
  (`app/index.html`, `app/style.css`, `app/app.js`) was built and iterated
  on with Claude directly in this chat, including the visual design pass
  (typography, color, layout) and the client-side inference logic.
- **Connector:** none of Anthropic's first-party connectors (Google Drive,
  Slack, etc.) were relevant to this project's data source (a static Kaggle
  CSV), so no connector was used. If a live financial data API were used
  instead, the appropriate connector step would be documented here.

## 7. Report-writing skill

**When to use:** writing the final PDF report.

**Steps:**
1. Write the report from the actual pipeline artifacts (metrics CSVs,
   plots, coefficient files) rather than from memory — pull real numbers.
2. Interpret metrics in plain language (what does RMSE=6.6 percentage
   points actually mean for someone reading a profit margin estimate?)
   rather than only listing scores.
3. State limitations honestly (e.g. R² of 0.44 means the model explains
   less than half the variance — useful as a directional estimate, not a
   precise forecast).
