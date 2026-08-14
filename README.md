# Margin Ledger — ESG-Linked Profit Margin Prediction

Does sustainability performance predict financial performance? This project
trains a regression model on 10,000 company-years of ESG and financial data
to estimate a company's **profit margin** from its industry, region,
financials, and ESG scores, and deploys an interactive estimator as a
static web app.

**Live app:** _add your Netlify URL here after deploying (see Deployment
below)_

---

## 1. Project overview

- **Task:** supervised regression — predict `ProfitMargin` (%)
- **Question it answers:** given a company's industry, region, size, growth,
  and ESG (Environmental / Social / Governance) scores, what profit margin
  is typical for companies with that profile?
- **Models compared:** Linear Regression, Ridge Regression, Random Forest,
  Gradient Boosting
- **Selected model (best offline performance):** Gradient Boosting
  (Test RMSE 6.62, R² 0.44)
- **Deployed model (client-side, static hosting):** Ridge Regression, ported
  to JavaScript — see [Deployment architecture](#5-deployment-architecture)
  for why these differ.

## 2. Dataset source

- **Name:** ESG & Financial Performance Dataset
- **Publisher:** Shriyash Jagtap, via Kaggle
- **URL:** https://www.kaggle.com/datasets/shriyashjagtap/esg-and-financial-performance-dataset
- **Access date:** August 2026
- **License:** see Kaggle dataset page
- **Size:** 11,000 rows × 16 columns (1,000 companies × 11 years, 2015–2025);
  10,000 rows after cleaning (see below)
- **Nature:** synthetic-but-realistic data simulating ESG and financial
  performance for 1,000 global companies across 9 industries and 7 regions.
  This is disclosed explicitly rather than presented as scraped real-company
  financials — see `report.pdf` §4 for the full discussion of this choice.

### Feature dictionary

| Column | Type | Description |
|---|---|---|
| CompanyID / CompanyName | identifier | dropped before modeling |
| Industry | categorical (9) | company's industry sector |
| Region | categorical (7) | company's global region |
| Year | numeric | reporting year, 2015–2025 |
| Revenue | numeric (USD millions) | annual revenue |
| ProfitMargin | numeric (%) | **target** — net profit margin |
| MarketCap | numeric (USD millions) | market capitalization |
| GrowthRate | numeric (%) | year-over-year revenue growth |
| ESG_Overall / Environmental / Social / Governance | numeric (0–100) | ESG sub-scores and composite |
| CarbonEmissions | numeric (t CO₂e) | annual carbon emissions |
| WaterUsage | numeric (m³) | annual water usage |
| EnergyConsumption | numeric (MWh) | annual energy consumption |

## 3. Setup & running locally

```bash
git clone <this-repo-url>
cd esg-financial-performance
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

Run the pipeline in order:

```bash
python3 src/preprocess.py   # cleans, splits, saves data/processed/split.joblib
python3 src/train.py        # trains 4 models, saves models/final_model.joblib
```

Preview the static app locally:

```bash
cd app
python3 -m http.server 8000
# open http://localhost:8000
```

## 4. Preprocessing summary

- **Missing values:** `GrowthRate` was missing for exactly the 1,000 rows
  corresponding to each company's first year (2015) — structural, since
  growth rate needs a prior year. These 1,000 rows were **dropped** rather
  than imputed (10,000 rows remain).
- **Duplicates / invalid values:** none found (0 duplicate rows, all ESG
  scores in [0,100], all financial magnitudes positive).
- **Encoding:** `Industry` (9 categories) and `Region` (7 categories)
  one-hot encoded → 16 binary columns.
- **Skew handling:** `Revenue`, `MarketCap`, `CarbonEmissions`, `WaterUsage`,
  `EnergyConsumption` had skewness of 7–16 (heavily right-skewed) and were
  log1p-transformed before scaling.
- **Scaling:** all numeric features standardized (zero mean, unit variance)
  using a `StandardScaler` fit only on the training split.
- **Split:** 80/20, **grouped by CompanyID** (not row-wise) using
  `GroupShuffleSplit`, so no company's data leaks between train and test.
  Verified programmatically: zero company-ID overlap between splits.

## 5. Deployment architecture

The assignment requires a Netlify (static-hosting) deployment. Static
hosting has no Python runtime, so the trained scikit-learn pipeline itself
cannot run in the browser. Two options were considered:

1. Stand up a separate backend API to serve the winning model
   (Gradient Boosting) — adds infrastructure and secret-management surface
   area for a course project.
2. **Port an interpretable linear model to JavaScript** and run inference
   entirely client-side, with no backend, no API keys, no secrets.

Option 2 was chosen. A Ridge Regression model was fit using the identical
preprocessing pipeline as the other models, and its intercept, coefficients,
and preprocessing statistics (log/scale means and standard deviations,
one-hot category lists) were exported to `app/model_coefficients.json`.
`app/app.js` reimplements the exact same transform in JavaScript.

**Verification:** the same input row was run through the Python pipeline
and the JS port; predictions matched to 12 significant figures
(`4.953498973521278` vs `4.953498973521279`), confirming the port is
faithful, not approximate.

The deployed app is transparent about this trade-off: it displays both the
live Ridge estimate and the offline-evaluated metrics for the actual
best-performing model (Gradient Boosting), so a user isn't misled about
which model is running where.

### Deploying to Netlify

The `app/` folder is a self-contained static site (no build step):

1. Go to [app.netlify.com/drop](https://app.netlify.com/drop)
2. Drag the `app/` folder in
3. Netlify gives you a live URL — paste it at the top of this README and
   in `report.pdf`

No environment variables, API keys, or secrets are required or present in
this repository.

## 6. Claude Artifact & workflow notes

See `skills.md` for the full reusable-skills writeup. In short: the
prediction UI (`app/`) was designed and built interactively with Claude in
this chat, including the visual design pass and the client-side inference
logic, which was numerically verified against the Python model rather than
assumed correct.

## 7. Evaluation results

| Model | Test MAE | Test RMSE | Test R² | CV RMSE (train, 5-fold) |
|---|---|---|---|---|
| **Gradient Boosting** (selected) | 4.55 | **6.62** | **0.441** | 6.15 ± 0.24 |
| Random Forest | 4.42 | 6.89 | 0.395 | 5.57 ± 0.16 |
| Ridge (deployed) | 4.73 | 6.89 | 0.395 | 7.03 ± 0.21 |
| Linear Regression | 4.76 | 6.90 | 0.393 | 7.03 ± 0.20 |

**In plain language:** the model's typical prediction is off by about
6.6 percentage points of profit margin, and explains roughly 44% of the
variance in profit margin across companies. That means ESG scores,
industry, region, and financial scale together carry real but partial
signal about profitability — useful as a directional estimate, not a
precise forecast. See `report.pdf` §10 for the full interpretation,
residual plots, and feature importance discussion.

## 8. Repository structure

```
esg-financial-performance/
├── data/
│   ├── raw/                # original, unmodified CSV
│   └── processed/          # cleaned.csv, split.joblib
├── notebooks/               # (optional) exploratory notebooks
├── src/
│   ├── preprocess.py        # cleaning + encoding + leakage-safe split
│   └── train.py              # trains & compares 4 models, saves best
├── app/                      # static Netlify-deployable prediction UI
│   ├── index.html
│   ├── style.css
│   ├── app.js
│   └── model_coefficients.json
├── models/
│   ├── final_model.joblib          # best model (Gradient Boosting)
│   ├── preprocessor.joblib
│   ├── model_comparison.csv
│   └── feature_importance.csv
├── screenshots/               # evaluation plots, app screenshots
├── skills.md
├── README.md
├── requirements.txt
└── report.pdf
```
