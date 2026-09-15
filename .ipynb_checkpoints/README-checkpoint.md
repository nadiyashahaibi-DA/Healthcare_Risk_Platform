# Healthcare Prescription Safety & Patient Risk Intelligence Platform

Ironhack Data Analytics bootcamp capstone project. A beginner-level, end-to-end healthcare analytics pipeline — cleaning, wrangling, SQL, statistics, and machine learning — built to answer four business questions about patient readmission risk and medication safety.

## Business Questions

1. Which patient groups are at highest risk for readmission?
2. Which diagnoses are most associated with repeat admissions?
3. Which medications have the highest reported adverse events?
4. Can we predict whether a patient will be readmitted?

## Data Sources

**MIMIC-IV Demo** — a public, de-identified 100-patient extract of hospital admission and diagnosis records (100 patients, 275 admissions, ~4,500 diagnosis records). Provides `patients`, `admissions`, and `diagnoses_icd` tables.

**OpenFDA Drug Adverse Event API** — a public feed of drug adverse-event reports (100 reports pulled for this project), used to explore medication safety independently.

These two sources use different identifiers (MIMIC patient IDs vs. OpenFDA safety report IDs) and there is no real key to join them at the row level, so they are kept as two separate master tables rather than forced into one — see [Design Decisions](#design-decisions) below.

## Key Insights & Findings

**Q1 — Highest-risk patients.** Using a simple, explainable rule (3+ admissions, or at least one 30-day readmission), **35 of 100 patients (35%)** are flagged high-risk. Directionally, high-risk patients skew toward younger (<30) and female patients, Medicaid insurance, and direct-emergency admissions — though with only 275 admissions these patterns are suggestive, not statistically proven (see Q2).

**Q2 — Readmission drivers.** 30-day readmission occurs after **54 of 275 admissions (~19.6%)**. Formal hypothesis tests (t-tests on age, length of stay, diagnosis count; a chi-square test on admission type) found **no statistically significant differences** between readmitted and non-readmitted admissions at this sample size (p > 0.05 across the board). This is reported honestly as a real finding: with a 100-patient demo dataset, apparent patterns don't hold up to statistical scrutiny yet.

**Q3 — Medication safety.** Ranking medications by *serious-event rate* (share of reports flagged serious, minimum 2 reports) is more informative than ranking by raw report count. Prednisone stands out with 7 reports, 100% flagged serious. Note: OpenFDA data is a public feed independent of the MIMIC patients — a real link would require prescription records tied to patient IDs, which this demo dataset doesn't provide.

**Q4 — Utilization & prediction.** Diagnosis count correlates with length of stay (r = 0.51) more strongly than age does. Admissions involving a chronic condition (diabetes, hypertension, CKD, etc.) account for 207 of 275 admissions (75.3%). A Logistic Regression and a Random Forest were trained to predict 30-day readmission from age, diagnosis count, prior admission count, and admission type (deliberately using *prior*, not lifetime, admissions to avoid data leakage). Logistic Regression outperformed Random Forest (F1 = 0.38 vs. 0.24; recall 0.69 vs. 0.31), a small-data lesson — the simpler, more regularized model generalizes better on only 275 admissions. Both models are modest and high-variance, consistent with Q2's non-significant tests: this is a working, leakage-free proof-of-concept pipeline, not a production-ready clinical risk score. Improving it would need the full MIMIC-IV dataset and richer features (vitals, labs, prior ICU stays).

## Design Decisions

- **Two master tables, not one.** `clinical_master` (275 rows, one per admission — powers Q1, Q2, Q4) and `med_adverse_event_master` (875 rows, one per report × medication × reaction — powers Q3) are kept separate because MIMIC and OpenFDA share no real join key.
- **Leakage-aware feature engineering.** The ML model uses `prior_admissions_count` (admissions *before* the current one), not a patient's lifetime total, since the lifetime total isn't knowable until after their last visit.
- **Honest reporting over overstated findings.** Non-significant statistical tests and modest model performance are reported as-is rather than hidden — an appropriately cautious read of a small demo dataset.

## Project Structure

```text
Healthcare_Risk_Platform/
├── README.md
├── requirements.txt
├── Data/
│   ├── Raw/                     # Original MIMIC-IV + OpenFDA extracts
│   └── Processed/                # Cleaned tables, master datasets, saved model
├── Notebook/
│   ├── 01_mimic_data_exploration.ipynb
│   ├── 02_openfda_data_collection.ipynb
│   ├── 03_data_wrangling_feature_engineering.ipynb
│   ├── 04_EDA_statistics.ipynb
│   └── 05_machine_learning.ipynb
├── Script/                       # Standalone, runnable .py versions of the pipeline
│   ├── clean_and_merge_data.py   #   Notebook 03 as a reproducible script
│   ├── train_readmission_model.py#   Notebook 05 as a reproducible script
│   └── README.md
├── SQL/
│   ├── Database_create.sql       # Schema (patients, admissions, diagnoses)
│   ├── Healthcare_Queries.sql    # 10 queries: joins, CTEs, window functions, CASE
│   ├── load_to_mysql.ipynb
│   └── schema.png                # ERD
|
├── Images/                       # Chart images referenced by Notebook
├── Dashboard/                    # Tableau dashboard (in progress)
└── Streamlit_app/                # Interactive app (in progress)
```

## Reproducing This Project

```bash
pip install -r requirements.txt
```

1. **Notebooks 01–02** pull and explore the raw MIMIC-IV and OpenFDA data (`Data/Raw/`).
2. **Notebook 03** (or `Script/clean_and_merge_data.py`) cleans both sources and builds the two master datasets in `Data/Processed/`.
3. **Notebook 04** runs the EDA and statistical tests behind the Key Insights above.
4. **Notebook 05** (or `Script/train_readmission_model.py`) trains and saves the readmission model.
5. **SQL/** — load `Data/Processed/*.csv` into MySQL using `Database_create.sql`, then run `Healthcare_Queries.sql`.

The notebooks are the main narrative; the `Script/` `.py` files reproduce the same cleaning and modeling pipeline without the walkthrough, for anyone who just wants to regenerate the processed data and model directly. See `Script/README.md` for details.

## Tech Stack

Python (pandas, numpy), Jupyter, scikit-learn, matplotlib/seaborn, MySQL, Streamlit, Tableau.

## Limitations

- Small cohort (100 patients / 275 admissions) — findings are directional, not statistically robust.
- OpenFDA adverse-event data is not linked to the MIMIC patient cohort.
- ICD code-to-label mapping is a partial, curated lookup, not a full crosswalk.

## Author

Nadiya — Ironhack Data Analytics Bootcamp, Final Project.
