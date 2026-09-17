# Healthcare Prescription Safety & Patient Risk Intelligence Platform

*An end-to-end healthcare analytics pipeline that predicts 30-day readmission risk and surfaces medication safety signals from real clinical and FDA data.*

> Hospitals are financially penalized for 30-day readmissions, yet most don't have a clear, data-driven view of who is most likely to come back — or whether what they're prescribing carries outsized safety risk. This project builds that view end to end: cleaning → SQL → statistics → machine learning → interactive dashboards, using two independent real-world data sources.

**Author:** Nadiya — Ironhack Data Analytics Bootcamp, Final Capstone Project
[GitHub](https://github.com/nadiyashahaibi-DA) · [LinkedIn](https://www.linkedin.com/in/nadiya-al-shahaibi-da/)

## Goal

Build a beginner-to-intermediate, portfolio-ready analytics pipeline that demonstrates the full data analyst toolkit — Python, data cleaning, data wrangling, API integration, EDA, statistics, SQL, machine learning, and two delivery layers (Streamlit and Tableau) — anchored around four real business questions rather than a generic tutorial dataset.

## Project Workflow

```text
Raw data                      Cleaning & wrangling            Master tables
┌─────────────────┐           ┌───────────────────┐           ┌─────────────────────────┐
│ MIMIC-IV Demo    │           │                    │           │ clinical_master           │
│ (patients,        │  ───▶   │  pandas cleaning,   │  ───▶    │ (275 admissions)          │
│ admissions,        │        │  feature            │          │                            │
│ diagnoses)         │        │  engineering,        │          │ med_adverse_event_master   │
│                    │        │  leakage-aware       │          │ (875 report×drug×reaction) │
│ OpenFDA Adverse     │        │  design             │           └─────────────────────────┘
│ Event API           │        └───────────────────┘                      │
└─────────────────┘                                                       │
                                                                            ▼
                                            ┌───────────────────────────────────────────────┐
                                            │  SQL (MySQL)   EDA & Statistics   ML (LogReg/RF) │
                                            └───────────────────────────────────────────────┘
                                                                            │
                                                                            ▼
                                            ┌───────────────────────────────────────────────┐
                                            │  Tableau dashboards      Streamlit app (WIP)     │
                                            └───────────────────────────────────────────────┘
```

## Research Questions

### Q1. Which patient groups are at highest risk for readmission?
### Q2. Which diagnoses are most associated with repeat admissions?
### Q3. Which medications have the highest reported adverse events?
### Q4. Can we predict whether a patient will be readmitted?

## Data Sources

| Source | Description | Size | Key fields |
|---|---|---|---|
| **MIMIC-IV Demo** | Public, de-identified hospital admission dataset | 100 patients · 275 admissions · ~4,500 diagnosis records | `subject_id`, `hadm_id`, admission type, insurance, length of stay, diagnoses |
| **OpenFDA Drug Adverse Event API** | Public feed of reported drug adverse events, pulled live via REST API | 100 safety reports · 145 distinct medications | `safetyreportid`, medication name, reaction, seriousness flags |

The two sources use different identifiers (MIMIC patient IDs vs. OpenFDA safety report IDs) with no real key to join them at the row level. Rather than force a fabricated join, they're kept as **two separate master tables** — see [Data Cleaning & Preparation](#data-cleaning--preparation).

## Data Cleaning & Preparation

**`patients`** — `dod` (69% missing) converted to a `deceased` flag, raw date dropped; de-identification artifacts (`anchor_year`, `anchor_year_group`) dropped.

**`admissions`** — `deathtime` dropped (redundant with `hospital_expire_flag`); `admit_provider_id` dropped; missing `discharge_location`/`marital_status` filled as `"UNKNOWN"`; `language` `"?"` placeholder standardized to `"UNKNOWN"`; ED timestamps collapsed into a `came_via_ed` flag + `ed_duration_hours`. Engineered `length_of_stay_days` and a true **30-day readmission flag** (`readmitted_30d`) from per-patient admission-date deltas.

**`diagnoses_icd`** — no missing values; added `is_primary_diagnosis` (`seq_num == 1`) and a curated ICD-9/ICD-10 → plain-language label lookup (~33 common codes). Admissions whose code falls outside this lookup are excluded from diagnosis-level rankings rather than lumped into a misleading "Other" bucket.

**`openfda_events`** — mostly-empty administrative columns dropped; the five `seriousness*` flags recoded to 0/1 (treating missing as "not applicable," not "unknown," rather than dropping them); `patient.drug` and `patient.reaction` exploded into long-format tables so each medication/reaction is analyzable at its own grain, counted by **distinct safety report** rather than raw row to avoid double-counting.

Two master tables, not one — kept separate because MIMIC and OpenFDA share no real join key:
- **`clinical_master`** (275 rows, one per admission) — powers Q1, Q2, Q4.
- **`med_adverse_event_master`** (875 rows, one per report × medication × reaction) — powers Q3.

## Feature Engineering

- `readmitted_30d` — target variable, derived from admission-date deltas per patient
- `prior_admissions_count` — admissions *before* the current one (deliberately not lifetime total, to avoid leaking the future into the model)
- `length_of_stay_days`, `came_via_ed`
- `is_primary_diagnosis`, chronic-condition flag (diabetes, hypertension, CKD, etc., from the ICD label lookup)
- One-hot encoded `admission_type`

## SQL Integration

`clinical_master` and related tables are loaded into MySQL (`SQL/load_to_mysql.ipynb`, via `sqlalchemy`/`pymysql`) against a schema defined in `SQL/Database_create.sql` (patients, admissions, diagnoses — see `SQL/schema.png` for the ERD). `SQL/Healthcare_Queries.sql` holds 10 queries covering joins, CTEs, window functions, and CASE-based bucketing, used to validate the same findings reported below independently of the Python pipeline.

## Exploratory Data Analysis & Statistical Methodology

- **T-tests** — age, length of stay, and diagnosis count compared between readmitted and non-readmitted admissions.
- **Chi-square test** — admission type vs. readmission.
- **Correlation analysis** — diagnosis count vs. length of stay (r = 0.51, the strongest relationship found, stronger than age).

All hypothesis tests were **not statistically significant** (p > 0.05) at this sample size — reported as-is rather than hidden, since an honest null result is still a finding.

## Key Findings

### Q1 — Which patient groups are at highest risk for readmission?

Using a simple, explainable rule (3+ admissions, or at least one 30-day readmission), **35 of 100 patients (35%)** are flagged high-risk.

| Group | Readmission rate | Note |
|---|---|---|
| Age < 30 | 30% | n = 10 admissions — directional |
| Age 45–59 | 22.5% | larger base — meaningful |
| Age 60–74 | 21.3% | larger base — meaningful |
| Direct emergency admission | 33% | highest of any admission type |
| ER admission | 24% | |
| Medicaid | ~32% | |
| Medicare | 14.4% | |

Age, admission type, and insurance each show a real gap, but none alone explains risk — which motivates the predictive model in Q4.

### Q2 — Which diagnoses are most associated with repeat admissions?

30-day readmission occurs after **53 of 275 admissions (~19.3%)**. Top 10 diagnoses by volume:

| Diagnosis | Admissions | Readmission rate |
|---|---|---|
| Coronary atherosclerosis | 7 | 0% |
| Acute kidney failure | 7 | 28.6% |
| Systolic heart failure | 6 | 0% |
| NSTEMI | 4 | 0% |
| Atrial fibrillation | 4 | 25% |
| Aortic valve disorder | 4 | 0% |
| Postoperative infection | 4 | 25% |
| Other encephalopathy | 3 | 33.3% |
| CAD with angina | 3 | 33.3% |
| Acute pancreatitis | 3 | 0% |

Acute kidney failure is the most robust signal here — real volume (7 admissions) *and* an elevated readmit rate. Formal hypothesis tests (above) found no statistically significant differences overall at this sample size — an honest limit of a 100-patient demo dataset, not a hidden result.

### Q3 — Which medications have the highest reported adverse events?

Counting **distinct safety reports** (not raw rows, so a report mentioning the same drug or reaction twice isn't double-counted):

| Medication | Reports | Share of all 100 reports |
|---|---|---|
| Letairis | 61 | 61% |
| Prednisone | 7 | 7% |
| Casodex | 3 | 3% |
| 7 other medications | 2 each | — |
| 122 other medications | 1 each | — |

Most commonly reported reactions: Dyspnoea (11 reports), Headache (8), Diarrhoea (6), Unevaluable Event (6), Malaise (6). Reports by seriousness category: Hospitalization (21), Other (14), Death (3), Life-threatening (3), Disabling (2).

Letairis's dominance reflects **reporting concentration** — a handful of reporters filing repeatedly — not necessarily a drug-specific safety signal; 122 of 145 distinct medications have only a single report. OpenFDA is a public feed, independent of the 100 MIMIC patients above — this question is answered on its own, not linked patient-by-patient.

### Q4 — Can we predict whether a patient will be readmitted?

Logistic Regression and Random Forest were trained on `anchor_age`, `diagnosis_count`, `prior_admissions_count`, and one-hot encoded `admission_type` — admission-time-only features, to avoid leaking the future into the prediction.

| Model | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| Logistic Regression | 58.0% | 26.5% | **69.2%** | **38.3%** |
| Random Forest | 62.3% | 19.0% | 30.8% | 23.5% |

Random Forest edges out on accuracy, but **Logistic Regression was chosen deliberately**: it catches more than twice as many true readmissions (69% vs. 31% recall) — in a readmission-screening context, missing an at-risk patient is costlier than a false alarm.

Strongest predictors (Logistic Regression coefficients): emergency-route admission types dominate — `EW EMER.` (+0.52), `DIRECT EMER.` (+0.36), `OBSERVATION ADMIT` (+0.32), `URGENT` (+0.30) — followed by `prior_admissions_count` (+0.24). `anchor_age` (−0.26) and `EU OBSERVATION` (−0.25) lower predicted risk; `diagnosis_count` alone barely moves the needle once prior admissions are already in the model.

## Business Recommendations

- **Prioritize discharge follow-up for direct-emergency and Medicaid-insured patients** — the two largest observed readmission gaps in this cohort.
- **Flag acute kidney failure admissions for structured follow-up** — the one diagnosis combining meaningful volume and an elevated readmit rate.
- **Treat the Letairis/Prednisone adverse-event concentration as a reporting-volume artifact to monitor**, not a confirmed drug-safety signal, until adverse events can be linked to actual prescriptions.
- **Use the Logistic Regression model as a recall-first screening flag**, routed to a human case reviewer — not as a standalone clinical decision.

## Limitations

- Small cohort (100 patients / 275 admissions) — findings and model metrics are directional, not statistically robust (F1 = 0.38 on a 275-row dataset).
- OpenFDA adverse-event data is a public feed, not linked to the MIMIC patient cohort.
- ICD code-to-label mapping is a partial, curated lookup (~33 codes), not a full crosswalk.
- The Streamlit app (below) is scaffolded but not yet built out.

## Next Steps

1. Finish and deploy the Streamlit app for live, single-patient risk scoring — the trained model, scaler, and feature list are already saved in `Data/Processed/`.
2. Validate the risk rule and model on a larger, prospective patient cohort.
3. Link medication/prescription history directly to patient records instead of an independent public feed.
4. Track model performance over time as new admissions come in.

## Technologies Used

| Category | Tools |
|---|---|
| Language | Python (pandas, numpy) |
| Data visualization | matplotlib, seaborn, plotly, Tableau |
| Machine learning | scikit-learn, joblib |
| Database | MySQL (sqlalchemy, pymysql) |
| Dashboards / app | Tableau Public, Streamlit *(in progress)* |
| Environment | Jupyter, python-dotenv |
| Data collection | requests (OpenFDA REST API) |

## Repository Structure

```text
Healthcare_Risk_Platform/
├── README.md
├── requirements.txt
├── Data/
│   ├── Raw/                          # Original MIMIC-IV + OpenFDA extracts
│   └── Processed/                    # Cleaned tables, master datasets, saved model
├── Notebook/
│   ├── 01_mimic_data_exploration.ipynb
│   ├── 02_openfda_data_collection.ipynb
│   ├── 03_data_wrangling_feature_engineering.ipynb
│   ├── 04_EDA_statistics.ipynb
│   └── 05_machine_learning.ipynb
├── Script/                           # Standalone, runnable .py versions of the pipeline
│   ├── clean_and_merge_data.py       #   Notebook 03 as a reproducible script
│   ├── train_readmission_model.py    #   Notebook 05 as a reproducible script
│   └── README.md
├── SQL/
│   ├── Database_create.sql           # Schema (patients, admissions, diagnoses)
│   ├── Healthcare_Queries.sql        # 10 queries: joins, CTEs, window functions, CASE
│   ├── load_to_mysql.ipynb
│   └── schema.png                    # ERD
├── Images/                           # Chart exports referenced by the notebooks
├── Dashboard/                        # 4 Tableau workbooks (.twbx)
├── Presentation/                     # Slide deck + speaker script
└── Streamlit_app/                    # Interactive risk-scoring app (in progress)
```

## Installation

```bash
git clone https://github.com/nadiyashahaibi-DA/Healthcare_Risk_Platform.git
cd Healthcare_Risk_Platform
pip install -r requirements.txt
```

1. **Notebooks 01–02** pull and explore the raw MIMIC-IV and OpenFDA data (`Data/Raw/`).
2. **Notebook 03** (or `Script/clean_and_merge_data.py`) cleans both sources and builds the two master datasets in `Data/Processed/`.
3. **Notebook 04** runs the EDA and statistical tests behind the Key Findings above.
4. **Notebook 05** (or `Script/train_readmission_model.py`) trains and saves the readmission model.
5. **SQL/** — load `Data/Processed/*.csv` into MySQL using `Database_create.sql`, then run `Healthcare_Queries.sql`.

The notebooks are the main narrative, with markdown explaining the reasoning behind each decision; the `Script/` `.py` files reproduce the same pipeline without the walkthrough, for regenerating the processed data and model directly. See `Script/README.md` for details.

## Live Dashboards & Presentation

All four Tableau workbooks live in `Dashboard/`. The two most central to the findings above are published live:

- **Medication Safety Intelligence (Q3)** — [public.tableau.com/.../03_Medication_Safety_Intelligence](https://public.tableau.com/app/profile/nadiya.al.shahaibi/viz/03_Medication_Safety_Intelligence/MedicationSafetyIntelligence)
- **Predictive Risk Modeling (Q4)** — [public.tableau.com/.../04_Predictive_Analysis](https://public.tableau.com/app/profile/nadiya.al.shahaibi/viz/04_Predictive_Analysis/ReadmissionRiskModeling)

The full slide deck and speaker script are in `Presentation/`.

## Conclusion

This project demonstrates a complete, leakage-aware analytics pipeline — from two messy real-world sources to a validated, honestly-reported model — built to answer four concrete business questions rather than to chase a polished-looking metric. The headline results (a 35% high-risk patient share, a 19.3% readmission rate with no single dominant driver, a concentrated medication-safety signal, and a recall-optimized predictive model) are all reported with their actual limitations attached. The next phase is scale: a larger cohort, linked medication history, and a deployed Streamlit app to put the model in front of an actual user.
