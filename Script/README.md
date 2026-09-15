# Script/

Standalone, runnable `.py` versions of the project's data pipeline — for reproducing the processed data and trained model without opening Jupyter.

The `Notebook/` files are the main deliverable: step-by-step, with markdown explaining the reasoning behind each decision. These scripts hold the same underlying logic, consolidated into plain functions, for anyone (a grader, a future Streamlit app, you in six months) who just wants to regenerate the outputs directly.

## Files

- **`clean_and_merge_data.py`** — reproduces `Notebook/03_data_wrangling_feature_engineering.ipynb`. Cleans the raw MIMIC-IV and OpenFDA data and builds the two master datasets.
- **`train_readmission_model.py`** — reproduces `Notebook/05_machine_learning.ipynb`. Trains Logistic Regression and Random Forest models, compares them, and saves the better one.

## Usage

Run both from inside this `Script/` folder, in order:

```bash
cd Script
python clean_and_merge_data.py       # Data/Raw  ->  Data/Processed (master datasets)
python train_readmission_model.py    # Data/Processed/clinical_master.csv  ->  readmission_model.pkl
```

**Requires:** `Data/Raw/` to already contain `patients.csv.gz`, `admissions.csv.gz`, `diagnoses_icd.csv.gz`, and `openfda_events.csv` (see `Notebook/01` and `02` for how these were obtained).

**Produces**, in `Data/Processed/`:

- `clinical_master.csv`, `med_adverse_event_master.csv` — the two master tables
- `patients_clean.csv`, `admissions_clean.csv`, `diagnoses_clean.csv`, `openfda_events_clean.csv` — individually cleaned tables
- `openfda_drug_events_long.csv`, `openfda_reaction_events_long.csv` — exploded long-format OpenFDA tables
- `readmission_model.pkl`, `readmission_scaler.pkl`, `readmission_model_features.pkl` — the trained model, scaler, and feature list

Both scripts were tested end-to-end against a cleared `Data/Processed/` folder and confirmed to regenerate identical results to the notebooks (275 admissions, 100 patients; Logistic Regression F1 = 0.38 vs. Random Forest F1 = 0.24).
