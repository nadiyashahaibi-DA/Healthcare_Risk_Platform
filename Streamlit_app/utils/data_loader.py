"""
Cached data-loading helpers shared by every page.

Using Path(__file__) (instead of a relative string like "../Data") means
these paths resolve correctly no matter which folder `streamlit run` is
launched from - a common beginner gotcha with multi-page apps.

@st.cache_data stores the *data* (DataFrames) so each CSV is only read from
disk once per session, not on every widget interaction.
@st.cache_resource is used for the trained model instead, since scikit-learn
objects are not the kind of thing cache_data is meant to store.
"""

from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent.parent       # .../Streamlit_app
REPO_ROOT = BASE_DIR.parent                              # repo root
DATA_DIR = REPO_ROOT / "Data" / "Processed"


@st.cache_data
def load_clinical_master() -> pd.DataFrame:
    """One row per admission. Powers Q1, Q2 and Q4."""
    df = pd.read_csv(DATA_DIR / "clinical_master.csv", parse_dates=["admittime", "dischtime"])
    df["age_bracket"] = pd.cut(
        df["anchor_age"],
        bins=[0, 30, 45, 60, 75, 100],
        labels=["<30", "30-44", "45-59", "60-74", "75+"],
    )
    return df


@st.cache_data
def load_patient_summary() -> pd.DataFrame:
    """One row per patient - same rollup used in Notebook 04's Q1 analysis."""
    cm = load_clinical_master()
    summary = cm.groupby("subject_id").agg(
        age=("anchor_age", "first"),
        gender=("gender", "first"),
        insurance=("insurance", "first"),
        total_admissions=("hadm_id", "count"),
        total_diagnoses=("diagnosis_count", "sum"),
        avg_length_of_stay=("length_of_stay_days", "mean"),
        ever_readmitted_30d=("readmitted_30d", "max"),
    ).reset_index()
    summary["avg_length_of_stay"] = summary["avg_length_of_stay"].round(1)
    summary["high_risk"] = (
        (summary["total_admissions"] >= 3) | (summary["ever_readmitted_30d"] == 1)
    ).astype(int)
    return summary


@st.cache_data
def load_med_ae_master() -> pd.DataFrame:
    """One row per report x medication x reaction. Powers Q3."""
    return pd.read_csv(DATA_DIR / "med_adverse_event_master.csv")


@st.cache_data
def load_drug_events() -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / "openfda_drug_events_long.csv")


@st.cache_resource
def load_model_bundle():
    """Returns (model, scaler, feature_cols) trained in Notebook 05 /
    Script/train_readmission_model.py."""
    model = joblib.load(DATA_DIR / "readmission_model.pkl")
    scaler = joblib.load(DATA_DIR / "readmission_scaler.pkl")
    feature_cols = joblib.load(DATA_DIR / "readmission_model_features.pkl")
    return model, scaler, feature_cols
