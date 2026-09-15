"""
Healthcare Prescription Safety & Patient Risk Intelligence Platform
--------------------------------------------------------------------
Reproduces the data cleaning, feature engineering and master-dataset
build from Notebook/03_data_wrangling_feature_engineering.ipynb as a
standalone, runnable script.

Run from the Script/ folder:
    python clean_and_merge_data.py

Reads:  Data/Raw/*.csv.gz, Data/Raw/openfda_events.csv
Writes: Data/Processed/*.csv (clinical_master, med_adverse_event_master,
        the individual cleaned tables, and the long-format OpenFDA tables)

See the notebook for the full reasoning behind each cleaning decision —
this script keeps that logic, just without the markdown explanations.
"""

import ast
import os

import numpy as np
import pandas as pd

RAW = "../Data/Raw"
PROCESSED = "../Data/Processed"

ICD_LABELS = {
    # Common chronic / secondary diagnoses (high overall frequency)
    "4019": "Hypertension", "I10": "Hypertension",
    "E785": "Hyperlipidemia", "2724": "Hyperlipidemia",
    "E039": "Hypothyroidism",
    "Z794": "Long-term drug therapy (insulin)",
    "Z87891": "History of nicotine dependence",
    "42731": "Atrial fibrillation", "I4891": "Atrial fibrillation",
    "25000": "Type 2 diabetes", "E119": "Type 2 diabetes",
    "I2510": "Coronary artery disease",
    "F329": "Depressive disorder",
    "N179": "Acute kidney failure",
    "K219": "GERD",
    "Z7901": "Long-term anticoagulant use",
    # Common PRIMARY (principal) diagnoses
    "41401": "Coronary atherosclerosis (native artery)",
    "4241": "Aortic valve disorder",
    "I214": "NSTEMI (heart attack)",
    "99859": "Postoperative infection",
    "5849": "Acute kidney failure",
    "5720": "Liver abscess",
    "42833": "Acute-on-chronic diastolic heart failure",
    "42823": "Acute-on-chronic systolic heart failure",
    "I25110": "Coronary artery disease with angina",
    "A419": "Sepsis, unspecified organism",
    "J9601": "Acute respiratory failure with hypoxia",
    "I5023": "Acute-on-chronic systolic heart failure",
    "34839": "Other encephalopathy",
    "5770": "Acute pancreatitis",
    "03842": "Septicemia (E. coli)",
    "5772": "Pancreatic cyst / pseudocyst",
    "41071": "Subendocardial infarction",
    "4373": "Cerebral aneurysm (nonruptured)",
}

DROP_FDA_COLS = [
    "receiver", "patient.patientdeath.patientdeathdateformat", "patient.patientdeath.patientdeathdate",
    "transmissiondateformat", "receivedateformat", "receiptdateformat",
    "companynumb", "sender.senderorganization", "sender.sendertype",
    "receiver.receivertype", "receiver.receiverorganization",
    "reportduplicate.duplicatesource", "reportduplicate.duplicatenumb",
    "safetyreportversion", "duplicate", "fulfillexpeditecriteria", "reporttype",
    "primarysource.qualification", "primarysource.reportercountry",
    "patient.summary.narrativeincludeclinical", "patient.patientweight",
    "patient.patientagegroup",
]
SERIOUSNESS_COLS = [
    "seriousnessdeath", "seriousnesshospitalization", "seriousnessdisabling",
    "seriousnesslifethreatening", "seriousnessother",
]


def clean_patients(patients: pd.DataFrame) -> pd.DataFrame:
    out = patients.copy()
    out["deceased"] = out["dod"].notna().astype(int)
    return out.drop(columns=["anchor_year", "anchor_year_group", "dod"])


def clean_admissions(admissions: pd.DataFrame) -> pd.DataFrame:
    out = admissions.copy()
    out["language"] = out["language"].replace("?", np.nan).fillna("UNKNOWN")
    out["discharge_location"] = out["discharge_location"].fillna("UNKNOWN")
    out["marital_status"] = out["marital_status"].fillna("UNKNOWN")

    out["length_of_stay_days"] = (
        (out["dischtime"] - out["admittime"]).dt.total_seconds() / 86400
    ).round(2)

    out["came_via_ed"] = out["edregtime"].notna().astype(int)
    out["ed_duration_hours"] = (
        (out["edouttime"] - out["edregtime"]).dt.total_seconds() / 3600
    ).round(2)
    out["in_hospital_death"] = out["hospital_expire_flag"]

    out = out.sort_values(["subject_id", "admittime"]).reset_index(drop=True)
    out["prior_admissions_count"] = out.groupby("subject_id").cumcount()
    out["next_admittime"] = out.groupby("subject_id")["admittime"].shift(-1)
    out["days_to_next_admit"] = (
        (out["next_admittime"] - out["dischtime"]).dt.total_seconds() / 86400
    )
    out["readmitted_30d"] = (
        (out["days_to_next_admit"] >= 0) & (out["days_to_next_admit"] <= 30)
    ).astype(int)

    return out.drop(columns=["admit_provider_id", "deathtime", "edregtime", "edouttime",
                              "next_admittime", "hospital_expire_flag"])


def clean_diagnoses(diagnoses: pd.DataFrame) -> pd.DataFrame:
    out = diagnoses.copy()
    out["is_primary_diagnosis"] = (out["seq_num"] == 1).astype(int)
    out["diagnosis_label"] = out["icd_code"].map(ICD_LABELS).fillna("Other / not mapped")
    return out


def build_clinical_master(admissions_clean, patients_clean, diagnoses_clean) -> pd.DataFrame:
    diag_agg = diagnoses_clean.groupby("hadm_id").agg(
        diagnosis_count=("icd_code", "count"),
    ).reset_index()

    primary_dx = (
        diagnoses_clean.loc[diagnoses_clean["is_primary_diagnosis"] == 1, ["hadm_id", "diagnosis_label"]]
        .rename(columns={"diagnosis_label": "primary_diagnosis"})
        .drop_duplicates(subset="hadm_id")
    )
    diag_agg = diag_agg.merge(primary_dx, on="hadm_id", how="left")
    diag_agg["primary_diagnosis"] = diag_agg["primary_diagnosis"].fillna("Unknown")

    clinical_master = (
        admissions_clean
        .merge(patients_clean, on="subject_id", how="left")
        .merge(diag_agg, on="hadm_id", how="left")
    )
    clinical_master["diagnosis_count"] = clinical_master["diagnosis_count"].fillna(0).astype(int)
    clinical_master["primary_diagnosis"] = clinical_master["primary_diagnosis"].fillna("Unknown")
    return clinical_master


def clean_openfda(fda: pd.DataFrame) -> pd.DataFrame:
    out = fda.drop(columns=[c for c in DROP_FDA_COLS if c in fda.columns]).copy()
    for c in SERIOUSNESS_COLS:
        out[c] = out[c].fillna(0).astype(int)
    out["serious"] = out["serious"].map({1: "Yes", 2: "No"})
    out = out.rename(columns={
        "patient.patientsex": "patient_sex",
        "patient.patientonsetage": "patient_age",
        "patient.patientonsetageunit": "patient_age_unit",
    })
    out["patient_sex"] = out["patient_sex"].map({1: "Male", 2: "Female"})
    return out


def _safe_eval(x):
    try:
        return ast.literal_eval(x)
    except Exception:
        return []


def explode_drug_and_reaction(fda_clean: pd.DataFrame):
    drug_rows, reaction_rows = [], []
    for _, row in fda_clean[["safetyreportid", "patient.drug"]].iterrows():
        for d in _safe_eval(row["patient.drug"]):
            drug_rows.append({
                "safetyreportid": row["safetyreportid"],
                "medicinalproduct": str(d.get("medicinalproduct", "UNKNOWN")).strip().upper(),
                "drugcharacterization": d.get("drugcharacterization"),
                "drugindication": d.get("drugindication"),
                "drugadministrationroute": d.get("drugadministrationroute"),
            })
    for _, row in fda_clean[["safetyreportid", "patient.reaction"]].iterrows():
        for r in _safe_eval(row["patient.reaction"]):
            reaction_rows.append({
                "safetyreportid": row["safetyreportid"],
                "reaction": str(r.get("reactionmeddrapt", "UNKNOWN")).strip().title(),
            })
    return pd.DataFrame(drug_rows), pd.DataFrame(reaction_rows)


def main():
    print("Loading raw data...")
    patients = pd.read_csv(f"{RAW}/patients.csv.gz")
    admissions = pd.read_csv(
        f"{RAW}/admissions.csv.gz",
        parse_dates=["admittime", "dischtime", "deathtime", "edregtime", "edouttime"],
    )
    diagnoses = pd.read_csv(f"{RAW}/diagnoses_icd.csv.gz")
    fda = pd.read_csv(f"{RAW}/openfda_events.csv", low_memory=False)

    print("Cleaning tables...")
    patients_clean = clean_patients(patients)
    admissions_clean = clean_admissions(admissions)
    diagnoses_clean = clean_diagnoses(diagnoses)
    fda_clean = clean_openfda(fda)

    print("Building clinical_master...")
    clinical_master = build_clinical_master(admissions_clean, patients_clean, diagnoses_clean)

    print("Exploding OpenFDA drug/reaction fields and building med_adverse_event_master...")
    drug_events, reaction_events = explode_drug_and_reaction(fda_clean)
    report_level = fda_clean.drop(columns=["patient.drug", "patient.reaction"])
    med_ae_master = (
        drug_events
        .merge(reaction_events, on="safetyreportid", how="left")
        .merge(report_level, on="safetyreportid", how="left")
    )

    os.makedirs(PROCESSED, exist_ok=True)
    print(f"Saving outputs to {PROCESSED} ...")
    clinical_master.to_csv(f"{PROCESSED}/clinical_master.csv", index=False)
    med_ae_master.to_csv(f"{PROCESSED}/med_adverse_event_master.csv", index=False)
    drug_events.to_csv(f"{PROCESSED}/openfda_drug_events_long.csv", index=False)
    reaction_events.to_csv(f"{PROCESSED}/openfda_reaction_events_long.csv", index=False)
    patients_clean.to_csv(f"{PROCESSED}/patients_clean.csv", index=False)
    admissions_clean.to_csv(f"{PROCESSED}/admissions_clean.csv", index=False)
    diagnoses_clean.to_csv(f"{PROCESSED}/diagnoses_clean.csv", index=False)
    fda_clean.to_csv(f"{PROCESSED}/openfda_events_clean.csv", index=False)

    print("Done. clinical_master:", clinical_master.shape, " med_adverse_event_master:", med_ae_master.shape)


if __name__ == "__main__":
    main()
