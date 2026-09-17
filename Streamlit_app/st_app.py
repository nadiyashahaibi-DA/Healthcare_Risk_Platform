"""
Healthcare Risk Intelligence Portal - Home / Overview page.

Run this app with:  streamlit run Home.py   (from inside the Streamlit_app folder)
"""

import streamlit as st

from utils import theme
from utils.data_loader import load_clinical_master, load_patient_summary

st.set_page_config(
    page_title="Healthcare Risk Intelligence Portal",
    page_icon="🏥",
    layout="wide",
)
st.markdown(theme.inject_global_css(), unsafe_allow_html=True)

st.title("🏥 Healthcare Risk Intelligence Portal")
st.markdown(
    "A patient risk and medication-safety dashboard built on the **MIMIC-IV Demo** "
    "dataset and the **OpenFDA** adverse-event feed, for the Ironhack Data Analytics "
    "capstone project."
)

st.divider()

# ---------------------------------------------------------------------------
# KPI row - the four headline numbers a reader should see in five seconds
# ---------------------------------------------------------------------------
clinical_master = load_clinical_master()
patient_summary = load_patient_summary()

n_patients = patient_summary.shape[0]
n_admissions = clinical_master.shape[0]
readmit_rate = clinical_master["readmitted_30d"].mean()
high_risk_rate = patient_summary["high_risk"].mean()

col1, col2, col3, col4 = st.columns(4)
col1.metric("Patients", f"{n_patients}")
col2.metric("Admissions", f"{n_admissions}")
col3.metric("30-Day Readmission Rate", f"{readmit_rate:.1%}")
col4.metric("High-Risk Patients", f"{high_risk_rate:.1%}")

st.divider()

# ---------------------------------------------------------------------------
# Risk split - a quick view of the patient population using the project's
# established high-risk rule (Notebook 04). This rule is binary (a patient
# either meets the high-risk criteria or doesn't), so it's shown as two
# groups here - the three-way Low/Medium/High split only appears on the
# Predict Readmission Risk page, where it comes from the model's continuous
# probability output instead of this yes/no rule. Same red for "high risk"
# in both places, so the color still means the same thing throughout the app.
# ---------------------------------------------------------------------------
st.subheader("Patient population at a glance")

standard = int((patient_summary["high_risk"] == 0).sum())
high = int(patient_summary["high_risk"].sum())

tier_col1, tier_col2 = st.columns(2)
for col, label, count, color in [
    (tier_col1, "Standard risk", standard, theme.RISK_NEUTRAL),
    (tier_col2, "High risk", high, theme.RISK_HIGH),
]:
    col.markdown(
        f"""
        <div style="border-left: 6px solid {color}; background-color:{theme.SURFACE_ALT};
                    border-radius:8px; padding:12px 16px;">
            <div style="color:{theme.INK_SECONDARY}; font-size:0.9rem;">{label}</div>
            <div style="font-size:1.8rem; font-weight:700; color:{theme.INK_PRIMARY};">{count} patients</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.caption(
    "This split uses the same simple, explainable rule as Notebook 04: "
    "**High risk** = 3+ admissions or a 30-day readmission; everyone else is "
    "**standard risk**. The *Predict Readmission Risk* page shows a finer-grained "
    "Low/Medium/High estimate for one patient at a time, from the trained model."
)

st.divider()

# ---------------------------------------------------------------------------
# Navigation guide
# ---------------------------------------------------------------------------
st.subheader("Explore the portal")
nav_col1, nav_col2 = st.columns(2)
with nav_col1:
    st.markdown(
        "**🩺 Patient Risk Explorer** — Which patient groups are at highest risk "
        "for readmission? *(Q1)*\n\n"
        "**📋 Diagnoses & Utilization** — Which diagnoses drive repeat admissions "
        "and longer stays? *(Q2, Q4)*"
    )
with nav_col2:
    st.markdown(
        "**💊 Medication Safety** — Which medications have the highest reported "
        "adverse events? *(Q3)*\n\n"
        "**🔮 Predict Readmission Risk** — Enter a patient's details to get a "
        "live readmission risk estimate. *(Q4 model)*"
    )
st.info("Use the sidebar on the left to open each page.")

st.divider()
with st.expander("About this data (please read before interpreting results)"):
    st.markdown(
        "- **MIMIC-IV Demo** is a small, de-identified extract: **100 patients, "
        "275 admissions**. Findings here are directional, not statistically robust.\n"
        "- **OpenFDA** adverse-event reports are a public feed **not linked** to "
        "these MIMIC patients - Q3 is answered independently of Q1/Q2/Q4.\n"
        "- The readmission model (see *Predict Readmission Risk*) is a "
        "transparent, leakage-free proof-of-concept, not a clinical decision tool."
    )
