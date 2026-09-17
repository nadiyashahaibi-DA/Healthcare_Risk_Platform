import streamlit as st
import pandas as pd

st.set_page_config(
    page_title="Healthcare Risk Intelligence Platform",
    page_icon="🏥",
    layout="wide"
)

st.title("🏥 Healthcare Risk Intelligence Platform")

st.markdown("""
Welcome to the Healthcare Prescription Safety & Patient Risk Intelligence Platform.

This project combines:

- MIMIC-IV Clinical Data
- OpenFDA Adverse Event Data

to identify patient risk factors, readmission patterns, and medication safety signals.
""")

st.markdown("---")

st.subheader("Project Business Questions")

st.markdown("""
### Q1
Who are the highest-risk patients?

### Q2
Which patient characteristics are associated with hospital readmission?

### Q3
What medication safety signals can be identified from adverse event reporting?

### Q4
Can we predict which patients are most likely to be readmitted?
""")