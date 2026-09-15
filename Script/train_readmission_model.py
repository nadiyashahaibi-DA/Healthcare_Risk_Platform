"""
Healthcare Prescription Safety & Patient Risk Intelligence Platform
--------------------------------------------------------------------
Reproduces the readmission prediction model from
Notebook/05_machine_learning.ipynb as a standalone, runnable script.

Run from the Script/ folder (after clean_and_merge_data.py has been run
at least once, so Data/Processed/clinical_master.csv exists):
    python train_readmission_model.py

Reads:  Data/Processed/clinical_master.csv
Writes: Data/Processed/readmission_model.pkl
        Data/Processed/readmission_scaler.pkl
        Data/Processed/readmission_model_features.pkl

See the notebook for the full reasoning behind each modeling decision
(why these features, why class_weight="balanced", why Logistic
Regression vs Random Forest, etc.) - this script keeps that logic,
just without the markdown explanations.
"""

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler

PROCESSED = "../Data/Processed"

# Features chosen in the notebook: anchor_age, diagnosis_count and
# admission_type are known at admission time; prior_admissions_count
# (not lifetime total admissions) is used specifically to avoid data
# leakage, since the lifetime total is not known until after the
# patient's final visit.
FEATURE_COLS = ["anchor_age", "diagnosis_count", "prior_admissions_count", "admission_type"]
TARGET_COL = "readmitted_30d"


def build_model_frame(clinical_master: pd.DataFrame) -> pd.DataFrame:
    model_df = clinical_master[FEATURE_COLS + [TARGET_COL]].copy()
    model_df = pd.get_dummies(model_df, columns=["admission_type"], drop_first=True)
    return model_df


def metrics_row(y_true, y_pred, name):
    return {
        "model": name,
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }


def main():
    print("Loading clinical_master...")
    clinical_master = pd.read_csv(f"{PROCESSED}/clinical_master.csv")

    model_df = build_model_frame(clinical_master)
    X = model_df.drop(columns=[TARGET_COL])
    y = model_df[TARGET_COL]
    feature_cols = list(X.columns)

    print("Feature columns:", feature_cols)
    print("Class balance:\n", y.value_counts(normalize=True).round(3))

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    # Logistic Regression needs scaled features; Random Forest does not.
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print("\nTraining Logistic Regression...")
    logreg = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
    logreg.fit(X_train_scaled, y_train)
    y_pred_lr = logreg.predict(X_test_scaled)
    results = [metrics_row(y_test, y_pred_lr, "Logistic Regression")]
    print("Confusion matrix:\n", confusion_matrix(y_test, y_pred_lr))

    print("\nTraining Random Forest...")
    rf = RandomForestClassifier(
        n_estimators=200, max_depth=5, min_samples_leaf=5,
        class_weight="balanced", random_state=42,
    )
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)
    results.append(metrics_row(y_test, y_pred_rf, "Random Forest"))
    print("Confusion matrix:\n", confusion_matrix(y_test, y_pred_rf))

    results_df = pd.DataFrame(results).set_index("model").round(3)
    print("\nModel comparison (test set):\n", results_df)

    # 5-fold cross-validation as a robustness check, since the test set
    # here is small (~70 rows) and a single split can be noisy.
    cv_lr = cross_val_score(
        LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42),
        scaler.fit_transform(X), y, cv=5, scoring="f1",
    )
    cv_rf = cross_val_score(
        RandomForestClassifier(n_estimators=200, max_depth=5, min_samples_leaf=5,
                                class_weight="balanced", random_state=42),
        X, y, cv=5, scoring="f1",
    )
    print(f"\nCV F1 Logistic Regression: {cv_lr.mean():.3f} +/- {cv_lr.std():.3f}")
    print(f"CV F1 Random Forest:       {cv_rf.mean():.3f} +/- {cv_rf.std():.3f}")

    # Keep whichever model scored higher F1 on the held-out test set -
    # this matches the notebook's own selection logic.
    best_name = results_df["f1"].idxmax()
    best_model = logreg if best_name == "Logistic Regression" else rf
    print(f"\nBest model by F1: {best_name}")

    joblib.dump(best_model, f"{PROCESSED}/readmission_model.pkl")
    joblib.dump(scaler, f"{PROCESSED}/readmission_scaler.pkl")
    joblib.dump(feature_cols, f"{PROCESSED}/readmission_model_features.pkl")
    print(f"\nSaved model, scaler and feature list to {PROCESSED}/")


if __name__ == "__main__":
    main()
