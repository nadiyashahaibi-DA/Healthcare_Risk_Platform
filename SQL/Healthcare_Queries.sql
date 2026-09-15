-- Query 1: Top patients by admission count
SELECT subject_id, COUNT(*) AS admission_count
FROM admissions
GROUP BY subject_id
ORDER BY admission_count DESC;

-- Query 2: Average patient age
SELECT AVG(anchor_age) AS avg_age
FROM patients;

-- Query 3: Most common diagnoses
SELECT icd_code, COUNT(*) AS total
FROM diagnoses
GROUP BY icd_code
ORDER BY total DESC
LIMIT 10;

-- Query 4: High-risk patients (more than 2 admissions)
SELECT subject_id, COUNT(*) AS admissions
FROM admissions
GROUP BY subject_id
HAVING COUNT(*) > 2;
-- ============================================================
-- Additional queries: JOIN, CTE, window functions, CASE
-- ============================================================

-- Query 5: JOIN across all three tables — primary diagnosis per admission
SELECT
    p.subject_id,
    p.anchor_age,
    a.admission_type,
    d.diagnosis_label
FROM patients p
JOIN admissions a ON p.subject_id = a.subject_id
JOIN diagnoses d ON a.hadm_id = d.hadm_id
WHERE d.is_primary_diagnosis = 1
ORDER BY p.subject_id;

-- Query 6: JOIN + GROUP BY — average length of stay by primary diagnosis
SELECT
    d.diagnosis_label,
    COUNT(*) AS admission_count,
    ROUND(AVG(a.length_of_stay_days), 1) AS avg_length_of_stay
FROM admissions a
JOIN diagnoses d ON a.hadm_id = d.hadm_id
WHERE d.is_primary_diagnosis = 1
GROUP BY d.diagnosis_label
HAVING COUNT(*) >= 2
ORDER BY avg_length_of_stay DESC;

-- Query 7: CASE statement — readmission rate by age bracket
SELECT
    CASE
        WHEN p.anchor_age < 30 THEN '<30'
        WHEN p.anchor_age BETWEEN 30 AND 44 THEN '30-44'
        WHEN p.anchor_age BETWEEN 45 AND 59 THEN '45-59'
        WHEN p.anchor_age BETWEEN 60 AND 74 THEN '60-74'
        ELSE '75+'
    END AS age_bracket,
    COUNT(*) AS total_admissions,
    SUM(a.readmitted_30d) AS readmissions,
    ROUND(SUM(a.readmitted_30d) / COUNT(*), 2) AS readmission_rate
FROM admissions a
JOIN patients p ON a.subject_id = p.subject_id
GROUP BY age_bracket
ORDER BY readmission_rate DESC;

-- Query 8: Window function — prior admissions count per patient
-- (mirrors the prior_admissions_count feature engineered in notebook 03/05)
SELECT
    subject_id,
    hadm_id,
    admittime,
    ROW_NUMBER() OVER (PARTITION BY subject_id ORDER BY admittime) - 1 AS prior_admissions_count
FROM admissions
ORDER BY subject_id, admittime;

-- Query 9: CTE + window function — rank patients by total admissions
WITH admission_counts AS (
    SELECT subject_id, COUNT(*) AS total_admissions
    FROM admissions
    GROUP BY subject_id
)
SELECT
    subject_id,
    total_admissions,
    RANK() OVER (ORDER BY total_admissions DESC) AS admission_rank
FROM admission_counts
ORDER BY admission_rank
LIMIT 10;

-- Query 10: CTE + JOIN + CASE — patient risk summary (capstone query)
-- Same high-risk rule used in notebook 04: 3+ admissions OR a 30-day readmission
WITH patient_stats AS (
    SELECT
        a.subject_id,
        COUNT(*) AS total_admissions,
        MAX(a.readmitted_30d) AS ever_readmitted
    FROM admissions a
    GROUP BY a.subject_id
)
SELECT
    p.subject_id,
    p.anchor_age,
    ps.total_admissions,
    ps.ever_readmitted,
    CASE
        WHEN ps.total_admissions >= 3 OR ps.ever_readmitted = 1 THEN 'High risk'
        ELSE 'Lower risk'
    END AS risk_flag
FROM patients p
JOIN patient_stats ps ON p.subject_id = ps.subject_id
ORDER BY ps.total_admissions DESC;
