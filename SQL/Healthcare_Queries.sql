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