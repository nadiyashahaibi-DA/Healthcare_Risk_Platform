CREATE DATABASE IF NOT EXISTS healthcare_risk_platform;
USE healthcare_risk_platform;

CREATE TABLE patients (
    subject_id   INT PRIMARY KEY,
    gender       CHAR(1),
    anchor_age   INT,
    deceased     TINYINT(1)
);

CREATE TABLE admissions (
    hadm_id              INT PRIMARY KEY,
    subject_id           INT NOT NULL,
    admittime             DATETIME,
    dischtime              DATETIME,
    admission_type         VARCHAR(50),
    insurance               VARCHAR(50),
    race                     VARCHAR(100),
    length_of_stay_days    DECIMAL(6,2),
    readmitted_30d         TINYINT(1),
    FOREIGN KEY (subject_id) REFERENCES patients(subject_id)
);

CREATE TABLE diagnoses (
    diagnosis_id          INT AUTO_INCREMENT PRIMARY KEY,
    hadm_id                INT NOT NULL,
    seq_num                 INT,
    icd_code                 VARCHAR(10),
    icd_version               INT,
    is_primary_diagnosis     TINYINT(1),
    diagnosis_label           VARCHAR(100),
    FOREIGN KEY (hadm_id) REFERENCES admissions(hadm_id)
);