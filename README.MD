# Data Before Doors – University Enrollment Prediction

A beginner-friendly machine learning project that predicts a student's **university outcome** as **Dropout, Enrolled, or Graduate** using real student data from the UCI Machine Learning Repository.

## 📌 Project Overview

**Data Before Doors** is an end-to-end machine learning project built to understand the complete workflow of a classification problem — from exploring raw data to training models and deploying the final model using Flask.

The project uses information such as:

- Age at enrollment
- Admission grade
- Previous qualification grade
- Scholarship status
- Tuition fee status
- Debtor status
- Gender
- International student status

The project is intentionally kept simple so that every part of the machine learning workflow can be understood and explained clearly.

---

## 🎯 Problem Statement

Universities may want to identify students who are more likely to have different academic outcomes.

The objective of this project is to build a machine learning model that predicts one of three outcomes:

- Dropout
- Enrolled
- Graduate

This is a **multiclass classification problem** because the target variable contains three discrete classes.

---

## 📊 Dataset

The project uses the **Predict Students' Dropout and Academic Success** dataset from the **UCI Machine Learning Repository**.

- **Source:** UCI Machine Learning Repository
- **Dataset:** Predict Students' Dropout and Academic Success
- **Dataset ID:** 697
- **Records:** 4,424 students
- **Features:** 36 predictor variables in the original dataset

### Target Variable

The target column is:

`Target`

| Target | Meaning |
|---|---|
| Dropout | Student eventually dropped out |
| Enrolled | Student remained enrolled |
| Graduate | Student eventually graduated |

### Dataset Source

https://archive.ics.uci.edu/dataset/697/predict+students+dropout+and+academic+success

---

## ⚠️ Data Leakage Consideration

The original dataset contains information from the first and second semesters, including academic-performance variables.

These variables were **not used in the main prediction model** because they become available after the student has already started university.

Using them would make an early prediction system unrealistic and could introduce **data leakage**.

Therefore, this project focuses on information available around the time of enrollment.

---

## 🧠 Features Used

The simplified model uses the following features:

| Feature | Description |
|---|---|
| `Age at enrollment` | Student's age when enrolling |
| `Admission grade` | Admission score |
| `Previous qualification (grade)` | Grade from previous qualification |
| `Scholarship holder` | Whether the student has a scholarship |
| `Tuition fees up to date` | Whether tuition fees are up to date |
| `Debtor` | Whether the student has outstanding debt |
| `Gender` | Gender category |
| `International` | Whether the student is an international student |

---

## 🔄 Machine Learning Workflow

```text
                    UCI Dataset
                         │
                         ▼
                 Data Understanding
                         │
                         ▼
                   Data Cleaning
                         │
                 ┌───────┴───────┐
                 ▼               ▼
          Missing Values      Duplicates
                 │               │
                 └───────┬───────┘
                         ▼
                        EDA
                         │
                         ▼
                  Feature Selection
                         │
                         ▼
                  Train/Test Split
                         │
                         ▼
                 One-Hot Encoding
                         │
                         ▼
                   Feature Scaling
                         │
              ┌──────────┼──────────┐
              ▼          ▼          ▼
          Logistic    Decision    Random
         Regression     Tree       Forest
              │          │          │
              └──────────┼──────────┘
                         ▼
                   Model Evaluation
                         │
                         ▼
                   Model Comparison
                         │
                         ▼
                    Final Model
                         │
                         ▼
                    Flask App