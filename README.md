# Diabetes Prediction ML Model

![Diabetes Prediction](https://img.shields.io/badge/ML-Python-blue) ![Status](https://img.shields.io/badge/status-Completed-green)

---

## Overview

This project implements a **Diabetes Prediction Machine Learning Model** that predicts whether a patient is likely to have diabetes based on multiple health parameters. It demonstrates the complete ML workflow: data preprocessing, model training, prediction.

The repository also includes a **Diabetes report** comparing predicted vs actual outcomes for patients, showing model performance and accuracy.

![View Sample PDF Report](https://github.com/MuhammadHamza123c/diabetes-risk-predictor/blob/main/diabetes_predictions_report_page-0001.jpg)

---

## Features Used for Prediction

The model predicts diabetes using the following features:

- **Age**  
- **Blood Pressure** (0 = Normal, 1 = High)  
- **Heart Disease** (0 = No, 1 = Yes)  
- **BMI** (Body Mass Index)  
- **HbA1c Level**  
- **Blood Glucose Level**  
- **Gender** (Female / Male / Other)  
- **Smoking History** (No Info / current / ever / former / never / not current)  

---

## Model Evaluation

The model was evaluated on a test dataset with the following metrics:

**Test Dataset:**

- Accuracy: 97.07%  
- Precision: 0.97 (class 0), 0.90 (class 1)  
- Recall: 1.00 (class 0), 0.52 (class 1)  
- F1-Score: 0.98 (class 0), 0.66 (class 1)  

**Train Dataset 2:**

- Accuracy: 97.69%  
- Precision: 0.98 (class 0), 0.98 (class 1)  
- Recall: 1.00 (class 0), 0.57 (class 1)  
- F1-Score: 0.99 (class 0), 0.72 (class 1)  



---

## Model Selection

During development, multiple machine learning algorithms were tested to predict diabetes, including:

- Logistic Regression  
- Random Forest  
- K-Nearest Neighbors (KNN)  

After evaluating performance metrics on the validation dataset, **XGBoost (Extreme Gradient Boosting)** was selected as the final model because it achieved the **best overall accuracy, precision, and recall**.

## Purpose / Use

- Evaluate diabetes risk for patients based on health metrics.  
- Demonstrate the workflow of building, training, and deploying an ML model in Python.  
- Generate professional reports for presentations, analysis, or documentation.


## Repository Contents

- `model.joblib` – Trained Gradient Boosting Model  
- `scaler.joblib` – Scaler for normalizing input features  
- `main.py` – Python script to take user input and predict diabetes  
- `diabetes_predictions_report.pdf` – Sample PDF report of predictions vs actual outcomes  
- `README.md` – This file  

---

## Installation & Usage

```bash
# Clone the repository
git clone https://github.com/MuhammadHamza123c/diabetes-risk-predictor.git
cd diabetes-risk-predictor

# Install dependencies
pip install pandas joblib reportlab

# Run the prediction script
python main.py

# Follow the prompts to enter patient details
# The script outputs whether the patient is predicted to have diabetes
# A PDF report (diabetes_predictions_report.pdf) will be generated automatically
