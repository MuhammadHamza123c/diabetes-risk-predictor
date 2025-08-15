# Diabetes Prediction ML Model

![Diabetes Prediction](https://img.shields.io/badge/ML-Python-blue) ![Status](https://img.shields.io/badge/status-Completed-green)

---

## Overview

This project implements a **Diabetes Prediction Machine Learning Model** that predicts whether a patient is likely to have diabetes based on multiple health parameters. It demonstrates the complete ML workflow: data preprocessing, model training, prediction, and report generation.

The repository also includes a **PDF report** comparing predicted vs actual outcomes for patients, showing model performance and accuracy.

[View Sample PDF Report](https://github.com/MuhammadHamza123c/diabetes-risk-predictor/blob/main/diabetes_predictions_report.pdf)

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

