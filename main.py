import joblib
import pandas as pd

scaler = joblib.load('diabetes_model_scaler.joblib')
boost_class = joblib.load('diabetes_model.joblib')

model_training_columns = [
    'age', 'Blood_pressure', 'heart_disease', 'bmi', 'HbA1c_level',
    'blood_glucose_level', 'gender_Female', 'gender_Male', 'gender_Other',
    'smoking_history_No Info', 'smoking_history_current',
    'smoking_history_ever', 'smoking_history_former',
    'smoking_history_never', 'smoking_history_not current'
]

age = int(input("Enter age: "))
blood_pressure = int(input("Blood pressure (0 = normal, 1 = high): "))
heart_disease = int(input("Do you have heart disease? (0 = No, 1 = Yes): "))
bmi = float(input("Enter BMI: "))
hba1c_level = float(input("Enter HbA1c level: "))
blood_glucose_level = float(input("Enter blood glucose level: "))
gender = input("Enter gender (Female/Male/Other): ")
smoking_history = input("Enter smoking history (No Info/current/ever/former/never/not current): ")

raw_test_data = {
    "age": age,
    "Blood_pressure": blood_pressure,
    "heart_disease": heart_disease,
    "bmi": bmi,
    "HbA1c_level": hba1c_level,
    "blood_glucose_level": blood_glucose_level,
    "gender": gender,
    "smoking_history": smoking_history
}

df_test = pd.DataFrame([raw_test_data])
df_test = pd.get_dummies(df_test, columns=["gender", "smoking_history"])

for col in model_training_columns:
    if col not in df_test.columns:
        df_test[col] = 0

df_test = df_test[model_training_columns]
test_data_scaled = scaler.transform(df_test)
prediction = boost_class.predict(test_data_scaled)
print("Diabetes Prediction:", "Yes" if prediction[0] == 1 else "No")
