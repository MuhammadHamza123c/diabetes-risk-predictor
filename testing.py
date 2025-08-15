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

test_entries = [
    {"age": 45, "Blood_pressure": 0, "heart_disease": 0, "bmi": 20.3, "HbA1c_level": 5.0, "blood_glucose_level": 134, "gender": "Female", "smoking_history": "never"},
    {"age": 60, "Blood_pressure": 1, "heart_disease": 1, "bmi": 28.0, "HbA1c_level": 7.0, "blood_glucose_level": 160, "gender": "Male", "smoking_history": "former"},
    {"age": 30, "Blood_pressure": 0, "heart_disease": 0, "bmi": 22.0, "HbA1c_level": 5.5, "blood_glucose_level": 110, "gender": "Female", "smoking_history": "never"},
    {"age": 50, "Blood_pressure": 1, "heart_disease": 0, "bmi": 30.5, "HbA1c_level": 8.0, "blood_glucose_level": 180, "gender": "Male", "smoking_history": "current"},
    {"age": 35, "Blood_pressure": 0, "heart_disease": 0, "bmi": 24.0, "HbA1c_level": 5.2, "blood_glucose_level": 120, "gender": "Female", "smoking_history": "never"},
    {"age": 55, "Blood_pressure": 1, "heart_disease": 1, "bmi": 29.5, "HbA1c_level": 7.2, "blood_glucose_level": 165, "gender": "Male", "smoking_history": "former"},
    {"age": 40, "Blood_pressure": 0, "heart_disease": 0, "bmi": 23.5, "HbA1c_level": 5.4, "blood_glucose_level": 125, "gender": "Female", "smoking_history": "ever"},
    {"age": 65, "Blood_pressure": 1, "heart_disease": 1, "bmi": 31.0, "HbA1c_level": 8.5, "blood_glucose_level": 190, "gender": "Male", "smoking_history": "current"},
    {"age": 28, "Blood_pressure": 0, "heart_disease": 0, "bmi": 21.5, "HbA1c_level": 5.1, "blood_glucose_level": 115, "gender": "Female", "smoking_history": "never"},
    {"age": 52, "Blood_pressure": 1, "heart_disease": 0, "bmi": 27.5, "HbA1c_level": 6.8, "blood_glucose_level": 155, "gender": "Male", "smoking_history": "former"},
    {"age": 38, "Blood_pressure": 0, "heart_disease": 0, "bmi": 23.0, "HbA1c_level": 5.6, "blood_glucose_level": 122, "gender": "Female", "smoking_history": "never"},
    {"age": 58, "Blood_pressure": 1, "heart_disease": 1, "bmi": 30.0, "HbA1c_level": 7.5, "blood_glucose_level": 170, "gender": "Male", "smoking_history": "current"},
    {"age": 33, "Blood_pressure": 0, "heart_disease": 0, "bmi": 22.5, "HbA1c_level": 5.3, "blood_glucose_level": 118, "gender": "Female", "smoking_history": "ever"},
    {"age": 61, "Blood_pressure": 1, "heart_disease": 1, "bmi": 29.0, "HbA1c_level": 7.0, "blood_glucose_level": 162, "gender": "Male", "smoking_history": "former"},
    {"age": 36, "Blood_pressure": 0, "heart_disease": 0, "bmi": 24.5, "HbA1c_level": 5.7, "blood_glucose_level": 126, "gender": "Female", "smoking_history": "never"},
    {"age": 57, "Blood_pressure": 1, "heart_disease": 1, "bmi": 28.5, "HbA1c_level": 7.1, "blood_glucose_level": 168, "gender": "Male", "smoking_history": "current"},
    {"age": 42, "Blood_pressure": 0, "heart_disease": 0, "bmi": 23.8, "HbA1c_level": 5.4, "blood_glucose_level": 124, "gender": "Female", "smoking_history": "never"},
    {"age": 63, "Blood_pressure": 1, "heart_disease": 1, "bmi": 31.5, "HbA1c_level": 8.2, "blood_glucose_level": 185, "gender": "Male", "smoking_history": "former"},
    {"age": 39, "Blood_pressure": 0, "heart_disease": 0, "bmi": 22.8, "HbA1c_level": 5.5, "blood_glucose_level": 121, "gender": "Female", "smoking_history": "ever"},
    {"age": 54, "Blood_pressure": 1, "heart_disease": 1, "bmi": 29.2, "HbA1c_level": 7.3, "blood_glucose_level": 166, "gender": "Male", "smoking_history": "current"},
]

df_test = pd.DataFrame(test_entries)

df_test_encoded = pd.get_dummies(df_test, columns=["gender", "smoking_history"])

for col in model_training_columns:
    if col not in df_test_encoded.columns:
        df_test_encoded[col] = 0

df_test_encoded = df_test_encoded[model_training_columns]

test_data_scaled = scaler.transform(df_test_encoded)
df_test['predicted_diabetes'] = boost_class.predict(test_data_scaled)
df_test['predicted_diabetes'] = df_test['predicted_diabetes'].map({0: 'No', 1: 'Yes'})


print(df_test[['age','Blood_pressure','bmi','HbA1c_level','blood_glucose_level','gender','smoking_history','predicted_diabetes']])

df_test.to_csv('predicted_20_patients.csv', index=False)
