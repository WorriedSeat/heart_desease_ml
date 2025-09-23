import streamlit as st
import requests

# FastAPI endpoint
FASTAPI_URL = "http://fastapi:8000/predict"

# Streamlit app UI
st.title("Heart disease Identificator")

'''
    age: float
    sex: float #(0-female, 1-male)
    cp: float #(1-typical angina, 2-atypical angina, 3-non-anginal pain, 4-asymptomatic)
    trestbps: float
    chol: float
    fbs: float #(1- >=120mg/dl, 0- <120mg/dl)
    restecg: float #(0-normal, 1-wave abnormability, 2-left ventricular hypertrophy)
    thalach: float
    exang: float #(1-yes, 0-no)
    oldpeak: float
    slope: float #(1-upsloping, 2-flat, 3-downsloping)
    ca: float
    thal: float #feature categorical (3-normal, 6-fixed defect, 7-reversable defect)
'''
# Input fields
age = st.number_input("Age", min_value=0.0, max_value=130.0)
sex = st.selectbox("Sex", options=[("Female", 0), ("Male", 1)], format_func=lambda x: x[0])[1]
cp = st.selectbox("Chest pain", options=[("Typical angina", 1), ("Atypical angina", 2), ("Non-anginal pain", 3), ("Asymptomatic", 4)], format_func=lambda x: x[0])[1]
trestbps = st.number_input("Resting blood pressure", min_value=0.0)
chol = st.number_input("Serum cholesterol", min_value=0.0)
fbs = 1 if st.number_input("Fasting blood sugar", min_value=0.0) >= 120 else 0
restecg = st.selectbox("Resting electrocardiographic results", options=[("Normal", 0), ("ST-T wave abnormability", 1), ("Left ventricular hypertrophy", 2)], format_func=lambda x: x[0])[1]
thalach = st.number_input("Maximum heart rate achieved", min_value=0.0)
exang = st.selectbox("Exercise including angina", options=[("No", 0), ("Yes", 1)], format_func=lambda x: x[0])[1]
oldpeak = st.number_input("ST depression induced by exercise relative to rest", min_value=0.0)
slope = st.selectbox("Slope of the peak exercise ST segment", options=[("Upsloping", 1), ("Flat", 2), ("Downsloping", 3)], format_func=lambda x: x[0])[1]
ca = st.number_input("Number of major vessels(0-3) colored by floursopy", min_value=0.0, max_value=3.0)
thal = st.selectbox("Thalassemia", options=[("Normal", 3), ("Fixed defect", 6), ("Reversable defect", 7)], format_func=lambda x: x[0])[1]

# Make predictions when the button is pressed
if st.button("Predict"):
    data = {
        "age": age,
        "sex": sex,
        "cp": cp,
        "trestbps": trestbps,
        "chol": chol,
        "fbs": fbs,
        "restecg": restecg,
        "thalach": thalach,
        "exang": exang,
        "oldpeak": oldpeak,
        "slope": slope,
        "ca": ca,
        "thal": thal
    }
    
    # Send a request and get the response
    response = requests.post(FASTAPI_URL, json=data)
    prediction = response.json()["prediction"]
    
    # Display the result
    risk = "high" if prediction == 1 else "low"
    st.success(f"The model predicts that you have {risk} of having heart disease")
