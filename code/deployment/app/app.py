import streamlit as st
import requests

# FastAPI endpoint
# FASTAPI_URL = "http://0.0.0.0:8000/predict" #localhost
FASTAPI_URL = "http://fastapi:8000/predict" #Docker

# Streamlit app UI
st.title("Heart disease Identificator")
st.write("This app will help you identify the risk of having the heart disease.")
st.write("It's based on Logistic Regression model and Clevelend dataset from https://archive.ics.uci.edu/dataset/45/heart+disease.")


# Input fields
age = float(st.number_input("Age", min_value=0, max_value=130, value=70))
sex = st.selectbox("Sex", options=[("Female", 0), ("Male", 1)], format_func=lambda x: x[0], index=1)[1]
cp = st.selectbox("Chest pain", options=[("Typical angina", 1), ("Atypical angina", 2), ("Non-anginal pain", 3), ("Asymptomatic", 4)], format_func=lambda x: x[0], index=3)[1]
trestbps = float(st.number_input("Resting blood pressure", min_value=0, value=130))
chol = float(st.number_input("Serum cholesterol", min_value=0, value=322))
fbs = 1 if st.number_input("Fasting blood sugar", min_value=0, value=90) >= 120 else 0
restecg = st.selectbox("Resting electrocardiographic results", options=[("Normal", 0), ("ST-T wave abnormability", 1), ("Left ventricular hypertrophy", 2)], format_func=lambda x: x[0], index=2)[1]
thalach = float(st.number_input("Maximum heart rate achieved", min_value=0, value=109))
exang = st.selectbox("Exercise including angina", options=[("No", 0), ("Yes", 1)], format_func=lambda x: x[0], index=0)[1]
oldpeak = st.number_input("ST depression induced by exercise relative to rest", min_value=0.0, value=2.4)
slope = st.selectbox("Slope of the peak exercise ST segment", options=[("Upsloping", 1), ("Flat", 2), ("Downsloping", 3)], format_func=lambda x: x[0], index=1)[1]
ca = float(st.number_input("Number of major vessels(0-3) colored by floursopy", min_value=0, max_value=3, value=3))
thal = st.selectbox("Thalassemia", options=[("Normal", 3), ("Fixed defect", 6), ("Reversable defect", 7)], format_func=lambda x: x[0], index=0)[1]

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
    risk = "High" if prediction == 1 else "Low"
    st.success(f"Risk having heart disease: {risk}")
