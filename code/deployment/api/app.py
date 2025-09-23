from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
from pathlib import Path
import pandas as pd

# Load the trained model
try:
    base_path = Path(__file__).resolve().parents[3]
    model = joblib.load(base_path / "models" / "logreg_model.pkl")
    scaler = joblib.load(base_path / "models" / "standard_scaler.pkl")
    encoder = joblib.load(base_path / "models" / "ohe.pkl")
except Exception as e:
    raise(f"Failed to load the models: {e}")

# Define the FastAPI app
app = FastAPI(title="Heart Disease Prediction API")

# Define the input data schema
class HeartDiseaseInput(BaseModel):
    age: float
    sex: float #(0-female, 1-male)
    cp: float #(1-typical angina, 2-atypical angina, 3-nin-anginal pain, 4-asymptomatic)
    trestbps: float
    chol: float
    fbs: float #(1- >=120mg/dl, 0- <120mg/dl)
    restecg: float #(0-normal, 1-wave abnormability)
    thalach: float
    exang: float #(1-yes, 0-no)
    oldpeak: float
    slope: float #(1-upsloping, 2-flat, 3-downsloping)
    ca: float
    thal: float #feature categorical (3-normal, 6-fixed defect, 7-reversable defect)
    
# Define the prediction endpoint
@app.post("/predict")
def predict(input: HeartDiseaseInput):
    ohe_columns = ['cp', 'restecg', 'slope', 'thal']
    numeric_columns = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
    
    data = pd.DataFrame([input.model_dump()])
    
    for item in data.columns:
        if data[item].isnull().any():
            raise HTTPException(status_code=400, detail=f"Missed value for {item}")
        
    # Encoding categorical features
    data_encoded = pd.concat(
        [
            pd.DataFrame(encoder.transform(data[ohe_columns]), columns=encoder.get_feature_names_out(ohe_columns)),
            data.drop(ohe_columns, axis=1)
        ],
        axis=1
    )
    
    # Scaling the numeric features
    data_prep = pd.concat(
        [
            pd.DataFrame(scaler.transform(data_encoded[numeric_columns]), columns=numeric_columns),
            data_encoded.drop(numeric_columns, axis=1)
        ],
        axis=1
    )
    
    prediction = model.predict(data_prep)
    return {"prediction": prediction[0]}