import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path
import os

'''The dataset used for the heart-disease identification is cleveland dataset
collected from: https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data  
all the information about the columns etc. were extracted from: https://archive.ics.uci.edu/dataset/45/heart+disease

Columns:
    age - feature numeric
    sex - feature categorical (0-female, 1-male)
    cp - feature categorical (1-typical angina, 2-atypical angina, 3-non-anginal pain, 4-asymptomatic)
    trestbps - feature numeric
    chol - feature numeric
    fbs - feature categorical (1- >=120mg/dl, 0- <120mg/dl)
    restecg - feature categorical (0-normal, 1-wave abnormability, 2-left ventricular hypertrophy)
    thalach - feature numeric
    exang - feature categorical (1-yes, 0-no)
    oldpeak - feature numeric
    slope - feature categorical (1-upsloping, 2-flat, 3-downsloping)
    ca - feature numeric
    thal - feature categorical (3-normal, 6-fixed defect, 7-reversable defect)
    target - target numeric
'''

class Data_Processor:
    def __init__(self):
        self.url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
        self.columns = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target']
        
        self.base_path = Path(__file__).resolve().parents[2]
        self.raw_path = self.base_path / "data" / "raw"
        self.processed_path = self.base_path / "data" / "processed"
        self.filename = "cleveland.csv"
    
    def process_data(self):
        
        # Check the presence of the raw dataset in the directory
        # if no download it
        os.makedirs(self.raw_path, exist_ok=True)
        os.makedirs(self.processed_path, exist_ok=True)
        if not os.path.exists(self.raw_path / self.filename):
            url_data = pd.read_csv(self.url, names=self.columns, na_values='?')
            url_data.to_csv(self.raw_path / self.filename, index=False) 
        
        data = pd.read_csv(self.raw_path / self.filename)
        
        # Dropping nan values since only 6 nans
        data.dropna(ignore_index=True, inplace=True)
        
        # Binarization of target value
        data['target'] = data['target'].apply(lambda x: 1 if x > 0 else 0)
        
        train_data, test_data = train_test_split(data, train_size=0.8, random_state=42)

        # Saving train and test data
        train_file = os.path.join(self.processed_path, "cleveland_train.csv")
        test_file = os.path.join(self.processed_path, "cleveland_test.csv")
        train_data.to_csv(train_file, index=False)
        test_data.to_csv(test_file, index=False)
        
if __name__ == "__main__":
    data_processor = Data_Processor()
    data_processor.process_data() 
