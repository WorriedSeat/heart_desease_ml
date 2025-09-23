import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import os
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from pathlib import Path
import joblib

class Model_trainer:
    def __init__(self):
        self.ohe_columns = ['cp', 'restecg', 'slope', 'thal']
        self.numeric_columns = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
        
        self.base_path = Path(__file__).resolve().parents[2]
        self.train_path = self.base_path / "data" / "processed" / "cleveland_train.csv" 
        self.test_path = self.base_path / "data" / "processed" / "cleveland_test.csv"
        self.model_path = self.base_path / "models"
        
        
    def encode_ohe(self, data: pd.DataFrame):
        ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')    
        ohe.fit(data[self.ohe_columns])

        encoded_data = ohe.transform(data[self.ohe_columns])
        encoded_df = pd.DataFrame(encoded_data, columns=ohe.get_feature_names_out(self.ohe_columns))

        result_data = data.drop(self.ohe_columns, axis=1)
        result_data = pd.concat([result_data, encoded_df], axis=1)
        
        return result_data

    def scale_numeric(self, data: pd.DataFrame):
        
        scaler = StandardScaler()
        scaler.fit(data[self.numeric_columns])

        scaled_data = scaler.transform(data[self.numeric_columns])
        scaled_df = pd.DataFrame(scaled_data, columns=self.numeric_columns)

        result_data = data.drop(self.numeric_columns, axis=1)
        result_data = pd.concat([result_data, scaled_df], axis=1)
        
        return result_data
    
    def train(self):
        try:
            # Reading the data
            train = pd.read_csv(self.train_path)
            test = pd.read_csv(self.test_path)
            
            # Encoding the categorical features of the data
            ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
            ohe.fit(train[self.ohe_columns])
            
            train_encoded = pd.concat(
                [
                    pd.DataFrame(
                        ohe.transform(train[self.ohe_columns]),
                        columns=ohe.get_feature_names_out(self.ohe_columns)),
                    train.drop(self.ohe_columns, axis=1)
                ],
                axis=1
            )

            test_encoded = pd.concat(
                [
                    pd.DataFrame(
                        ohe.transform(test[self.ohe_columns]),
                        columns=ohe.get_feature_names_out(self.ohe_columns)),
                    test.drop(self.ohe_columns, axis=1)
                ], 
                axis=1
            )

            # Scaling the numeric features of the data
            scaler = StandardScaler()
            scaler.fit(train[self.numeric_columns])
            
            train_prep = pd.concat(
                [
                    pd.DataFrame(
                        scaler.transform(train_encoded[self.numeric_columns]),
                        columns=self.numeric_columns
                    ),
                    train_encoded.drop(self.numeric_columns, axis=1)
                ],
                axis=1
            )
            
            test_prep = pd.concat(
                [
                    pd.DataFrame(
                        scaler.transform(test_encoded[self.numeric_columns]),
                        columns=self.numeric_columns
                    ),
                    test_encoded.drop(self.numeric_columns, axis=1)
                ],
                axis=1
            )
            
            # Splitting the data onto features and target
            x_train = train_prep.drop('target', axis=1)
            y_train = train_prep['target']

            x_test = test_prep.drop('target', axis=1)
            y_test = test_prep['target']
            
            # Creating and Training a model
            model = LogisticRegression(random_state=42)
            model.fit(x_train, y_train)
            
            # Measure the performance on test data
            preds = model.predict(x_test)
            accuracy = accuracy_score(y_test, preds)
            f1 = f1_score(y_test, preds)
            print(f"Accuracy {accuracy:.3f}, F1: {f1:.3f}")
            
            # Saving models
            joblib.dump(model, self.model_path / "logreg_model.pkl")
            joblib.dump(ohe, self.model_path / "ohe.pkl")
            joblib.dump(scaler, self.model_path / "standard_scaler.pkl")

        except Exception as e:
            print(f"ERROR in Model_trainer.train(): {e}")
        
if __name__ == "__main__":
    model_trainer = Model_trainer()
    model_trainer.train()