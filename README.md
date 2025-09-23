# Heart Disease Prediction ML Project

This project provides a machine learning pipeline based on log-regression for predicting heart disease using the Cleveland dataset. It includes data processing, model training, and deployment via FastAPI and Streamlit.

This project is Assignment1 for Practical Machine Learning Deep Learning course in Innopolis University

## Usage

- You need to have Docker installed
- Run the following command (in the root directory of the project) to build and run the app in the Docker:

```bash
docker compose -f code/deployment/docker-compose.yml up --build
```

- The API will be available at `http://localhost:8000/predict`.

- The UI will be available at `http://localhost:8501`.
