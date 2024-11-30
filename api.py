from fastapi import FastAPI
import mlflow.pyfunc
import pandas as pd
from pydantic import BaseModel
from typing import Dict, Union
import csv
from datetime import datetime

######################
# Data model for prediction requests
# Defines the expected structure of incoming JSON for predictions.
# Each field represents a feature required by the model.
######################
class PredictionRequest(BaseModel):
    MedInc: float
    HouseAge: float
    AveRooms: float
    AveBedrms: float
    Population: float
    AveOccup: float
    Latitude: float
    Longitude: float

######################
# Data model for feedback requests
# Defines the expected structure of incoming JSON for feedback submissions.
# Includes both the feature values and the actual observed value (target variable).
######################
class FeedbackRequest(BaseModel):
    MedInc: float
    HouseAge: float
    AveRooms: float
    AveBedrms: float
    Population: float
    AveOccup: float
    Latitude: float
    Longitude: float
    actual_value: float

######################
# Data model for prediction responses
# Defines the structure of the response sent back to the user after a prediction.
# Includes the input data and the predicted value.
######################
class PredictionResponse(BaseModel):
    input_data: Dict[str, Union[float, int]]
    prediction: float

######################
# Variable for managing the model version
# The model version corresponds to the registered version in the MLflow Model Registry.
# This should align with the latest production-ready model.
######################
model_version = 5  # Automatically updated by the pipeline

######################
# Load the model from the MLflow Model Registry
# Attempts to load the specified model version. If successful, `model_loaded` is set to True.
# Otherwise, an error message is stored in `model_error`.
######################
try:
    model = mlflow.pyfunc.load_model(f"models:/CaliforniaHousingModel/{model_version}")
    model_loaded = True
except Exception as e:
    model_loaded = False
    model_error = str(e)

######################
# Initialize FastAPI application
# Sets up the API with metadata such as title, description, and version.
######################
app = FastAPI(
    title="API de Predição de Preços de Casas",
    description="API para realizar predições e registrar feedback",
    version="1.0.0",
)

######################
# Endpoint: Health Check
# Purpose:
#   - Checks the status of the API and verifies if the model is loaded successfully.
# Response:
#   - Returns "ok" if the model is loaded.
#   - Returns an error message if the model failed to load.
######################
@app.get("/")
def health_check():
    if model_loaded:
        return {"status": "ok", "message": "API está funcionando corretamente."}
    else:
        return {"status": "error", "message": f"Erro ao carregar o modelo: {model_error}"}

######################
# Endpoint: Predict
# Purpose:
#   - Accepts a JSON payload matching the `PredictionRequest` structure.
#   - Uses the loaded ML model to make predictions.
#   - Saves predictions and input data to a CSV file for logging.
# Response:
#   - Returns the prediction and input data in a structured format.
# Logs:
#   - Saves prediction requests and results to `predictions.csv`.
######################
@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    if not model_loaded:
        return {"status": "error", "message": "Modelo não carregado. Verifique o registro no MLflow."}

    # Prepare input data for the model
    input_data = pd.DataFrame([request.dict()])

    # Make a prediction using the loaded model
    prediction = model.predict(input_data)[0]  # Extracts the first prediction result

    # Log prediction details to the terminal
    print(f"[{datetime.now()}] Predição realizada: {prediction:.2f} para entrada: {request.dict()}")

    # Append input data and prediction to a CSV file for logging
    with open("predictions.csv", "a", newline="") as f:
        writer = csv.writer(f)
        # Write header row if the file is empty
        if f.tell() == 0:  # Check if the file is empty
            writer.writerow(list(request.dict().keys()) + ["Prediction"])
        # Write input data and prediction
        writer.writerow(list(request.dict().values()) + [prediction])

    # Return the prediction result in the API response
    return PredictionResponse(input_data=request.dict(), prediction=prediction)

######################
# Endpoint: Feedback
# Purpose:
#   - Accepts a JSON payload matching the `FeedbackRequest` structure.
#   - Logs user-provided feedback to a CSV file for future retraining or analysis.
# Response:
#   - Confirms successful recording of feedback.
# Logs:
#   - Saves feedback requests to `feedback.csv`.
######################
@app.post("/feedback")
def feedback(request: FeedbackRequest):
    # Append feedback data to a CSV file
    with open("feedback.csv", "a", newline="") as f:
        writer = csv.writer(f)
        # Write header row if the file is empty
        if f.tell() == 0:  # Check if the file is empty
            writer.writerow(list(request.dict().keys()))
        # Write feedback data
        writer.writerow(list(request.dict().values()))
    # Return a success message to the client
    return {"status": "success", "message": "Feedback registrado com sucesso."}
