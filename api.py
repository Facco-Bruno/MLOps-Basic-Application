from fastapi import FastAPI
import mlflow.pyfunc
import pandas as pd
from pydantic import BaseModel
from typing import Dict, Union
import csv
from datetime import datetime

# Modelo de entrada para predição
class PredictionRequest(BaseModel):
    MedInc: float
    HouseAge: float
    AveRooms: float
    AveBedrms: float
    Population: float
    AveOccup: float
    Latitude: float
    Longitude: float

# Modelo de entrada para feedback
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

# Modelo de resposta para predição
class PredictionResponse(BaseModel):
    input_data: Dict[str, Union[float, int]]
    prediction: float

# Variável para a versão do modelo gerenciada pelo pipeline
model_version = 5  # Atualizado automaticamente

# Carregar o modelo do MLflow Model Registry
try:
    model = mlflow.pyfunc.load_model(f"models:/CaliforniaHousingModel/{model_version}")
    model_loaded = True
except Exception as e:
    model_loaded = False
    model_error = str(e)

# Inicializar FastAPI
app = FastAPI(
    title="API de Predição de Preços de Casas",
    description="API para realizar predições e registrar feedback",
    version="1.0.0",
)

# Verificar o status da API
@app.get("/")
def health_check():
    if model_loaded:
        return {"status": "ok", "message": "API está funcionando corretamente."}
    else:
        return {"status": "error", "message": f"Erro ao carregar o modelo: {model_error}"}

# Endpoint para predições
@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    if not model_loaded:
        return {"status": "error", "message": "Modelo não carregado. Verifique o registro no MLflow."}

    # Dados de entrada
    input_data = pd.DataFrame([request.dict()])

    # Predição
    prediction = model.predict(input_data)[0]  # Obtém o primeiro valor da lista

    # Print no terminal
    print(f"[{datetime.now()}] Predição realizada: {prediction:.2f} para entrada: {request.dict()}")

    # Salvar na tabela (arquivo CSV)
    with open("predictions.csv", "a", newline="") as f:
        writer = csv.writer(f)
        # Adiciona os dados de entrada e o valor previsto
        if f.tell() == 0:  # Verifica se o arquivo está vazio
            writer.writerow(list(request.dict().keys()) + ["Prediction"])
        writer.writerow(list(request.dict().values()) + [prediction])

    # Retornar resposta formatada
    return PredictionResponse(input_data=request.dict(), prediction=prediction)

# Endpoint para registrar feedback
@app.post("/feedback")
def feedback(request: FeedbackRequest):
    with open("feedback.csv", "a", newline="") as f:
        writer = csv.writer(f)
        # Adiciona os dados de entrada e o valor real
        if f.tell() == 0:  # Verifica se o arquivo está vazio
            writer.writerow(list(request.dict().keys()))
        writer.writerow(list(request.dict().values()))
    return {"status": "success", "message": "Feedback registrado com sucesso."}
