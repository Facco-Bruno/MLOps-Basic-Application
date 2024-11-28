from prefect import flow, task
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from evidently.report import Report
from evidently.metrics import DataDriftTable
import mlflow
import mlflow.sklearn
import joblib
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

# Função para salvar logs localmente
def log_message(step, message):
    log_entry = f"[{datetime.now()}] {step}: {message}"
    print(log_entry)
    with open("mlops_pipeline_logs.txt", "a") as log_file:
        log_file.write(log_entry + "\n")

# 1. Carregar e validar os dados
@task
def load_and_validate_data():
    log_message("load_and_validate_data", "Iniciando carregamento e validação dos dados.")
    data = fetch_california_housing(as_frame=True)
    df = data.frame

    reference_data = df.sample(frac=0.8, random_state=42)
    current_data = df.sample(frac=0.2, random_state=42)

    # Gerar relatório de Data Drift
    report = Report(metrics=[DataDriftTable()])
    report.run(reference_data=reference_data, current_data=current_data)

    drift_detected = report.as_dict()["metrics"][0]["result"]["dataset_drift"]
    report_path = f"data_drift_report_{datetime.now().strftime('%Y%m%d%H%M%S')}.html"
    report.save_html(report_path)

    log_message("load_and_validate_data", f"Relatório salvo em {report_path}. Drift detectado: {drift_detected}")
    return df, reference_data, drift_detected

# 2. Pré-processar os dados
@task
def preprocess_data(df):
    log_message("preprocess_data", "Iniciando pré-processamento dos dados.")
    X = df.drop(columns=["MedHouseVal"])
    y = df["MedHouseVal"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    log_message("preprocess_data", "Dados divididos em treino e teste.")
    return X_train, X_test, y_train, y_test

# 3. Refinar e registrar o modelo
@task
def refine_and_register_model(X_train, y_train, X_test, y_test):
    log_message("refine_and_register_model", "Iniciando treinamento e registro do modelo.")
    mlflow.set_experiment("California Housing Refinement")
    with mlflow.start_run():
        model = RandomForestRegressor(n_estimators=150, max_depth=15, random_state=42)
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        rmse = mean_squared_error(y_test, predictions, squared=False)
        r2 = r2_score(y_test, predictions)

        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2_score", r2)
        mlflow.log_param("n_estimators", 150)
        mlflow.log_param("max_depth", 15)

        joblib.dump(model, "model.pkl")
        mlflow.sklearn.log_model(model, "model")

        log_message("refine_and_register_model", f"Modelo treinado. RMSE: {rmse:.2f}, R²: {r2:.2f}")

        result = mlflow.register_model(f"runs:/{mlflow.active_run().info.run_id}/model", "CaliforniaHousingModel")
        log_message("refine_and_register_model", f"Modelo registrado como versão {result.version}.")
        return model, result.version

# 4. Monitoramento de Drift em Produção
@task
def monitor_drift(reference_data, production_data):
    log_message("monitor_drift", "Monitorando drift nos dados de produção.")
    report = Report(metrics=[DataDriftTable()])
    report.run(reference_data=reference_data, current_data=production_data)
    drift_detected = report.as_dict()["metrics"][0]["result"]["dataset_drift"]
    log_message("monitor_drift", f"Drift detectado: {drift_detected}")
    return drift_detected

# 5. Incorporar feedback do usuário
@task
def incorporate_feedback():
    log_message("incorporate_feedback", "Verificando feedback para re-treinamento.")
    try:
        feedback_data = pd.read_csv("feedback.csv", names=["MedInc", "HouseAge", "AveRooms", "AveBedrms", "Population", "AveOccup", "Latitude", "Longitude", "actual_value"])
        log_message("incorporate_feedback", "Feedback encontrado e carregado.")
        return feedback_data
    except FileNotFoundError:
        log_message("incorporate_feedback", "Nenhum feedback encontrado.")
        return None

# Pipeline principal
@flow(name="mlops-pipeline")
def mlops_pipeline():
    log_message("mlops_pipeline", "Iniciando execução do pipeline.")
    df, reference_data, drift_detected = load_and_validate_data()
    
    if drift_detected:
        X_train, X_test, y_train, y_test = preprocess_data(df)
        model, new_version = refine_and_register_model(X_train, y_train, X_test, y_test)
        log_message("mlops_pipeline", f"Novo modelo treinado e registrado como versão {new_version}.")
    else:
        log_message("mlops_pipeline", "Nenhum drift detectado. Re-treinamento não necessário.")

    feedback_data = incorporate_feedback()
    if feedback_data is not None:
        log_message("mlops_pipeline", "Feedback processado para re-treinamento futuro.")
    else:
        log_message("mlops_pipeline", "Nenhum feedback disponível no momento.")

    log_message("mlops_pipeline", "Execução do pipeline concluída.")

# Executar o pipeline
if __name__ == "__main__":
    mlops_pipeline()
