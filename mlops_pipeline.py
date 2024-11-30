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

######################
# Function: log_message
# Purpose:
#   - Saves logs locally for tracking pipeline activities.
#   - Logs messages both to the console for real-time monitoring and a local file for auditing.
# Interaction:
#   - Called by all other functions to log their execution steps and results.
######################
def log_message(step, message):
    log_entry = f"[{datetime.now()}] {step}: {message}"
    print(log_entry)
    with open("mlops_pipeline_logs.txt", "a") as log_file:
        log_file.write(log_entry + "\n")

######################
# Task 1: load_and_validate_data
# Purpose:
#   - Loads the California Housing dataset.
#   - Splits the data into reference (80%) and current (20%) datasets for drift detection.
#   - Generates a data drift report using `evidently` library.
#   - Determines whether dataset drift is present.
# Interaction:
#   - Returns the full dataset (df), reference data, and a drift detection flag.
#   - If drift is detected, subsequent functions retrain the model.
#   - Logs execution steps and the path of the generated data drift report.
######################
@task
def load_and_validate_data():
    log_message("load_and_validate_data", "Iniciando carregamento e validação dos dados.")
    data = fetch_california_housing(as_frame=True)
    df = data.frame

    # Splitting data into reference and current datasets
    reference_data = df.sample(frac=0.8, random_state=42)
    current_data = df.sample(frac=0.2, random_state=42)

    # Generating a data drift report
    report = Report(metrics=[DataDriftTable()])
    report.run(reference_data=reference_data, current_data=current_data)

    # Checking for dataset drift
    drift_detected = report.as_dict()["metrics"][0]["result"]["dataset_drift"]
    report_path = f"data_drift_report_{datetime.now().strftime('%Y%m%d%H%M%S')}.html"
    report.save_html(report_path)

    log_message("load_and_validate_data", f"Relatório salvo em {report_path}. Drift detectado: {drift_detected}")
    return df, reference_data, drift_detected

######################
# Task 2: preprocess_data
# Purpose:
#   - Preprocesses the data by separating features (X) and the target variable (y).
#   - Splits the dataset into training and testing sets (80%/20% split).
# Interaction:
#   - Receives the full dataset (df) from `load_and_validate_data`.
#   - Outputs training and testing data for features and target variables.
#   - Provides the data for model training in `refine_and_register_model`.
######################
@task
def preprocess_data(df):
    log_message("preprocess_data", "Iniciando pré-processamento dos dados.")
    X = df.drop(columns=["MedHouseVal"])  # Dropping target column to get features
    y = df["MedHouseVal"]  # Target column
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    log_message("preprocess_data", "Dados divididos em treino e teste.")
    return X_train, X_test, y_train, y_test

######################
# Task 3: refine_and_register_model
# Purpose:
#   - Trains a Random Forest Regressor model using the training data.
#   - Evaluates the model using test data and calculates metrics (RMSE, R²).
#   - Logs metrics, parameters, and the model to MLflow for tracking.
#   - Registers the trained model in the MLflow Model Registry.
# Interaction:
#   - Takes training and testing data from `preprocess_data`.
#   - If data drift is detected (from `load_and_validate_data`), this function retrains and registers a new model version.
#   - Logs the model metrics and registration details for auditability.
######################
@task
def refine_and_register_model(X_train, y_train, X_test, y_test):
    log_message("refine_and_register_model", "Iniciando treinamento e registro do modelo.")
    mlflow.set_experiment("California Housing Refinement")  # Setting experiment name in MLflow
    with mlflow.start_run():  # Starting an MLflow run
        # Train a Random Forest Regressor model
        model = RandomForestRegressor(n_estimators=150, max_depth=15, random_state=42)
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

        # Calculate evaluation metrics
        rmse = mean_squared_error(y_test, predictions, squared=False)
        r2 = r2_score(y_test, predictions)

        # Log metrics and parameters
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2_score", r2)
        mlflow.log_param("n_estimators", 150)
        mlflow.log_param("max_depth", 15)

        # Save the model locally and log it to MLflow
        joblib.dump(model, "model.pkl")
        mlflow.sklearn.log_model(model, "model")

        log_message("refine_and_register_model", f"Modelo treinado. RMSE: {rmse:.2f}, R²: {r2:.2f}")

        # Register the model in MLflow Model Registry
        result = mlflow.register_model(f"runs:/{mlflow.active_run().info.run_id}/model", "CaliforniaHousingModel")
        log_message("refine_and_register_model", f"Modelo registrado como versão {result.version}.")
        return model, result.version

######################
# Task 4: monitor_drift
# Purpose:
#   - Monitors data drift between the reference dataset and current production data.
#   - Generates a drift report to detect significant changes in data distribution.
# Interaction:
#   - Takes reference data (from `load_and_validate_data`) and current production data.
#   - Returns a drift detection flag for further actions, such as retraining.
######################
@task
def monitor_drift(reference_data, production_data):
    log_message("monitor_drift", "Monitorando drift nos dados de produção.")
    report = Report(metrics=[DataDriftTable()])
    report.run(reference_data=reference_data, current_data=production_data)
    drift_detected = report.as_dict()["metrics"][0]["result"]["dataset_drift"]
    log_message("monitor_drift", f"Drift detectado: {drift_detected}")
    return drift_detected

######################
# Task 5: incorporate_feedback
# Purpose:
#   - Incorporates user-provided feedback data for future retraining or refinement.
#   - Reads feedback data from a CSV file and processes it into a structured format.
# Interaction:
#   - Searches for a `feedback.csv` file with user-provided corrections or updates.
#   - If available, returns the feedback data for further use; otherwise, logs that no feedback is found.
######################
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

######################
# Main Pipeline: mlops_pipeline
# Purpose:
#   - Orchestrates the entire workflow, from data validation to model training and monitoring.
# Interaction:
#   - Calls `load_and_validate_data` to check for data drift.
#   - If drift is detected, calls `preprocess_data` and `refine_and_register_model` to retrain the model.
#   - Calls `incorporate_feedback` to process user feedback for potential retraining.
######################
@flow(name="mlops-pipeline")
def mlops_pipeline():
    log_message("mlops_pipeline", "Iniciando execução do pipeline.")
    
    # Step 1: Load and validate data
    df, reference_data, drift_detected = load_and_validate_data()
    
    # Step 2: Check for data drift and retrain model if necessary
    if drift_detected:
        X_train, X_test, y_train, y_test = preprocess_data(df)
        model, new_version = refine_and_register_model(X_train, y_train, X_test, y_test)
        log_message("mlops_pipeline", f"Novo modelo treinado e registrado como versão {new_version}.")
    else:
        log_message("mlops_pipeline", "Nenhum drift detectado. Re-treinamento não necessário.")

    # Step 3: Process user feedback for potential retraining
    feedback_data = incorporate_feedback()
    if feedback_data is not None:
        log_message("mlops_pipeline", "Feedback processado para re-treinamento futuro.")
    else:
        log_message("mlops_pipeline", "Nenhum feedback disponível no momento.")

    log_message("mlops_pipeline", "Execução do pipeline concluída.")

######################
# Run the pipeline
######################
if __name__ == "__main__":
    mlops_pipeline()
