import sys
import os

sys.path.append(
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../..")
    )
)

import numpy as np
import pandas as pd
import pickle
import json

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    roc_auc_score
)

import mlflow
import mlflow.sklearn
import dagshub

from dotenv import load_dotenv
from src.logger import logger

load_dotenv()

# =========================================================
# DAGSHUB / MLFLOW CONFIG
# =========================================================

dagshub_token = os.getenv("DAGSHUB_TOKEN")
REPO_OWNER = os.getenv("REPO_OWNER")
REPO_NAME = os.getenv("REPO_NAME")

if not dagshub_token:
    raise EnvironmentError("DAGSHUB_TOKEN is not set")

if not REPO_OWNER or not REPO_NAME:
    raise EnvironmentError("REPO_OWNER or REPO_NAME is not set")

os.environ["MLFLOW_TRACKING_USERNAME"] = REPO_OWNER
os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

mlflow.set_tracking_uri(
    f"https://dagshub.com/{REPO_OWNER}/{REPO_NAME}.mlflow"
)

# =========================================================
# UTIL FUNCTIONS
# =========================================================

def load_model(file_path: str):
    try:
        with open(file_path, "rb") as file:
            model = pickle.load(file)

        logger.info(f"Model loaded from {file_path}")
        return model

    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise


def load_data(file_path: str):
    try:
        df = pd.read_csv(file_path)

        logger.info(f"Data loaded from {file_path}")
        return df

    except Exception as e:
        logger.error(f"Error loading data: {e}")
        raise


def evaluate_model(model, X_test, y_test):
    try:
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
            "auc": roc_auc_score(y_test, y_proba)
        }

        logger.info("Evaluation completed")
        return metrics

    except Exception as e:
        logger.error(f"Evaluation error: {e}")
        raise


def save_metrics(metrics, path):
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)

        with open(path, "w") as f:
            json.dump(metrics, f, indent=4)

        logger.info(f"Metrics saved to {path}")

    except Exception as e:
        logger.error(f"Error saving metrics: {e}")
        raise


def save_model_info(run_id, model_uri, path):
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)

        model_info = {
            "run_id": run_id,
            "model_uri": model_uri
        }

        with open(path, "w") as f:
            json.dump(model_info, f, indent=4)

        logger.info(f"Model info saved to {path}")

    except Exception as e:
        logger.error(f"Error saving model info: {e}")
        raise


# =========================================================
# MAIN PIPELINE
# =========================================================

def main():
    try:
        mlflow.set_experiment("my-dvc-pipeline")

        with mlflow.start_run() as run:

            # =========================
            # LOAD MODEL + DATA
            # =========================
            model = load_model("./models/model.pkl")

            test_data = load_data("./data/processed/test_bow.csv")

            X_test = test_data.iloc[:, :-1].values
            y_test = test_data.iloc[:, -1].values

            # =========================
            # EVALUATE
            # =========================
            metrics = evaluate_model(model, X_test, y_test)

            # =========================
            # SAVE METRICS
            # =========================
            save_metrics(metrics, "reports/metrics.json")

            # =========================
            # LOG METRICS
            # =========================
            for k, v in metrics.items():
                mlflow.log_metric(k, v)

            # =========================
            # LOG PARAMS
            # =========================
            if hasattr(model, "get_params"):
                for k, v in model.get_params().items():
                    mlflow.log_param(k, v)

            # =========================
            # LOG MODEL (FIXED)
            # =========================
            mlflow.sklearn.log_model(
                sk_model=model,
                artifact_path="model"
            )

            logger.info("Model logged to MLflow successfully")

            # =========================
            # MODEL INFO (SAFE)
            # =========================
            model_uri = f"runs:/{run.info.run_id}/model"

            save_model_info(
                run_id=run.info.run_id,
                model_uri=model_uri,
                path="reports/experiment_info.json"
            )

            # =========================
            # LOG ARTIFACTS
            # =========================
            mlflow.log_artifact("reports/metrics.json")

            logger.info("Model evaluation pipeline completed successfully")

    except Exception as e:
        logger.error(f"Model evaluation pipeline failed: {e}")
        print(f"Error: {e}")


if __name__ == "__main__":
    main()