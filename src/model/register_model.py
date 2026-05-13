import sys
import os

# =========================================================
# FIX: allow "src" imports in DVC / CI / local
# =========================================================
sys.path.append(
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../..")
    )
)

import json
import mlflow
import dagshub

from mlflow.tracking import MlflowClient
from src.logger import logger
from dotenv import load_dotenv

import warnings
warnings.simplefilter("ignore", UserWarning)
warnings.filterwarnings("ignore")

load_dotenv()


# =========================================================
# PRODUCTION MODE
# =========================================================

dagshub_token = os.getenv("DAGSHUB_TOKEN")
REPO_OWNER = os.getenv("REPO_OWNER")
REPO_NAME = os.getenv("REPO_NAME")

if not dagshub_token:
    raise EnvironmentError("DAGSHUB_TOKEN is not set")

if not REPO_OWNER or not REPO_NAME:
    raise EnvironmentError("REPO_OWNER or REPO_NAME is missing")

# Authentication
os.environ["MLFLOW_TRACKING_USERNAME"] = REPO_OWNER
os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

mlflow.set_tracking_uri(
    f"https://dagshub.com/{REPO_OWNER}/{REPO_NAME}.mlflow"
)


# =========================================================
# FUNCTIONS
# =========================================================

def load_model_info(file_path: str) -> dict:

    try:

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"{file_path} not found")

        with open(file_path, "r") as file:
            model_info = json.load(file)

        logger.info(f"Model info loaded from {file_path}")

        return model_info

    except Exception as e:
        logger.error(f"Error loading model info: {e}")
        raise


def register_model(model_name: str, model_info: dict):

    try:

        model_uri = model_info["model_uri"]

        logger.info(f"Registering model from URI: {model_uri}")

        # Register model
        model_version = mlflow.register_model(
            model_uri=model_uri,
            name=model_name
        )

        logger.info(
            f"Registered model version: {model_version.version}"
        )

        # Tag as Staging
        client = MlflowClient()

        client.set_model_version_tag(
            name=model_name,
            version=model_version.version,
            key="stage",
            value="Staging"
        )

        logger.info(
            f"Model {model_name} v{model_version.version} tagged as Staging"
        )

    except Exception as e:
        logger.error(f"Model registration failed: {e}")
        raise


# =========================================================
# MAIN
# =========================================================

def main():

    try:

        model_info = load_model_info(
            "reports/experiment_info.json"
        )

        register_model(
            model_name="my_model",
            model_info=model_info
        )

        logger.info(
            "Model registration pipeline completed successfully"
        )

    except Exception as e:

        logger.error(f"Pipeline failed: {e}")
        print(f"Error: {e}")


if __name__ == "__main__":
    main()