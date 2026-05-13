import os
import mlflow
from mlflow.tracking import MlflowClient
from dotenv import load_dotenv

load_dotenv()


def promote_model():

    dagshub_token = os.getenv("DAGSHUB_TOKEN")
    repo_owner = os.getenv("REPO_OWNER")
    repo_name = os.getenv("REPO_NAME")
    mlflow_uri = os.getenv("MLFLOW_TRACKING_URI")

    if not dagshub_token or not repo_owner or not mlflow_uri:
        raise EnvironmentError("Missing environment variables")

    # Auth
    os.environ["MLFLOW_TRACKING_USERNAME"] = repo_owner
    os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

    mlflow.set_tracking_uri(mlflow_uri)

    client = MlflowClient()
    model_name = "my_model"

    # =========================
    # GET ALL VERSIONS
    # =========================
    versions = client.search_model_versions(f"name='{model_name}'")

    if not versions:
        raise ValueError("No model exists in registry")

    # latest version (safe conversion)
    latest = max(versions, key=lambda v: int(v.version))

    print("Latest model version:", latest.version)

    # =========================
    # NEW WAY (NO DEPRECATED STAGES)
    # =========================
    client.set_registered_model_alias(
        name=model_name,
        alias="production",
        version=latest.version
    )

    print(f"Model version {latest.version} set as PRODUCTION (alias)")


if __name__ == "__main__":
    promote_model()