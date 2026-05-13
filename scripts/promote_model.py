# promote_model.py

import os
import mlflow
from dotenv import load_dotenv

load_dotenv()


def promote_model():
    
    # Load environment variables
    dagshub_token = os.getenv("DAGSHUB_TOKEN")
    repo_owner = os.getenv("REPO_OWNER")
    repo_name = os.getenv("REPO_NAME")
    mlflow_uri = os.getenv("MLFLOW_TRACKING_URI")

    if not dagshub_token:
        raise EnvironmentError("DAGSHUB_TOKEN is not set")

    if not repo_owner or not repo_name:
        raise EnvironmentError("REPO_OWNER / REPO_NAME not set")

    if not mlflow_uri:
        raise EnvironmentError("MLFLOW_TRACKING_URI not set")

    
    # Set MLflow auth for DagsHub
    os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
    os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

    mlflow.set_tracking_uri(mlflow_uri)

    client = mlflow.MlflowClient()

    model_name = "my_model"

    
    # Get staging model
    staging_versions = client.get_latest_versions(
        model_name,
        stages=["Staging"]
    )

    if not staging_versions:
        raise ValueError("No model found in Staging")

    latest_version = staging_versions[0].version


    # Archive production models
    prod_versions = client.get_latest_versions(
        model_name,
        stages=["Production"]
    )

    for v in prod_versions:
        client.transition_model_version_stage(
            name=model_name,
            version=v.version,
            stage="Archived"
        )

    
    # Promote new model
    client.transition_model_version_stage(
        name=model_name,
        version=latest_version,
        stage="Production"
    )

    print(f"Model version {latest_version} promoted to Production")


if __name__ == "__main__":
    promote_model()