import mlflow
from mlflow.tracking import MlflowClient

mlflow.set_tracking_uri(
    "https://dagshub.com/chabdullah7/Sentiment_Analysis.mlflow"
)

client = MlflowClient()

for m in client.search_registered_models():
    print("Model Name:", m.name)