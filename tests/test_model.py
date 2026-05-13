import unittest
import mlflow
import os
import pandas as pd
import pickle
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)


class TestModelLoading(unittest.TestCase):

    @classmethod
    def setUpClass(cls):

        # =========================
        # ENV VARIABLES
        # =========================
        dagshub_token = os.getenv("DAGSHUB_TOKEN")
        repo_owner = os.getenv("REPO_OWNER")
        mlflow_uri = os.getenv("MLFLOW_TRACKING_URI")

        if not dagshub_token:
            raise EnvironmentError("DAGSHUB_TOKEN not set")

        if not mlflow_uri:
            raise EnvironmentError("MLFLOW_TRACKING_URI not set")

        # =========================
        # MLFLOW AUTH
        # =========================
        os.environ["MLFLOW_TRACKING_USERNAME"] = repo_owner
        os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

        mlflow.set_tracking_uri(mlflow_uri)

        # =========================
        # LOAD MODEL
        # =========================
        cls.model_name = "my_model"

        cls.model_version = cls.get_latest_model_version(
            cls.model_name
        )

        if not cls.model_version:
            raise ValueError(
                "No model found in MLflow Registry"
            )

        cls.model_uri = (
            f"models:/{cls.model_name}/{cls.model_version}"
        )

        cls.model = mlflow.pyfunc.load_model(
            cls.model_uri
        )

        # =========================
        # LOAD VECTORIZER
        # =========================
        with open("models/vectorizer.pkl", "rb") as f:
            cls.vectorizer = pickle.load(f)

        # =========================
        # LOAD TEST DATA
        # =========================
        cls.data = pd.read_csv(
            "data/processed/test_bow.csv"
        )

    # =========================
    # GET MODEL VERSION
    # =========================
    @staticmethod
    def get_latest_model_version(model_name):

        client = mlflow.MlflowClient()

        # Try Production
        versions = client.get_latest_versions(
            model_name,
            stages=["Production"]
        )

        if versions:
            return versions[0].version

        # Try Staging
        versions = client.get_latest_versions(
            model_name,
            stages=["Staging"]
        )

        if versions:
            return versions[0].version

        # Try latest registered model
        latest_versions = client.search_model_versions(
            f"name='{model_name}'"
        )

        if latest_versions:
            return latest_versions[-1].version

        return None

    # =========================
    # TEST 1
    # =========================
    def test_model_loaded(self):

        self.assertIsNotNone(self.model)

    # =========================
    # TEST 2
    # =========================
    def test_signature(self):

        text = "this is a test sentence"

        X = self.vectorizer.transform([text])

        input_df = pd.DataFrame(
            X.toarray()
        )

        prediction = self.model.predict(
            input_df
        )

        self.assertEqual(
            input_df.shape[0],
            len(prediction)
        )

    # =========================
    # TEST 3
    # =========================
    def test_performance(self):

        X = self.data.iloc[:, :-1]
        y = self.data.iloc[:, -1]

        preds = self.model.predict(X)

        acc = accuracy_score(y, preds)
        prec = precision_score(y, preds)
        rec = recall_score(y, preds)
        f1 = f1_score(y, preds)

        print("\nModel Performance:")
        print("Accuracy:", acc)
        print("Precision:", prec)
        print("Recall:", rec)
        print("F1:", f1)

        self.assertGreaterEqual(acc, 0.50)
        self.assertGreaterEqual(f1, 0.50)


if __name__ == "__main__":
    unittest.main()