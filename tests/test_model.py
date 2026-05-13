import unittest
import mlflow
import os
import pandas as pd
import pickle
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


class TestModelLoading(unittest.TestCase):

    @classmethod
    def setUpClass(cls):

        # =========================
        # ENV VARIABLES
        # =========================
        dagshub_token = os.getenv("DAGSHUB_TOKEN")
        repo_owner = os.getenv("REPO_OWNER")
        repo_name = os.getenv("REPO_NAME")
        mlflow_uri = os.getenv("MLFLOW_TRACKING_URI")

        if not dagshub_token:
            raise EnvironmentError("DAGSHUB_TOKEN not set")

        if not mlflow_uri:
            raise EnvironmentError("MLFLOW_TRACKING_URI not set")

        # =========================
        # MLflow AUTH
        # =========================
        os.environ["MLFLOW_TRACKING_USERNAME"] = repo_owner
        os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

        mlflow.set_tracking_uri(mlflow_uri)

        # =========================
        # LOAD MODEL (PRODUCTION)
        # =========================
        cls.model_name = "my_model"
        cls.model_version = cls.get_latest_model_version(cls.model_name)

        if not cls.model_version:
            raise ValueError("No model found in Production")

        cls.model_uri = f"models:/{cls.model_name}/{cls.model_version}"
        cls.model = mlflow.pyfunc.load_model(cls.model_uri)

        # =========================
        # LOAD VECTORIZER
        # =========================
        cls.vectorizer = pickle.load(open("models/vectorizer.pkl", "rb"))

        # =========================
        # LOAD TEST DATA
        # =========================
        cls.data = pd.read_csv("data/processed/test_bow.csv")

    # =========================
    # GET LATEST MODEL VERSION
    # =========================
    @staticmethod
    def get_latest_model_version(model_name, stage="Production"):
        client = mlflow.MlflowClient()
        versions = client.get_latest_versions(model_name, stages=[stage])
        return versions[0].version if versions else None

    # =========================
    # TEST 1: MODEL LOADED
    # =========================
    def test_model_loaded(self):
        self.assertIsNotNone(self.model)

    # =========================
    # TEST 2: SIGNATURE TEST
    # =========================
    def test_signature(self):

        text = "this is a test sentence"

        X = self.vectorizer.transform([text])
        input_df = pd.DataFrame(X.toarray())

        prediction = self.model.predict(input_df)

        self.assertEqual(input_df.shape[0], len(prediction))

    # =========================
    # TEST 3: PERFORMANCE TEST
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

        # minimum thresholds
        self.assertGreaterEqual(acc, 0.50)
        self.assertGreaterEqual(f1, 0.50)


if __name__ == "__main__":
    unittest.main()