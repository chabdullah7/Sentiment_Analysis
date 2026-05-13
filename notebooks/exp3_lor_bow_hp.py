import os
import re
import string
import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
import dagshub
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from dotenv import load_dotenv

import warnings
warnings.simplefilter("ignore", UserWarning)
warnings.filterwarnings("ignore")

# ========================== ENV LOAD ==========================
load_dotenv()

REPO_OWNER = os.getenv("REPO_OWNER")
REPO_NAME = os.getenv("REPO_NAME")
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")
DAGSHUB_TOKEN = os.getenv("DAGSHUB_TOKEN")

# MLflow Auth (IMPORTANT for 403 fix)
os.environ["MLFLOW_TRACKING_USERNAME"] = REPO_OWNER
os.environ["MLFLOW_TRACKING_PASSWORD"] = DAGSHUB_TOKEN

# ========================== MLflow + DAGSHUB ==========================
dagshub.init(
    repo_owner=REPO_OWNER,
    repo_name=REPO_NAME,
    mlflow=True
)

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment("LoR Hyperparameter Tuning")


# ========================== TEXT PREPROCESSING ==========================
def preprocess_text(text):
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words("english"))

    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = re.sub(f"[{re.escape(string.punctuation)}]", " ", text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = " ".join([
        lemmatizer.lemmatize(word)
        for word in text.split()
        if word not in stop_words
    ])

    return text.strip()


# ========================== DATA ==========================
def load_and_prepare_data(filepath):
    df = pd.read_csv(filepath)

    df["review"] = df["review"].astype(str).apply(preprocess_text)

    df = df[df["sentiment"].isin(["positive", "negative"])]
    df["sentiment"] = df["sentiment"].map({"negative": 0, "positive": 1}).astype(int)

    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(df["review"])
    y = df["sentiment"]

    return train_test_split(X, y, test_size=0.2, random_state=42), vectorizer


# ========================== TRAINING ==========================
def train_and_log_model(X_train, X_test, y_train, y_test, vectorizer):

    param_grid = {
        "C": [0.1, 1, 10],
        "penalty": ["l1", "l2"],
        "solver": ["liblinear"]
    }

    with mlflow.start_run(run_name="GridSearch_LR"):

        grid_search = GridSearchCV(
            LogisticRegression(),
            param_grid,
            cv=5,
            scoring="f1",
            n_jobs=1
        )

        grid_search.fit(X_train, y_train)

        for params, mean_score, std_score in zip(
            grid_search.cv_results_["params"],
            grid_search.cv_results_["mean_test_score"],
            grid_search.cv_results_["std_test_score"]
        ):

            with mlflow.start_run(run_name=f"LR {params}", nested=True):

                model = LogisticRegression(**params)
                model.fit(X_train, y_train)

                y_pred = model.predict(X_test)

                metrics = {
                    "accuracy": accuracy_score(y_test, y_pred),
                    "precision": precision_score(y_test, y_pred),
                    "recall": recall_score(y_test, y_pred),
                    "f1_score": f1_score(y_test, y_pred),
                    "cv_mean": mean_score,
                    "cv_std": std_score
                }

                mlflow.log_params(params)
                mlflow.log_metrics(metrics)

                print(f"Params: {params} | F1: {metrics['f1_score']:.4f}")

        best_model = grid_search.best_estimator_

        mlflow.log_params(grid_search.best_params_)
        mlflow.log_metric("best_f1", grid_search.best_score_)
        mlflow.sklearn.log_model(best_model, "model")

        print("\nBest Params:", grid_search.best_params_)
        print("Best F1:", grid_search.best_score_)


# ========================== RUN ==========================
if __name__ == "__main__":
    (X_train, X_test, y_train, y_test), vectorizer = load_and_prepare_data("notebooks/data.csv")
    train_and_log_model(X_train, X_test, y_train, y_test, vectorizer)