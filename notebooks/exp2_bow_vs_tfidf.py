import setuptools
import os
import re
import string
import pandas as pd
pd.set_option('future.no_silent_downcasting', True)

import numpy as np
import mlflow
import mlflow.sklearn
import dagshub
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import scipy.sparse
from dotenv import load_dotenv
import nltk
import warnings

warnings.simplefilter("ignore", UserWarning)
warnings.filterwarnings("ignore")

# ========================== NLTK SETUP ==========================
nltk.download("stopwords")
nltk.download("wordnet")

# ========================== LOAD ENV ==========================
load_dotenv()

# IMPORTANT: MLflow AUTH
os.environ["MLFLOW_TRACKING_USERNAME"] = os.getenv("REPO_OWNER")
os.environ["MLFLOW_TRACKING_PASSWORD"] = os.getenv("DAGSHUB_TOKEN")

# ========================== CONFIG ==========================
CONFIG = {
    "data_path": "notebooks/data.csv",
    "test_size": 0.2,
    "mlflow_tracking_uri": os.getenv("MLFLOW_TRACKING_URI"),
    "dagshub_repo_owner": os.getenv("REPO_OWNER"),
    "dagshub_repo_name": os.getenv("REPO_NAME"),
    "experiment_name": "BOW vs TfIdf"
}

# safety check
if not CONFIG["mlflow_tracking_uri"]:
    raise ValueError("MLFLOW_TRACKING_URI missing in .env")

# ========================== MLflow + DAGSHUB SETUP ==========================
mlflow.set_tracking_uri(CONFIG["mlflow_tracking_uri"])

dagshub.init(
    repo_owner=CONFIG["dagshub_repo_owner"],
    repo_name=CONFIG["dagshub_repo_name"],
    mlflow=True
)

mlflow.set_experiment(CONFIG["experiment_name"])

# ========================== TEXT PREPROCESSING ==========================
def lemmatization(text):
    lemmatizer = WordNetLemmatizer()
    return " ".join([lemmatizer.lemmatize(word) for word in text.split()])

def remove_stop_words(text):
    stop_words = set(stopwords.words("english"))
    return " ".join([word for word in text.split() if word not in stop_words])

def removing_numbers(text):
    return ''.join([char for char in text if not char.isdigit()])

def lower_case(text):
    return text.lower()

def removing_punctuations(text):
    return re.sub(f"[{re.escape(string.punctuation)}]", ' ', text)

def removing_urls(text):
    return re.sub(r'https?://\S+|www\.\S+', '', text)

def normalize_text(df):
    df['review'] = df['review'].apply(lower_case)
    df['review'] = df['review'].apply(remove_stop_words)
    df['review'] = df['review'].apply(removing_numbers)
    df['review'] = df['review'].apply(removing_punctuations)
    df['review'] = df['review'].apply(removing_urls)
    df['review'] = df['review'].apply(lemmatization)
    return df

# ========================== LOAD DATA ==========================
def load_data(file_path):
    df = pd.read_csv(file_path)
    df = normalize_text(df)
    df = df[df['sentiment'].isin(['positive', 'negative'])]
    df['sentiment'] = df['sentiment'].map({'negative': 0, 'positive': 1}).astype(int)
    return df

# ========================== MODELS ==========================
VECTORIZERS = {
    'BoW': CountVectorizer(),
    'TF-IDF': TfidfVectorizer()
}

ALGORITHMS = {
    'LogisticRegression': LogisticRegression(),
    'MultinomialNB': MultinomialNB(),
    'XGBoost': XGBClassifier(eval_metric="logloss"),
    'RandomForest': RandomForestClassifier(),
    'GradientBoosting': GradientBoostingClassifier()
}

# ========================== TRAINING ==========================
def train_and_evaluate(df):
    with mlflow.start_run(run_name="All Experiments"):
        for algo_name, algorithm in ALGORITHMS.items():
            for vec_name, vectorizer in VECTORIZERS.items():

                with mlflow.start_run(run_name=f"{algo_name} with {vec_name}", nested=True):

                    X = vectorizer.fit_transform(df['review'])
                    y = df['sentiment']

                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y, test_size=CONFIG["test_size"], random_state=42
                    )

                    mlflow.log_params({
                        "vectorizer": vec_name,
                        "algorithm": algo_name,
                        "test_size": CONFIG["test_size"]
                    })

                    model = algorithm
                    model.fit(X_train, y_train)

                    log_model_params(algo_name, model)

                    y_pred = model.predict(X_test)

                    metrics = {
                        "accuracy": accuracy_score(y_test, y_pred),
                        "precision": precision_score(y_test, y_pred),
                        "recall": recall_score(y_test, y_pred),
                        "f1_score": f1_score(y_test, y_pred)
                    }

                    mlflow.log_metrics(metrics)

                    input_example = (
                        X_test[:5].toarray()
                        if scipy.sparse.issparse(X_test)
                        else X_test[:5]
                    )

                    mlflow.sklearn.log_model(model, "model", input_example=input_example)

                    print(f"\n{algo_name} + {vec_name}")
                    print(metrics)

# ========================== LOG PARAMS ==========================
def log_model_params(algo_name, model):
    params = {}

    if algo_name == 'LogisticRegression':
        params["C"] = model.C
    elif algo_name == 'MultinomialNB':
        params["alpha"] = model.alpha
    elif algo_name == 'XGBoost':
        params["n_estimators"] = model.n_estimators
        params["learning_rate"] = model.learning_rate
    elif algo_name == 'RandomForest':
        params["n_estimators"] = model.n_estimators
        params["max_depth"] = model.max_depth
    elif algo_name == 'GradientBoosting':
        params["n_estimators"] = model.n_estimators
        params["learning_rate"] = model.learning_rate
        params["max_depth"] = model.max_depth

    mlflow.log_params(params)

# ========================== RUN ==========================
if __name__ == "__main__":
    df = load_data(CONFIG["data_path"])
    train_and_evaluate(df)