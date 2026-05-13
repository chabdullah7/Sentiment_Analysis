from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse, Response
from fastapi.templating import Jinja2Templates

import mlflow
import pickle
import os
import pandas as pd
import time
import string
import re
import warnings
import dagshub

from dotenv import load_dotenv

from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

from prometheus_client import (
    Counter,
    Histogram,
    generate_latest,
    CollectorRegistry,
    CONTENT_TYPE_LATEST
)

load_dotenv()

warnings.simplefilter("ignore", UserWarning)
warnings.filterwarnings("ignore")


# =========================================================
# TEXT PREPROCESSING
# =========================================================

def lemmatization(text):
    lemmatizer = WordNetLemmatizer()
    text = text.split()
    text = [lemmatizer.lemmatize(word) for word in text]
    return " ".join(text)


def remove_stop_words(text):
    stop_words = set(stopwords.words("english"))
    text = [word for word in str(text).split() if word not in stop_words]
    return " ".join(text)


def removing_numbers(text):
    return ''.join([char for char in text if not char.isdigit()])


def lower_case(text):
    text = text.split()
    text = [word.lower() for word in text]
    return " ".join(text)


def removing_punctuations(text):
    text = re.sub(r'[%s]' % re.escape(string.punctuation), ' ', text)
    text = text.replace('؛', "")
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def removing_urls(text):
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    return url_pattern.sub(r'', text)


def normalize_text(text):
    text = lower_case(text)
    text = remove_stop_words(text)
    text = removing_numbers(text)
    text = removing_punctuations(text)
    text = removing_urls(text)
    text = lemmatization(text)
    return text


# =========================================================
# LOCAL MODE (ACTIVE)
# =========================================================

REPO_OWNER = os.getenv("REPO_OWNER")
REPO_NAME = os.getenv("REPO_NAME")
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")
DAGSHUB_TOKEN = os.getenv("DAGSHUB_TOKEN")

if MLFLOW_TRACKING_URI:
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

if REPO_OWNER and REPO_NAME:
    dagshub.init(
        repo_owner=REPO_OWNER,
        repo_name=REPO_NAME,
        mlflow=True
    )


# =========================================================
# PRODUCTION / REMOTE MODE (COMMENTED - KEEP AS YOU WANTED)
# =========================================================

# DAGSHUB_TOKEN = os.getenv("DAGSHUB_TOKEN")
# REPO_OWNER = os.getenv("REPO_OWNER")
# REPO_NAME = os.getenv("REPO_NAME")
# MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")

# if not DAGSHUB_TOKEN:
#     raise EnvironmentError("DAGSHUB_TOKEN environment variable is not set")

# os.environ["MLFLOW_TRACKING_USERNAME"] = DAGSHUB_TOKEN
# os.environ["MLFLOW_TRACKING_PASSWORD"] = DAGSHUB_TOKEN

# mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)


# =========================================================
# FASTAPI APP
# =========================================================

app = FastAPI()
templates = Jinja2Templates(directory="templates")


# =========================================================
# PROMETHEUS METRICS (COMMENTED - SAFE)
# =========================================================

registry = CollectorRegistry()

# REQUEST_COUNT = Counter(
#     "app_request_count",
#     "Total number of requests",
#     ["method", "endpoint"],
#     registry=registry
# )

# REQUEST_LATENCY = Histogram(
#     "app_request_latency_seconds",
#     "Latency of requests",
#     ["endpoint"],
#     registry=registry
# )

# PREDICTION_COUNT = Counter(
#     "model_prediction_count",
#     "Prediction count",
#     ["prediction"],
#     registry=registry
# )


# =========================================================
# LOAD MODEL
# =========================================================

model_name = "my_model"


def get_latest_model_version(model_name):

    client = mlflow.MlflowClient()

    latest_version = client.get_latest_versions(
        model_name,
        stages=["Production"]
    )

    if not latest_version:
        latest_version = client.get_latest_versions(
            model_name,
            stages=["None"]
        )

    return latest_version[0].version if latest_version else None


model_version = get_latest_model_version(model_name)

model_uri = f"models:/{model_name}/{model_version}"

print(f"Fetching model from: {model_uri}")

model = mlflow.pyfunc.load_model(model_uri)


# =========================================================
# FIXED VECTORIZER PATH (MAIN BUG FIX)
# =========================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

vectorizer_path = os.path.join(BASE_DIR, "models", "vectorizer.pkl")

vectorizer = pickle.load(open(vectorizer_path, "rb"))


# =========================================================
# ROUTES
# =========================================================

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):

    # REQUEST_COUNT.labels(method="GET", endpoint="/").inc()

    start_time = time.time()

    response = templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "result": None
        }
    )

    # REQUEST_LATENCY.labels(endpoint="/").observe(time.time() - start_time)

    return response


@app.post("/predict", response_class=HTMLResponse)
async def predict(request: Request, text: str = Form(...)):

    # REQUEST_COUNT.labels(method="POST", endpoint="/predict").inc()

    start_time = time.time()

    text = normalize_text(text)

    features = vectorizer.transform([text])

    features_df = pd.DataFrame(
        features.toarray(),
        columns=[str(i) for i in range(features.shape[1])]
    )

    result = model.predict(features_df)
    prediction = int(result[0])

    # PREDICTION_COUNT.labels(prediction=str(prediction)).inc()
    # REQUEST_LATENCY.labels(endpoint="/predict").observe(time.time() - start_time)

    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "result": prediction
        }
    )


@app.get("/metrics")
async def metrics():

    return Response(
        content=generate_latest(registry),
        media_type=CONTENT_TYPE_LATEST
    )


# =========================================================
# MAIN
# =========================================================

if __name__ == "__main__":

    import uvicorn

    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=5000,
        reload=True
    )