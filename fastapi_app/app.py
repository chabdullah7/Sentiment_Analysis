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

from dotenv import load_dotenv
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

from prometheus_client import (
    generate_latest,
    CollectorRegistry,
    CONTENT_TYPE_LATEST
)

load_dotenv()
warnings.filterwarnings("ignore")

# =========================
# TEXT PREPROCESSING
# =========================

def lemmatization(text):
    lemmatizer = WordNetLemmatizer()
    return " ".join([lemmatizer.lemmatize(w) for w in text.split()])

def remove_stop_words(text):
    stop_words = set(stopwords.words("english"))
    return " ".join([w for w in str(text).split() if w not in stop_words])

def removing_numbers(text):
    return ''.join([c for c in text if not c.isdigit()])

def lower_case(text):
    return " ".join([w.lower() for w in text.split()])

def removing_punctuations(text):
    text = re.sub(r'[^\w\s]', ' ', text)
    return re.sub(r'\s+', ' ', text).strip()

def removing_urls(text):
    return re.sub(r'https?://\S+|www\.\S+', '', text)

def normalize_text(text):
    text = lower_case(text)
    text = remove_stop_words(text)
    text = removing_numbers(text)
    text = removing_punctuations(text)
    text = removing_urls(text)
    text = lemmatization(text)
    return text


# =========================
# ENV CHECK
# =========================

DAGSHUB_TOKEN = os.getenv("DAGSHUB_TOKEN")
REPO_OWNER = os.getenv("REPO_OWNER")
REPO_NAME = os.getenv("REPO_NAME")
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")

if not DAGSHUB_TOKEN or not REPO_OWNER or not REPO_NAME or not MLFLOW_TRACKING_URI:
    raise EnvironmentError("Missing environment variables")

os.environ["MLFLOW_TRACKING_USERNAME"] = REPO_OWNER
os.environ["MLFLOW_TRACKING_PASSWORD"] = DAGSHUB_TOKEN
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

# =========================
# FASTAPI APP
# =========================

app = FastAPI()
templates = Jinja2Templates(directory="templates")

registry = CollectorRegistry()

# =========================
# LAZY MODEL LOADING (IMPORTANT FIX)
# =========================

model = None
vectorizer = None

def load_model_and_vectorizer():
    global model, vectorizer

    if model is None:
        client = mlflow.MlflowClient()
        versions = client.get_latest_versions("my_model", stages=["Production"])
        version = versions[0].version if versions else "1"

        model_uri = f"models:/my_model/{version}"
        model = mlflow.pyfunc.load_model(model_uri)

    if vectorizer is None:
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        path = os.path.join(base_dir, "models", "vectorizer.pkl")
        vectorizer = pickle.load(open(path, "rb"))

    return model, vectorizer


# =========================
# ROUTES
# =========================

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse(
        "index.html",
        {"request": request, "result": None}
    )


@app.post("/predict", response_class=HTMLResponse)
async def predict(request: Request, text: str = Form(...)):

    model, vectorizer = load_model_and_vectorizer()

    text = normalize_text(text)
    features = vectorizer.transform([text])

    df = pd.DataFrame(features.toarray())

    result = model.predict(df)
    prediction = int(result[0])

    return templates.TemplateResponse(
        "index.html",
        {"request": request, "result": prediction}
    )


@app.get("/metrics")
async def metrics():
    return Response(
        content=generate_latest(registry),
        media_type=CONTENT_TYPE_LATEST
    )