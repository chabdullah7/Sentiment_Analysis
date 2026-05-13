from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse, Response
from fastapi.templating import Jinja2Templates

import pickle
import pandas as pd
from prometheus_client import (
    generate_latest,
    CollectorRegistry,
    CONTENT_TYPE_LATEST
)

app = FastAPI()

templates = Jinja2Templates(directory="fastapi_app/templates")
registry = CollectorRegistry()

# =========================
# LAZY LOAD
# =========================

model = None
vectorizer = None


def load_artifacts():
    global model, vectorizer

    if model is None:
        with open("models/model.pkl", "rb") as f:
            model = pickle.load(f)

    if vectorizer is None:
        with open("models/vectorizer.pkl", "rb") as f:
            vectorizer = pickle.load(f)

    return model, vectorizer


# =========================
# HOME
# =========================

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):

    return templates.TemplateResponse(
        request,
        "index.html",
        {"result": None}
    )


# =========================
# PREDICT
# =========================

@app.post("/predict", response_class=HTMLResponse)
async def predict(request: Request, text: str = Form(...)):

    model, vectorizer = load_artifacts()

    X = vectorizer.transform([text])

    df = pd.DataFrame(X.toarray())

    prediction = model.predict(df)[0]

    return templates.TemplateResponse(
        request,
        "index.html",
        {"result": int(prediction)}
    )


# =========================
# METRICS
# =========================

@app.get("/metrics")
async def metrics():

    return Response(
        content=generate_latest(registry),
        media_type=CONTENT_TYPE_LATEST
    )