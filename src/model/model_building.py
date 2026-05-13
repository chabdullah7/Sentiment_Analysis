from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse, Response
from fastapi.templating import Jinja2Templates

import os
import pickle
import pandas as pd
import mlflow
from prometheus_client import generate_latest, CollectorRegistry, CONTENT_TYPE_LATEST

app = FastAPI()
templates = Jinja2Templates(directory="fastapi_app/templates")

registry = CollectorRegistry()

# =========================
# LOAD MODEL + VECTORIZER
# =========================

model = pickle.load(open("models/model.pkl", "rb"))
vectorizer = pickle.load(open("models/vectorizer.pkl", "rb"))


# =========================
# HOME PAGE
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

    # transform text using SAME vectorizer
    X = vectorizer.transform([text])

    # convert to dataframe (safe for sklearn/mlflow)
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