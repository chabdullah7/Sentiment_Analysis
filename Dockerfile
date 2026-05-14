FROM python:3.10-slim

WORKDIR /app

# =========================
# Install system deps
# =========================
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# =========================
# requirements first (cache layer)
# =========================
COPY docker_requirements.txt .

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r docker_requirements.txt

# =========================
# copy app code
# =========================
COPY fastapi_app/ /app/fastapi_app/
COPY models/ /app/models/
COPY src/ /app/src/

# =========================
# nltk data
# =========================
RUN python -c "import nltk; nltk.download('stopwords'); nltk.download('wordnet')"

# =========================
# expose port
# =========================
EXPOSE 8000

# =========================
# DEFAULT (LOCAL DEV)
# =========================
CMD ["uvicorn", "fastapi_app.app:app", "--host", "0.0.0.0", "--port", "8000"]

# =========================
# PRODUCTION (use in deploy)
# =========================
# CMD ["gunicorn", "-k", "uvicorn.workers.UvicornWorker", \
#      "fastapi_app.app:app", "--bind", "0.0.0.0:8000", "--timeout", "120"]