# ==============================================================================
# KishoLens Backend Dockerfile (Optimized for Google Cloud Run & Fast Container Startup)
# ==============================================================================

FROM python:3.11-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=8080

WORKDIR /app

# Install system build dependencies required for NLP tokenizers (sudachipy, mecab)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python requirements
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Download lightweight spacy and nltk models
RUN python -m spacy download en_core_web_sm || true && \
    python -c "import nltk; nltk.download('vader_lexicon', quiet=True); nltk.download('punkt', quiet=True); nltk.download('punkt_tab', quiet=True)" || true

# Copy project source and pre-computed data caches
COPY kisholens/ ./kisholens/
COPY data/ ./data/
COPY pyproject.toml .

EXPOSE 8080

# Run uvicorn on dynamically bound $PORT for Google Cloud Run
CMD exec uvicorn kisholens.api.main:app --host 0.0.0.0 --port ${PORT:-8080}
