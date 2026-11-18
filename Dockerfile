FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1

WORKDIR /app

COPY requirements.txt ./
RUN python -m pip install --upgrade pip && python -m pip install -r requirements.txt

COPY . .
RUN test -f /app/artifacts/model_bundle/metadata.json \
    && useradd --create-home --uid 10001 supportflow \
    && mkdir -p /app/data \
    && chown -R supportflow:supportflow /app

USER supportflow

EXPOSE 8000 8501

CMD ["python", "-m", "scripts.run_api"]
