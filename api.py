from __future__ import annotations

from dataclasses import asdict
from functools import lru_cache
from pathlib import Path
from typing import Annotated, Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from supportflow.model import SupportFlowModel
from supportflow.text import TicketTextError, clean_ticket_text


ROOT = Path(__file__).resolve().parent
ARTIFACT_DIR = ROOT / "artifacts" / "model_bundle"


class TicketRequest(BaseModel):
    message: Annotated[str, Field(min_length=1, max_length=2_000)]
    similar_count: Annotated[int, Field(ge=0, le=5)] = 3


class BatchRequest(BaseModel):
    messages: Annotated[list[str], Field(min_length=1, max_length=50)]
    similar_count: Annotated[int, Field(ge=0, le=5)] = 0


class PredictionResponse(BaseModel):
    intent: str
    intent_display: str
    team: str
    priority: str
    priority_reason: str
    sla_hours: int
    sla_reason: str
    confidence: float
    decision: str
    threshold: float
    top_predictions: list[dict[str, Any]]
    similar_tickets: list[dict[str, Any]]
    inference_ms: float


class BatchResponse(BaseModel):
    count: int
    results: list[PredictionResponse]


app = FastAPI(
    title="SupportFlow AI API",
    version="1.0.0",
    description=(
        "Local demonstration API for confidence-aware banking support-ticket routing. "
        "Priority and SLA values are transparent demo business rules, not ML predictions or contractual commitments. "
        "Do not submit real personal or banking information."
    ),
)


@lru_cache(maxsize=1)
def get_model() -> SupportFlowModel:
    if not (ARTIFACT_DIR / "metadata.json").exists():
        raise RuntimeError("The trained model bundle is missing. Run the training script first.")
    return SupportFlowModel(ARTIFACT_DIR)


def _predict(message: str, similar_count: int) -> dict[str, Any]:
    try:
        cleaned = clean_ticket_text(message)
        return asdict(get_model().predict(cleaned, similar_count=similar_count))
    except TicketTextError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.get("/", tags=["service"])
def service_info() -> dict[str, str]:
    return {
        "service": "SupportFlow AI API",
        "version": app.version,
        "health": "/health",
        "interactive_docs": "/docs",
    }


@app.get("/health", tags=["service"])
def health() -> dict[str, str | bool]:
    artifacts_ready = (ARTIFACT_DIR / "metadata.json").exists()
    return {
        "status": "ready" if artifacts_ready else "model_missing",
        "artifacts_ready": artifacts_ready,
        "model_loaded": get_model.cache_info().currsize > 0,
    }


@app.post("/v1/predict", response_model=PredictionResponse, tags=["predictions"])
def predict_ticket(request: TicketRequest) -> dict[str, Any]:
    """Classify one ticket without saving it to the operations database."""
    return _predict(request.message, request.similar_count)


@app.post("/v1/predict/batch", response_model=BatchResponse, tags=["predictions"])
def predict_batch(request: BatchRequest) -> dict[str, Any]:
    """Classify up to 50 tickets without saving them to the operations database."""
    results = [_predict(message, request.similar_count) for message in request.messages]
    return {"count": len(results), "results": results}
