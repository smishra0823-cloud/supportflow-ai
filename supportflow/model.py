from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import joblib
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

from .taxonomy import priority_for_intent, sla_for_priority, team_for_intent
from .text import clean_ticket_text, humanize_intent


@dataclass(frozen=True)
class Prediction:
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
    top_predictions: list[dict[str, float | str]]
    similar_tickets: list[dict[str, float | str]]
    inference_ms: float


class SupportFlowModel:
    def __init__(self, artifact_dir: Path) -> None:
        self.artifact_dir = artifact_dir
        self.metadata = json.loads((artifact_dir / "metadata.json").read_text(encoding="utf-8"))
        self.classifier = joblib.load(artifact_dir / "classifier.joblib")
        self.encoder = SentenceTransformer(str(artifact_dir / "embedding_model"), device="cpu")
        retrieval = np.load(artifact_dir / "retrieval_embeddings.npz")
        self.retrieval_embeddings = retrieval["embeddings"].astype(np.float32)
        self.retrieval_rows = pd.read_csv(artifact_dir / "resolved_tickets.csv")
        if len(self.retrieval_embeddings) != len(self.retrieval_rows):
            raise ValueError("Retrieval artifacts do not contain the same number of rows.")
        self.threshold = float(self.metadata["auto_route_threshold"])

    def predict(self, text: str, similar_count: int = 3) -> Prediction:
        cleaned = clean_ticket_text(text)
        start = perf_counter()
        embedding = self.encoder.encode(
            [cleaned],
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        ).astype(np.float32)
        probabilities = self.classifier.predict_proba(embedding)[0]
        ranked = np.argsort(probabilities)[::-1]
        best_index = int(ranked[0])
        intent = str(self.classifier.classes_[best_index])
        confidence = float(probabilities[best_index])
        similarities = self.retrieval_embeddings @ embedding[0]
        nearest = np.argsort(similarities)[::-1][:similar_count]
        similar = [
            {
                "message": str(self.retrieval_rows.iloc[index]["text"]),
                "intent": humanize_intent(str(self.retrieval_rows.iloc[index]["label_text"])),
                "similarity": float(similarities[index]),
            }
            for index in nearest
        ]
        top = [
            {
                "intent": str(self.classifier.classes_[index]),
                "intent_display": humanize_intent(str(self.classifier.classes_[index])),
                "confidence": float(probabilities[index]),
            }
            for index in ranked[:3]
        ]
        priority, priority_reason = priority_for_intent(intent)
        sla_hours, sla_reason = sla_for_priority(priority)
        elapsed = (perf_counter() - start) * 1000
        return Prediction(
            intent=intent,
            intent_display=humanize_intent(intent),
            team=team_for_intent(intent),
            priority=priority,
            priority_reason=priority_reason,
            sla_hours=sla_hours,
            sla_reason=sla_reason,
            confidence=confidence,
            decision="Automatically route" if confidence >= self.threshold else "Human review required",
            threshold=self.threshold,
            top_predictions=top,
            similar_tickets=similar,
            inference_ms=elapsed,
        )
