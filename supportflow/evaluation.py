from __future__ import annotations

from collections import Counter

import numpy as np
from sklearn.metrics import accuracy_score, f1_score

from .confidence import expected_calibration_error


def classification_metrics(
    labels: np.ndarray,
    probabilities: np.ndarray,
    classes: np.ndarray,
    threshold: float,
) -> dict[str, float | int]:
    predicted_indexes = probabilities.argmax(axis=1)
    predicted_labels = classes[predicted_indexes]
    confidence = probabilities.max(axis=1)
    accepted = confidence >= threshold
    correct = predicted_labels == labels
    top_k = min(3, probabilities.shape[1])
    class_indexes = {label: index for index, label in enumerate(classes)}
    true_indexes = np.array([class_indexes[label] for label in labels])
    top_indexes = np.argpartition(probabilities, -top_k, axis=1)[:, -top_k:]
    top_k_accuracy = float(np.any(top_indexes == true_indexes[:, None], axis=1).mean())
    return {
        "accuracy": float(accuracy_score(labels, predicted_labels)),
        "macro_f1": float(f1_score(labels, predicted_labels, average="macro")),
        "top_3_accuracy": top_k_accuracy,
        "auto_route_threshold": float(threshold),
        "auto_route_coverage": float(accepted.mean()),
        "auto_route_accuracy": float(correct[accepted].mean()) if accepted.any() else 0.0,
        "human_review_rate": float((~accepted).mean()),
        "auto_route_examples": int(accepted.sum()),
        "human_review_examples": int((~accepted).sum()),
        "incorrect_auto_routes": int((accepted & ~correct).sum()),
        "expected_calibration_error": float(expected_calibration_error(labels, probabilities, classes=classes)),
        "test_examples": int(len(labels)),
    }


def most_common_confusions(
    labels: np.ndarray,
    predicted: np.ndarray,
    limit: int = 15,
) -> list[dict[str, str | int]]:
    pairs = Counter((str(actual), str(guess)) for actual, guess in zip(labels, predicted, strict=True) if actual != guess)
    return [
        {"actual_intent": actual, "predicted_intent": guess, "count": count}
        for (actual, guess), count in pairs.most_common(limit)
    ]
