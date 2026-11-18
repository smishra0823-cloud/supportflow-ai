from __future__ import annotations

import numpy as np


def choose_auto_route_threshold(
    labels: np.ndarray,
    probabilities: np.ndarray,
    classes: np.ndarray | None = None,
    target_accuracy: float = 0.90,
    minimum_coverage: float = 0.20,
) -> dict[str, float]:
    if probabilities.ndim != 2 or len(labels) != len(probabilities):
        raise ValueError("Labels and probability rows must align.")
    predicted_indexes = probabilities.argmax(axis=1)
    if classes is None:
        predicted = predicted_indexes
    else:
        predicted = np.asarray(classes)[predicted_indexes]
    confidence = probabilities.max(axis=1)
    correct = predicted == labels
    candidates = np.unique(np.r_[0.0, confidence, 1.0])
    viable: list[tuple[float, float, float]] = []
    fallbacks: list[tuple[float, float, float]] = []
    for threshold in candidates:
        accepted = confidence >= threshold
        coverage = float(accepted.mean())
        if not accepted.any():
            continue
        accepted_accuracy = float(correct[accepted].mean())
        item = (coverage, accepted_accuracy, float(threshold))
        fallbacks.append(item)
        if coverage >= minimum_coverage and accepted_accuracy >= target_accuracy:
            viable.append(item)
    if viable:
        coverage, accepted_accuracy, threshold = max(viable, key=lambda item: (item[0], item[1], -item[2]))
    else:
        coverage, accepted_accuracy, threshold = max(fallbacks, key=lambda item: (item[1], item[0], -item[2]))
    return {
        "threshold": threshold,
        "validation_coverage": coverage,
        "validation_auto_route_accuracy": accepted_accuracy,
        "target_accuracy": target_accuracy,
    }


def expected_calibration_error(
    labels: np.ndarray,
    probabilities: np.ndarray,
    classes: np.ndarray | None = None,
    bins: int = 10,
) -> float:
    predicted_indexes = probabilities.argmax(axis=1)
    predicted = predicted_indexes if classes is None else np.asarray(classes)[predicted_indexes]
    confidence = probabilities.max(axis=1)
    correct = (predicted == labels).astype(float)
    edges = np.linspace(0.0, 1.0, bins + 1)
    error = 0.0
    for lower, upper in zip(edges[:-1], edges[1:], strict=True):
        mask = (confidence > lower) & (confidence <= upper)
        if mask.any():
            error += float(mask.mean()) * abs(float(correct[mask].mean()) - float(confidence[mask].mean()))
    return error
