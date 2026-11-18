import numpy as np

from supportflow.evaluation import classification_metrics, most_common_confusions


def test_classification_metrics_counts_coverage() -> None:
    labels = np.array(["a", "b", "a"])
    classes = np.array(["a", "b"])
    probabilities = np.array([[0.9, 0.1], [0.2, 0.8], [0.4, 0.6]])
    metrics = classification_metrics(labels, probabilities, classes, threshold=0.75)
    assert metrics["accuracy"] == 2 / 3
    assert metrics["auto_route_coverage"] == 2 / 3
    assert metrics["auto_route_accuracy"] == 1.0


def test_confusion_pairs_exclude_correct_rows() -> None:
    rows = most_common_confusions(np.array(["a", "a", "b"]), np.array(["a", "b", "a"]))
    assert sum(int(row["count"]) for row in rows) == 2
