import numpy as np

from supportflow.confidence import choose_auto_route_threshold, expected_calibration_error


def test_threshold_reaches_target_accuracy() -> None:
    labels = np.array([0, 1, 0, 1])
    probabilities = np.array([[0.9, 0.1], [0.2, 0.8], [0.4, 0.6], [0.55, 0.45]])
    result = choose_auto_route_threshold(labels, probabilities, target_accuracy=1.0, minimum_coverage=0.25)
    accepted = probabilities.max(axis=1) >= result["threshold"]
    predicted = probabilities.argmax(axis=1)
    assert (predicted[accepted] == labels[accepted]).mean() == 1.0


def test_calibration_error_is_zero_for_perfect_certainty() -> None:
    labels = np.array([0, 1])
    probabilities = np.array([[1.0, 0.0], [0.0, 1.0]])
    assert expected_calibration_error(labels, probabilities) == 0.0


def test_string_labels_align_with_classifier_classes() -> None:
    labels = np.array(["card", "transfer"])
    classes = np.array(["card", "transfer"])
    probabilities = np.array([[0.9, 0.1], [0.1, 0.9]])
    result = choose_auto_route_threshold(labels, probabilities, classes=classes, target_accuracy=1.0)
    assert result["validation_auto_route_accuracy"] == 1.0
    assert expected_calibration_error(labels, probabilities, classes=classes) < 0.11
