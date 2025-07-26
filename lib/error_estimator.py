# error_estimator.py

import numpy as np


def root_mean_square_error(labels: np.ndarray, predictions: np.ndarray) -> float:
    n = len(labels)
    differences = np.subtract(labels, predictions)
    return np.sqrt(1.0/n * np.dot(differences, differences))