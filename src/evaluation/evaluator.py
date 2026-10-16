from typing import Any
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score


class Evaluator:
    """
    Evaluator computes classification metrics to assess model performance.

    Responsibilities:
    - Calculate overall accuracy.
    - Generate confusion matrix (62x62 for character classes).
    - Return a structured evaluation report dictionary.

    Keeping evaluation separate from training/prediction maintains clear
    separation of concerns and enables easy extension with additional metrics.
    """

    def calculate_accuracy(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Computes classification accuracy.

        Args:
            y_true (np.ndarray): Ground truth labels (N,).
            y_pred (np.ndarray): Predicted labels (N,).

        Returns:
            float: Accuracy score in [0.0, 1.0].
        """
        return float(accuracy_score(y_true, y_pred))

    def get_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray, labels: np.ndarray | None = None) -> np.ndarray:
        """
        Computes the confusion matrix.

        Args:
            y_true (np.ndarray): Ground truth labels (N,).
            y_pred (np.ndarray): Predicted labels (N,).
            labels (np.ndarray | None): Optional list of label values to include (useful for ordering).

        Returns:
            np.ndarray: Confusion matrix of shape (C, C) where C = number of classes.
        """
        if labels is None:
            # Default: include all unique labels present
            all_labels = np.unique(np.concatenate([y_true, y_pred]))
            labels = np.sort(all_labels)
        return confusion_matrix(y_true, y_pred, labels=labels)

    def evaluate(self, y_true: np.ndarray, y_pred: np.ndarray, labels: np.ndarray | None = None) -> dict[str, Any]:
        """
        Performs full evaluation and returns a summary dictionary.

        Args:
            y_true (np.ndarray): Ground truth labels (N,).
            y_pred (np.ndarray): Predicted labels (N,).
            labels (np.ndarray | None): Optional class label ordering.

        Returns:
            dict: Dictionary containing accuracy and confusion matrix.
        """
        accuracy = self.calculate_accuracy(y_true, y_pred)
        cm = self.get_confusion_matrix(y_true, y_pred, labels=labels)
        return {
            "accuracy": accuracy,
            "confusion_matrix": cm,
        }
