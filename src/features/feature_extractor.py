from abc import ABC, abstractmethod
import numpy as np


class FeatureExtractor(ABC):
    """
    Abstract base class for all feature extractors.

    Concept:
        A feature extractor transforms input image data (e.g., (N, 784) normalized
        pixel arrays or (N, 28, 28) 2D images) into a numerical feature representation
        (feature matrix X_feat of shape (N, D)) suitable for machine learning classifiers.

    Subclasses must implement the `extract()` method.
    """

    @abstractmethod
    def extract(self, X: np.ndarray) -> np.ndarray:
        """
        Extracts features from input data X.

        Args:
            X (np.ndarray): Input array (commonly preprocessed pixel data).

        Returns:
            np.ndarray: Feature matrix of shape (N, D) where D is feature dimensionality.
        """
        pass
