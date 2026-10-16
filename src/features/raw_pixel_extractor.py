import numpy as np
from .feature_extractor import FeatureExtractor


class RawPixelExtractor(FeatureExtractor):
    """
    RawPixelExtractor: Uses raw pixel values as features.

    For the first experiment, the 28x28 image is flattened to a 784-dimensional vector.
    The SVM learns decision boundaries in the raw pixel space.

    This extractor is intentionally simple to make the data flow explicit:
    (N, 28, 28) or (N, 784) -> (N, 784) feature matrix.
    """

    def extract(self, X: np.ndarray) -> np.ndarray:
        """
        Extracts raw pixel features.

        Args:
            X (np.ndarray): Input array. Can be shape (N, 784) or (N, 28, 28).
                            If 2D images are passed, they are flattened to 1D vectors.

        Returns:
            np.ndarray: Feature matrix of shape (N, 784), dtype float32.
        """
        if X.ndim == 2 and X.shape[1] == 784:
            return X.astype(np.float32)
        elif X.ndim == 3 and X.shape[1] == 28 and X.shape[2] == 28:
            # Flatten 2D images to 1D feature vectors
            return X.reshape((X.shape[0], -1)).astype(np.float32)
        else:
            raise ValueError(
                f"Unsupported input shape {X.shape}. Expected (N, 784) or (N, 28, 28)."
            )
