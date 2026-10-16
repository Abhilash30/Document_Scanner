import numpy as np


class ImageProcessor:
    """
    ImageProcessor is responsible for image data transformation and normalization.

    Key functions:
    1. Reshaping 1D feature vectors (784,) into 2D image matrices (28, 28) and vice versa.
    2. Normalizing pixel intensity values from [0, 255] integer range to [0.0, 1.0] float range.

    This keeps spatial image operations and intensity scaling cleanly separated from dataset loading
    and feature extraction models.
    """

    def __init__(self, target_shape: tuple[int, int] = (28, 28), normalize: bool = True):
        self.target_shape = target_shape  # (height, width) e.g., (28, 28)
        self.expected_vector_len = target_shape[0] * target_shape[1]  # 784
        self.should_normalize = normalize

    def vector_to_image(self, vector: np.ndarray) -> np.ndarray:
        """
        Converts a 1D pixel array of shape (784,) or batch of shape (N, 784)
        into 2D image array of shape (28, 28) or batch (N, 28, 28).

        Args:
            vector (np.ndarray): Array with last dimension equal to 784.

        Returns:
            np.ndarray: Reshaped image array.
        """
        if vector.shape[-1] != self.expected_vector_len:
            raise ValueError(
                f"Expected vector length {self.expected_vector_len}, but got shape {vector.shape}"
            )

        if vector.ndim == 1:
            return vector.reshape(self.target_shape)
        elif vector.ndim == 2:
            return vector.reshape((-1, self.target_shape[0], self.target_shape[1]))
        else:
            raise ValueError(f"Unsupported array dimension: {vector.ndim}. Expected 1D or 2D array.")

    def image_to_vector(self, image: np.ndarray) -> np.ndarray:
        """
        Flattens a 2D image of shape (28, 28) or batch (N, 28, 28)
        back into 1D vector of shape (784,) or batch (N, 784).

        Args:
            image (np.ndarray): Image array of shape (28, 28) or (N, 28, 28).

        Returns:
            np.ndarray: Flattened 1D vector or batch of vectors.
        """
        if image.ndim == 2 and image.shape == self.target_shape:
            return image.reshape(-1)
        elif image.ndim == 3 and image.shape[1:] == self.target_shape:
            return image.reshape((image.shape[0], -1))
        else:
            raise ValueError(
                f"Invalid image shape {image.shape}. Expected shape {self.target_shape} or (N, {self.target_shape[0]}, {self.target_shape[1]})."
            )

    def normalize_pixels(self, X: np.ndarray) -> np.ndarray:
        """
        Normalizes pixel values from range [0, 255] to range [0.0, 1.0].

        Args:
            X (np.ndarray): Array of pixel values (uint8 or float).

        Returns:
            np.ndarray: Float32 array normalized to [0.0, 1.0].
        """
        X_float = X.astype(np.float32)
        if X_float.max() > 1.0:
            return X_float / 255.0
        return X_float

    def process(self, X: np.ndarray) -> np.ndarray:
        """
        Applies standard preprocessing pipeline to input dataset X.

        Args:
            X (np.ndarray): Raw pixel data matrix of shape (N, 784) or single sample (784,).

        Returns:
            np.ndarray: Processed pixel matrix ready for feature extraction/classification.
        """
        if self.should_normalize:
            return self.normalize_pixels(X)
        return X.astype(np.float32)
