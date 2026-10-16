import numpy as np
from .feature_extractor import FeatureExtractor


class HOGExtractor(FeatureExtractor):
    """
    Histogram of Oriented Gradients (HOG) feature extractor.

    Computes HOG descriptors from 28x28 character images.
    Transforms spatial images into a shape/edge-based feature representation
    suitable for classical ML classifiers like SVM.

    Parameters:
        orientations (int): Number of orientation bins (unsigned, 0-180 degrees). Default 9.
        pixels_per_cell (tuple): Size of each cell in pixels (height, width). Default (7,7).
        cells_per_block (tuple): Number of cells per block (height, width). Default (2,2).
        block_norm (str): Normalization method. Supported: 'L2' (simple) and 'L2-Hys' (clipped+renorm).
                          Default 'L2-Hys'.
        transform_sqrt (bool): If True, apply sqrt on input (gamma correction). Default False.
        feature_vector (bool): If True, return flattened feature vector. Default True.
    """

    def __init__(
        self,
        orientations: int = 9,
        pixels_per_cell: tuple[int, int] = (3, 3),
        cells_per_block: tuple[int, int] = (2, 2),
        block_norm: str = "L2-Hys",
        transform_sqrt: bool = False,
        feature_vector: bool = True,
    ):
        self.orientations = orientations
        self.pixels_per_cell = pixels_per_cell
        self.cells_per_block = cells_per_block
        self.block_norm = block_norm
        self.transform_sqrt = transform_sqrt
        self.feature_vector = feature_vector
        self._feature_dim = None

    def extract(self, X: np.ndarray) -> np.ndarray:
        """
        Extract HOG features from input images.

        Args:
            X (np.ndarray): Input array. Can be (N, 784) or (N, 28, 28).
                            Values expected in [0, 1] range (float32) from preprocessing.

        Returns:
            np.ndarray: HOG feature matrix of shape (N, D), dtype float32.
        """
        # Convert to (N, H, W)
        if X.ndim == 2:
            if X.shape[1] == 784:
                images = X.reshape(-1, 28, 28)
            else:
                raise ValueError(f"Unsupported shape {X.shape}; expected (N,784) or (N,28,28)")
        elif X.ndim == 3:
            if X.shape[1] == 28 and X.shape[2] == 28:
                images = X
            else:
                raise ValueError(f"Unsupported shape {X.shape}; expected (N,28,28)")
        else:
            raise ValueError(f"Unsupported input dimension {X.ndim}")

        features = []
        for img in images:
            feat = self._hog_single(img.astype(np.float32))
            features.append(feat)

        result = np.array(features, dtype=np.float32)
        self._feature_dim = result.shape[1] if result.ndim == 2 else 0
        return result

    def _hog_single(self, img: np.ndarray) -> np.ndarray:
        """Compute HOG descriptor for a single 2D image."""
        if img.ndim != 2:
            raise ValueError("Expected 2D image")

        if self.transform_sqrt:
            img = np.sqrt(np.clip(img, 0.0, 1.0))

        h, w = img.shape
        cx, cy = self.pixels_per_cell
        bx, by = self.cells_per_block

        # Compute gradients using central differences (simple, clear)
        grad_x = np.zeros_like(img)
        grad_y = np.zeros_like(img)

        # Horizontal gradient
        grad_x[:, 1:-1] = (img[:, 2:] - img[:, :-2]) / 2.0
        grad_x[:, 0] = img[:, 1] - img[:, 0]
        grad_x[:, -1] = img[:, -1] - img[:, -2]

        # Vertical gradient
        grad_y[1:-1, :] = (img[2:, :] - img[:-2, :]) / 2.0
        grad_y[0, :] = img[1, :] - img[0, :]
        grad_y[-1, :] = img[-2, :] - img[-1, :]

        # Magnitude and orientation (unsigned 0-180 degrees)
        magnitude = np.sqrt(grad_x**2 + grad_y**2)
        orientation = np.arctan2(grad_y, grad_x)  # range [-pi, pi]

        # Convert to unsigned orientations [0, pi]
        orientation = orientation % np.pi  # 0 <= orient < pi
        angle_step = np.pi / self.orientations

        # Number of cells
        n_cells_x = h // cx
        n_cells_y = w // cy
        if n_cells_x < 1 or n_cells_y < 1:
            raise ValueError(
                f"Image too small for pixels_per_cell={self.pixels_per_cell}: got {img.shape}"
            )

        # Build cell histograms: (n_cells_x, n_cells_y, orientations)
        cell_hist = np.zeros((n_cells_x, n_cells_y, self.orientations), dtype=np.float32)

        for i in range(n_cells_x):
            for j in range(n_cells_y):
                cell_mag = magnitude[i * cx : (i + 1) * cx, j * cy : (j + 1) * cy]
                cell_orient = orientation[i * cx : (i + 1) * cx, j * cy : (j + 1) * cy]
                hist = self._compute_cell_hist(cell_mag, cell_orient, angle_step)
                cell_hist[i, j] = hist

        # Compute blocks with overlap: stride = 1 cell
        block_stride = 1
        n_blocks_x = (n_cells_x - bx) // block_stride + 1
        n_blocks_y = (n_cells_y - by) // block_stride + 1

        blocks = []
        for i in range(n_blocks_x):
            for j in range(n_blocks_y):
                block = cell_hist[i : i + bx, j : j + by].reshape(-1)
                block = self._normalize_block(block)
                blocks.append(block)

        hog_vec = np.concatenate(blocks) if blocks else np.array([])
        return hog_vec.astype(np.float32)

    def _compute_cell_hist(self, mag: np.ndarray, orient: np.ndarray, angle_step: float) -> np.ndarray:
        """Compute orientation histogram for a single cell."""
        hist = np.zeros(self.orientations, dtype=np.float32)
        for mval, oval in zip(mag.ravel(), orient.ravel()):
            if mval < 1e-10:
                continue
            # Find bin (unsigned)
            bin_idx = int(oval / angle_step) % self.orientations
            hist[bin_idx] += mval
        return hist

    def _normalize_block(self, block_vec: np.ndarray) -> np.ndarray:
        """Normalize block vector."""
        eps = 1e-12
        norm = np.sqrt(np.sum(block_vec**2) + eps)
        if norm < eps:
            return block_vec.copy()
        block_vec = block_vec / norm

        if self.block_norm == "L2-Hys":
            block_vec = np.clip(block_vec, 0.0, 0.2)
            norm2 = np.sqrt(np.sum(block_vec**2) + eps)
            if norm2 > eps:
                block_vec = block_vec / norm2
        return block_vec.astype(np.float32)

    @property
    def feature_dim(self) -> int | None:
        """Return feature dimension if known."""
        return self._feature_dim
