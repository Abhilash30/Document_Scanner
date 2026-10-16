import os
import pandas as pd
import numpy as np


class CharacterDataset:
    """
    CharacterDataset is responsible for loading the raw CSV dataset,
    separating pixel feature vectors (X) from class labels (y), and
    providing utility methods for data querying and character mappings.

    Class Label Mapping (62 total classes):
      Labels 0  .. 9  -> Digits '0' .. '9'
      Labels 10 .. 35 -> Uppercase 'A' .. 'Z'
      Labels 36 .. 61 -> Lowercase 'a' .. 'z'
    """

    def __init__(self, csv_file_path: str):
        if not os.path.exists(csv_file_path):
            raise FileNotFoundError(f"CSV file not found at path: {csv_file_path}")

        self.csv_file_path = csv_file_path
        self._X: np.ndarray | None = None
        self._y: np.ndarray | None = None

    def load_data(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Loads the CSV file into memory and extracts features (X) and labels (y).

        Returns:
            X (np.ndarray): Pixel intensities matrix of shape (N, 784), dtype uint8.
            y (np.ndarray): Target class labels vector of shape (N,), dtype int64.
        """
        df = pd.read_csv(self.csv_file_path)

        # Extract 'label' column as y
        self._y = df["label"].values.astype(np.int64)

        # Extract 'pixel 0' through 'pixel 783' as X
        pixel_cols = [f"pixel {i}" for i in range(784)]
        self._X = df[pixel_cols].values.astype(np.uint8)

        return self._X, self._y

    def get_num_samples(self) -> int:
        """Returns the total number of character samples (N)."""
        if self._X is None:
            raise RuntimeError("Data not loaded yet. Call load_data() first.")
        return len(self._X)

    def get_num_features(self) -> int:
        """Returns the number of pixel features per sample (expected: 784)."""
        if self._X is None:
            raise RuntimeError("Data not loaded yet. Call load_data() first.")
        return self._X.shape[1]

    def get_num_classes(self) -> int:
        """Returns the total number of unique character classes (expected: 62)."""
        if self._y is None:
            raise RuntimeError("Data not loaded yet. Call load_data() first.")
        return len(np.unique(self._y))

    def get_label_distribution(self) -> dict[int, int]:
        """Returns a dictionary mapping class label index to its sample count."""
        if self._y is None:
            raise RuntimeError("Data not loaded yet. Call load_data() first.")
        unique, counts = np.unique(self._y, return_counts=True)
        return dict(zip(unique, counts))

    @staticmethod
    def label_to_char(label: int) -> str:
        """
        Converts integer class label (0..61) to corresponding character.

        0-9   -> '0'-'9'
        10-35 -> 'A'-'Z'
        36-61 -> 'a'-'z'
        """
        if 0 <= label <= 9:
            return chr(ord("0") + label)
        elif 10 <= label <= 35:
            return chr(ord("A") + (label - 10))
        elif 36 <= label <= 61:
            return chr(ord("a") + (label - 36))
        else:
            raise ValueError(f"Label {label} is out of valid range [0, 61].")

    @staticmethod
    def char_to_label(char: str) -> int:
        """
        Converts character to corresponding integer class label (0..61).
        """
        if len(char) != 1:
            raise ValueError("Expected a single character string.")

        if "0" <= char <= "9":
            return ord(char) - ord("0")
        elif "A" <= char <= "Z":
            return 10 + (ord(char) - ord("A"))
        elif "a" <= char <= "z":
            return 36 + (ord(char) - ord("a"))
        else:
            raise ValueError(f"Character '{char}' is not in [0-9, A-Z, a-z].")
