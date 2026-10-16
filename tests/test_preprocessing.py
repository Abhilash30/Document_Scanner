import sys
import os
import numpy as np

# Add src directory to python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.dataset.character_dataset import CharacterDataset
from src.preprocessing.image_processor import ImageProcessor


def test_image_processor():
    csv_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "typedCSV.csv"))
    dataset = CharacterDataset(csv_path)
    X, y = dataset.load_data()

    processor = ImageProcessor(target_shape=(28, 28), normalize=True)

    print("\n--- IMAGE PROCESSOR VERIFICATION ---")

    # 1. Single sample vector to 2D image test
    single_vec = X[0]
    single_img = processor.vector_to_image(single_vec)
    print(f"Single Vector Shape : {single_vec.shape} -> 2D Image Shape: {single_img.shape}")
    assert single_img.shape == (28, 28), "Single image shape should be (28, 28)"

    # 2. Convert 2D image back to 1D vector
    reconstructed_vec = processor.image_to_vector(single_img)
    assert reconstructed_vec.shape == (784,), "Reconstructed vector shape should be (784,)"
    assert np.array_equal(single_vec, reconstructed_vec), "Vector reconstruction failed!"

    # 3. Batch vector to batch 2D image test
    batch_vec = X[:100]
    batch_img = processor.vector_to_image(batch_vec)
    print(f"Batch Vector Shape  : {batch_vec.shape} -> Batch Image Shape: {batch_img.shape}")
    assert batch_img.shape == (100, 28, 28), "Batch image shape should be (100, 28, 28)"

    reconstructed_batch = processor.image_to_vector(batch_img)
    assert np.array_equal(batch_vec, reconstructed_batch), "Batch vector reconstruction failed!"

    # 4. Normalization test
    processed_X = processor.process(X[:1000])
    print(f"Raw Pixel Min/Max   : {X[:1000].min()} / {X[:1000].max()}")
    print(f"Processed Min/Max   : {processed_X.min():.4f} / {processed_X.max():.4f}")
    print(f"Processed Dtype     : {processed_X.dtype}")

    assert processed_X.min() >= 0.0 and processed_X.max() <= 1.0, "Normalized values must be in [0.0, 1.0]"
    assert processed_X.dtype == np.float32, "Processed array dtype must be float32"

    print("\nAll ImageProcessor checks PASSED successfully!")


if __name__ == "__main__":
    test_image_processor()
