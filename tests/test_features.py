import sys
import os
import numpy as np

# Add src directory to python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.dataset.character_dataset import CharacterDataset
from src.preprocessing.image_processor import ImageProcessor
from src.features.raw_pixel_extractor import RawPixelExtractor


def test_raw_pixel_extractor():
    csv_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "typedCSV.csv"))
    dataset = CharacterDataset(csv_path)
    X, y = dataset.load_data()

    processor = ImageProcessor(target_shape=(28, 28), normalize=True)
    extractor = RawPixelExtractor()

    print("\n--- RAW PIXEL EXTRACTOR VERIFICATION ---")

    # Test with normalized 2D batch
    processed = processor.process(X[:50])
    images_2d = processor.vector_to_image(processed)
    features = extractor.extract(images_2d)

    print(f"Input Shape (2D images) : {images_2d.shape}")
    print(f"Output Shape (features) : {features.shape}")
    print(f"Feature Dtype           : {features.dtype}")
    assert features.shape == (50, 784), "Features must be (N, 784)"
    assert features.dtype == np.float32, "Features dtype must be float32"

    # Test with 1D normalized batch
    features_1d = extractor.extract(processed)
    print(f"Input Shape (1D vecs)   : {processed.shape}")
    print(f"Output Shape (features) : {features_1d.shape}")
    assert features_1d.shape == (50, 784), "Features from 1D must be (N, 784)"
    assert np.allclose(features, features_1d), "1D vs 2D extraction mismatch"

    print("\nAll RawPixelExtractor checks PASSED successfully!")


if __name__ == "__main__":
    test_raw_pixel_extractor()
