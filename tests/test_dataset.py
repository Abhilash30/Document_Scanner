import sys
import os

# Add src directory to python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.dataset.character_dataset import CharacterDataset


def test_character_dataset():
    csv_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "typedCSV.csv"))
    print(f"Loading dataset from: {csv_path}")

    dataset = CharacterDataset(csv_path)
    X, y = dataset.load_data()

    num_samples = dataset.get_num_samples()
    num_features = dataset.get_num_features()
    num_classes = dataset.get_num_classes()
    dist = dataset.get_label_distribution()

    print("\n--- DATASET VERIFICATION ---")
    print(f"Total Samples (N) : {num_samples}")
    print(f"Features (D)      : {num_features}")
    print(f"Unique Classes    : {num_classes}")
    print(f"X Data Shape      : {X.shape}")
    print(f"y Target Shape    : {y.shape}")
    print(f"Pixel Min / Max   : {X.min()} / {X.max()}")

    # Assertions
    assert num_samples == len(X) == len(y), "Sample count mismatch!"
    assert num_features == 784, f"Expected 784 features, got {num_features}"
    assert num_classes == 62, f"Expected 62 classes, got {num_classes}"
    assert X.shape[1] == 784, "X feature dimension must be 784"

    print("\n--- CHARACTER MAPPING VERIFICATION ---")
    # Test digits 0-9
    for i in range(10):
        char = CharacterDataset.label_to_char(i)
        rev = CharacterDataset.char_to_label(char)
        assert rev == i
        print(f"Label {i:2d} -> Char '{char}' (count: {dist.get(i, 0)})")

    # Test uppercase A-Z (labels 10-35)
    print("...")
    for i in [10, 35]:
        char = CharacterDataset.label_to_char(i)
        rev = CharacterDataset.char_to_label(char)
        assert rev == i
        print(f"Label {i:2d} -> Char '{char}' (count: {dist.get(i, 0)})")

    # Test lowercase a-z (labels 36-61)
    print("...")
    for i in [36, 61]:
        char = CharacterDataset.label_to_char(i)
        rev = CharacterDataset.char_to_label(char)
        assert rev == i
        print(f"Label {i:2d} -> Char '{char}' (count: {dist.get(i, 0)})")

    print("\nAll dataset checks PASSED successfully!")


if __name__ == "__main__":
    test_character_dataset()
