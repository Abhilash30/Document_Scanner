import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from models.svm_classifier import SVMClassifier
from features.hog_extractor import HOGExtractor
from dataset.character_dataset import CharacterDataset
import argparse
import joblib
import numpy as np
from PIL import Image

from src.pipeline.ocr_pipeline import OCRPipeline


def main():
    # parser = argparse.ArgumentParser(description="OCR Pipeline for Document Text Recognition")
    # parser.add_argument("--image", "-i", type=str, required=True, help="Path to document image")
    # parser.add_argument("--model", "-m", type=str, default=None, help="Path to trained model")
    # parser.add_argument("--feature", "-f", type=str, choices=["raw", "hog"], default=None, 
    #                     help="Feature type (auto-detected if not specified)")
    # args = parser.parse_args()
    
    try:
        dataset = CharacterDataset("/home/abhilash/Projects/doc_scanner/data/typedCSV.csv")
        HOG = HOGExtractor()
        X_train, Y_train = dataset.load_data()
        X_train = HOG.extract(X_train)
        svmc = SVMClassifier()
        svmc.train(X_train, Y_train)
        joblib.dump(svmc.model, "models/svm_model.joblib")
        # Load and predict
        #pipeline = OCRPipeline(model_path=args.model, feature_type=args.feature)
        #text, char_results = pipeline.predict_text(image_path=args.image)
        
        print("\n" + "="*60)
        print("OCR RESULTS")
        print("="*60)
        #print(text if text else "[No text detected]")
        print("="*60)
        #print(f"\nTotal characters detected: {len(char_results)}")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
