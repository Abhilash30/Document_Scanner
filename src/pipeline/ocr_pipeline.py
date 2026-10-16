import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import numpy as np
from typing import List, Tuple
from src.features.raw_pixel_extractor import RawPixelExtractor
from src.features.hog_extractor import HOGExtractor
from src.models.svm_classifier import SVMClassifier
from src.segmentation.document_segmenter import DocumentSegmenter
from src.segmentation.text_reconstructor import TextReconstructor
import joblib

class OCRPipeline:
    """
    End-to-end OCR pipeline.
    Combines segmentation, feature extraction, classification, reconstruction.
    """
    def __init__(self, model_path: str = None, feature_type: str = None):
        self.model_path = model_path or os.path.join(
            os.path.dirname(__file__), '..', '..', 'models', 'svm_ocr.joblib'
        )
        self.feature_type = feature_type  # if None, infer from model
        self.segmenter = DocumentSegmenter()
        self.reconstructor = TextReconstructor()
        self.classifier = None
        self.extractor = None
        self._load_components()
    
    def _load_components(self):
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model not found: {self.model_path}")
        self.classifier = joblib.load(self.model_path)
        n_feat = getattr(self.classifier._model, 'n_features_in_', None)
        
        # Infer feature type from model dims
        if self.feature_type is None:
            if n_feat == 784:
                self.feature_type = 'raw'
            elif n_feat == 2304:
                self.feature_type = 'hog'
            elif n_feat == 324:  # our custom HOG
                self.feature_type = 'hog_custom'
            else:
                # Default to raw if matches
                self.feature_type = 'raw'
        
        if self.feature_type == 'raw':
            self.extractor = RawPixelExtractor()
        elif self.feature_type == 'hog':
            # Match the saved model's HOG params
            self.extractor = HOGExtractor(pixels_per_cell=(3,3), cells_per_block=(2,2), orientations=9)
        elif self.feature_type == 'hog_custom':
            self.extractor = HOGExtractor(pixels_per_cell=(7,7), cells_per_block=(2,2), orientations=9)
        else:
            self.extractor = RawPixelExtractor()
    
    def predict_text(self, image_path: str = None, pil_image=None) -> Tuple[str, List]:
        """Predict text from document image."""
        segments = self.segmenter.segment(image_path=image_path, pil_image=pil_image)
        if not segments:
            return "", []
        crops = np.array([s['crop'] for s in segments])
        features = self.extractor.extract(crops)
        preds = self.classifier.predict(features)
        char_results = []
        for i, s in enumerate(segments):
            pred_label = int(preds[i])
            char = self._label_to_char(pred_label)
            char_results.append((char, s['bbox'][0], s['bbox'][1], s['bbox'][2], s['bbox'][3], 
                               s['line_idx']))
        text_lines = self._reconstruct_by_lines(char_results)
        return '\n'.join(text_lines), char_results
    
    def _label_to_char(self, label: int) -> str:
        if label < 0 or label > 61:
            return ''
        if 0 <= label <= 9:
            return chr(ord('0') + label)
        elif 10 <= label <= 35:
            return chr(ord('A') + (label - 10))
        elif 36 <= label <= 61:
            return chr(ord('a') + (label - 36))
        return ''
    
    def _reconstruct_by_lines(self, char_results: List) -> List[str]:
        lines = {}
        for cr in char_results:
            char, x1,y1,x2,y2, line_idx = cr
            if line_idx not in lines:
                lines[line_idx] = []
            lines[line_idx].append((char, x1,y1,x2,y2))
        result_lines = []
        for line_idx in sorted(lines.keys()):
            line_chars = sorted(lines[line_idx], key=lambda x: x[1])
            text = self.reconstructor.add_spaces_for_words(line_chars)
            result_lines.append(text)
        return result_lines
