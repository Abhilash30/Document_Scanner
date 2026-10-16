import cv2
import numpy as np
from PIL import Image
from .document_preprocessor import DocumentPreprocessor
from .line_segmenter import LineSegmenter
from .char_segmenter import CharSegmenter
from .crop_normalizer import CropNormalizer

class DocumentSegmenter:
    """
    Orchestrates full document segmentation pipeline.
    """
    def __init__(self):
        self.preprocessor = DocumentPreprocessor()
        self.line_segmenter = LineSegmenter()
        self.char_segmenter = CharSegmenter()
        self.normalizer = CropNormalizer()
    
    def segment(self, image_path: str = None, image_array: np.ndarray = None,
                pil_image: Image.Image = None) -> list:
        """
        Segment document into normalized character crops with positions.
        Returns: list of dicts with 'crop' (28x28 normalized), 'bbox', 'line_idx'
        """
        # Load image
        if image_array is not None:
            img = image_array
        elif pil_image is not None:
            img = np.array(pil_image)
            if len(img.shape) == 3:
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        elif image_path is not None:
            img = cv2.imread(image_path)
            if img is None:
                raise FileNotFoundError(f"Cannot read {image_path}")
        else:
            raise ValueError("Need image_path, image_array, or pil_image")
        
        # Preprocess
        processed = self.preprocessor.preprocess(img)
        binary = self.preprocessor.invert_if_needed(processed)
        
        # Segment lines
        line_boxes = self.line_segmenter.segment_lines(binary)
        
        results = []
        for line_idx, (y1,y2,x1,x2) in enumerate(line_boxes):
            line_img = binary[y1:y2, x1:x2]
            char_boxes = self.char_segmenter.segment_chars(line_img)
            for cb in char_boxes:
                cx1, cy1, cx2, cy2 = cb
                # Map to original coords
                ocx1, ocy1, ocx2, ocy2 = x1+cx1, y1+cy1, x1+cx2, y1+cy2
                # Crop from original processed/binary
                char_crop = binary[ocy1:ocy2, ocx1:ocx2]
                # Normalize
                norm_crop = self.normalizer.normalize_crop(char_crop)
                results.append({
                    'crop': norm_crop,
                    'bbox': (ocx1, ocy1, ocx2, ocy2),
                    'line_idx': line_idx,
                    'char_box': cb
                })
        return results
