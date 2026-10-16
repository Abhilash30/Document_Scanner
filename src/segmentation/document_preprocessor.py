import cv2
import numpy as np

class DocumentPreprocessor:
    """
    Preprocess document images for OCR segmentation.
    Handles grayscale conversion, denoising, binarization, morphology.
    """
    def __init__(self, blur_kernel: tuple = (3,3), morph_kernel: tuple = (3,3)):
        self.blur_kernel = blur_kernel
        self.morph_kernel = np.ones(morph_kernel, np.uint8)
    
    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """
        Full preprocessing pipeline.
        Input: RGB or BGR image (H,W,3)
        Output: Binary image (H,W) - text as white, background as black or vice versa
        """
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Denoise
        blurred = cv2.GaussianBlur(gray, self.blur_kernel, 0)
        
        # Binarization using Otsu's method
        _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Apply morphology to clean up
        cleaned = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, self.morph_kernel)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, self.morph_kernel)
        
        return cleaned
    
    def invert_if_needed(self, binary: np.ndarray) -> np.ndarray:
        """Ensure text is foreground (255) on background (0)."""
        # If more background pixels, invert
        if np.mean(binary) > 127:
            return cv2.bitwise_not(binary)
        return binary
