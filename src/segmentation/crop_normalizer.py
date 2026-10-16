import cv2
import numpy as np

class CropNormalizer:
    """
    Normalize character crops to match training data format (28x28).
    Training data: 28x28, text as dark pixels on light background, normalized to [0,1].
    """
    def normalize_crop(self, crop: np.ndarray, margin: int = 4) -> np.ndarray:
        if len(crop.shape) == 3:
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        else:
            gray = crop.copy()
        
        # Ensure text is dark (0) on light background (255) for consistent processing
        # If background is dark (0) and text is light (255), invert
        if np.mean(gray) < 127:  # mostly dark background
            gray = cv2.bitwise_not(gray)
        
        h, w = gray.shape
        if h == 0 or w == 0:
            return np.zeros((28, 28), dtype=np.float32)
        
        # Find text region to compute proper background
        # Use median of light pixels as background
        light_pixels = gray[gray > 128]
        if len(light_pixels) == 0:
            bg = 255
        else:
            bg = int(np.median(light_pixels))
        
        size = max(h, w) + 2 * margin
        canvas = np.full((size, size), bg, dtype=np.uint8)
        y0 = (size - h) // 2
        x0 = (size - w) // 2
        canvas[y0:y0 + h, x0:x0 + w] = gray
        
        # Resize to 28x28
        resized = cv2.resize(canvas, (28, 28), interpolation=cv2.INTER_AREA)
        
        # Normalize to [0,1] - in training, text is dark (0), background is light (1)
        # Our resized has text as dark (0) and bg as light (255); dividing by 255 gives correct format
        norm = resized.astype(np.float32) / 255.0
        return norm
