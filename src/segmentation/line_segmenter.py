import cv2
import numpy as np

class LineSegmenter:
    """
    Segment document into text lines using horizontal projection profiles.
    """
    def __init__(self, min_line_height: int = 10):
        self.min_line_height = min_line_height
    
    def segment_lines(self, binary: np.ndarray) -> list:
        """
        Extract line bounding boxes from binary image.
        binary: Binary image where text is foreground (255) and background is (0)
        Returns: list of (y1, y2, x1, x2)
        """
        # binary from preprocessor with invert_if_needed gives text as white (255) when mean < 127? 
        # Wait - in our case, text regions are white in binary_inv; we need text as white for projection
        # If text is black in binary, invert it
        if np.mean(binary) > 127:  # mostly white background, text black
            binary_text = cv2.bitwise_not(binary)
        else:  # text is white, background black
            binary_text = binary.copy()
        
        h_proj = np.sum(binary_text, axis=1) / 255.0
        lines = []
        in_line = False
        y1 = 0
        for i, val in enumerate(h_proj):
            if val > 5 and not in_line:
                in_line = True
                y1 = i
            elif val <= 5 and in_line:
                y2 = i
                if y2 - y1 >= self.min_line_height:
                    lines.append((y1, y2))
                in_line = False
        if in_line and (len(h_proj) - y1) >= self.min_line_height:
            lines.append((y1, len(h_proj)))
        
        line_boxes = []
        for y1, y2 in lines:
            line_boxes.append((y1, y2, 0, binary.shape[1]))
        return line_boxes
