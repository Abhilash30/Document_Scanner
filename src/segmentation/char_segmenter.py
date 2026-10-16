import cv2
import numpy as np

class CharSegmenter:
    def __init__(self, min_area: int = 5):
        self.min_area = min_area
    
    def segment_chars(self, line_img: np.ndarray) -> list:
        if len(line_img.shape) == 3:
            gray = cv2.cvtColor(line_img, cv2.COLOR_BGR2GRAY)
        else:
            gray = line_img.copy()
        
        if np.mean(gray) < 127:
            binary = cv2.bitwise_not(gray)
        else:
            binary = gray.copy()
        
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        boxes = []
        h, w = binary.shape
        for cnt in contours:
            area = cv2.contourArea(cnt)
            x, y, bw_box, bh_box = cv2.boundingRect(cnt)
            if area < self.min_area:
                continue
            boxes.append((x, y, x + bw_box, y + bh_box))
        if not boxes:
            return []
        boxes.sort(key=lambda b: b[0])
        merged = self._merge_dots(boxes, h)
        merged.sort(key=lambda b: b[0])
        final = []
        for box in merged:
            final.extend(self._split_wide_box(box, binary))
        final.sort(key=lambda b: b[0])
        return final if final else merged
    
    def _split_wide_box(self, box, binary):
        x1, y1, x2, y2 = box
        w = x2 - x1
        h_box = y2 - y1
        if w < h_box * 1.5:
            return [box]
        roi = binary[y1:y2, x1:x2]
        vproj = np.sum(roi, axis=0)
        splits = []
        in_char = False
        start = 0
        for i in range(len(vproj)):
            if vproj[i] > 0 and not in_char:
                in_char = True
                start = i
            elif vproj[i] == 0 and in_char:
                end = i
                if end - start > 2:
                    splits.append((x1 + start, x1 + end))
                in_char = False
                start = i
        if in_char and len(vproj) - start > 2:
            splits.append((x1 + start, x2))
        if len(splits) <= 1:
            return [box]
        result = []
        for sx1, sx2 in splits:
            result.append((sx1, y1, sx2, y2))
        return result
    
    def _merge_dots(self, boxes, line_h):
        if len(boxes) < 2:
            return boxes
        merged = []
        used = [False] * len(boxes)
        for i in range(len(boxes)):
            if used[i]:
                continue
            bx1, by1, bx2, by2 = boxes[i]
            bh = by2 - by1
            bw = bx2 - bx1
            is_dot = bw < 0.4 * line_h and bh < 0.4 * line_h
            if is_dot:
                attached = False
                for j in range(len(boxes)):
                    if used[j] or i == j:
                        continue
                    jx1, jy1, jx2, jy2 = boxes[j]
                    if abs(jx2 - bx1) < 12 and not (by2 < jy1 or by1 > jy2):
                        used[j] = used[i] = True
                        merged.append((min(bx1, jx1), min(by1, jy1), max(bx2, jx2), max(by2, jy2)))
                        attached = True
                        break
                    if abs(bx2 - jx1) < 12 and not (by2 < jy1 or by1 > jy2):
                        used[j] = used[i] = True
                        merged.append((min(bx1, jx1), min(by1, jy1), max(bx2, jx2), max(by2, jy2)))
                        attached = True
                        break
                if attached:
                    continue
            merged.append(boxes[i])
            used[i] = True
        return merged
