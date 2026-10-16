class TextReconstructor:
    """
    Reconstruct text from character predictions in order.
    """
    def reconstruct(self, chars_with_pos: list) -> str:
        """
        chars_with_pos: list of (char, x1,y1,x2,y2)
        Returns reconstructed text string
        """
        if not chars_with_pos:
            return ""
        text = ''.join([c[0] for c in chars_with_pos])
        return text
    
    def add_spaces_for_words(self, line_chars: list, char_width_thresh: float = 1.5) -> str:
        """
        Add spaces based on gap between characters.
        line_chars: list of (char, x1,y1,x2,y2)
        """
        if not line_chars:
            return ""
        result = line_chars[0][0] if line_chars else ''
        for i in range(1, len(line_chars)):
            prev = line_chars[i-1]
            curr = line_chars[i]
            gap = curr[1] - prev[3]  # x1_curr - x2_prev
            avg_w = (prev[3]-prev[1] + curr[3]-curr[1]) / 2.0
            if gap > char_width_thresh * avg_w:
                result += ' ' + curr[0]
            else:
                result += curr[0]
        return result
