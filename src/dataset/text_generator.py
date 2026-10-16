import os
import random
from typing import Tuple, List, Dict
from PIL import Image, ImageDraw, ImageFont

class TextImageGenerator:
    """
    Generate synthetic text images using PIL for testing OCR segmentation.
    """
    def __init__(self, font_size: int = 32, font_path: str = None, 
                 image_width: int = 800, line_spacing: int = 20):
        self.font_size = font_size
        self.image_width = image_width
        self.line_spacing = line_spacing
        self.font = self._load_font(font_path)
        
    def _load_font(self, font_path: str = None):
        try:
            if font_path and os.path.exists(font_path):
                return ImageFont.truetype(font_path, self.font_size)
            for f in ['DejaVuSans.ttf', 'Arial.ttf', 'LiberationSans-Regular.ttf', 'FreeSans.ttf']:
                try:
                    return ImageFont.truetype(f, self.font_size)
                except:
                    continue
            return ImageFont.load_default()
        except Exception:
            return ImageFont.load_default()
    
    def generate_image(self, text: str, bg_color: str = 'white', 
                      text_color: str = 'black') -> Tuple[Image.Image, str]:
        lines = text.split('\n')
        line_height = self.font_size + self.line_spacing
        height = len(lines) * line_height + 30
        img = Image.new('RGB', (self.image_width, height), bg_color)
        draw = ImageDraw.Draw(img)
        y = 15
        for line in lines:
            draw.text((10, y), line, fill=text_color, font=self.font)
            y += line_height
        return img, text
    
    def generate_random_text(self, num_lines: int = 2, chars_per_line: int = 30,
                           include_spaces: bool = True) -> Tuple[Image.Image, str]:
        chars = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'
        lines = []
        for _ in range(num_lines):
            line = ''
            for i in range(chars_per_line):
                line += random.choice(chars)
                if include_spaces and i > 5 and i % 7 == 0:
                    line += ' '
            lines.append(line)
        text = '\n'.join(lines)
        return self.generate_image(text)
    
    def generate_test_document(self, save_path: str = None) -> Tuple[Image.Image, str]:
        text = "Hello World 123\nABCD efgh\nTestOCR 456"
        img, gt = self.generate_image(text)
        if save_path:
            img.save(save_path)
        return img, gt
    
    def save_synthetic_dataset(self, output_dir: str = 'data/synthetic', 
                             num_samples: int = 10) -> List[Dict]:
        os.makedirs(output_dir, exist_ok=True)
        samples = []
        for i in range(num_samples):
            img, gt = self.generate_random_text(num_lines=random.randint(1,3), 
                                               chars_per_line=random.randint(15,40))
            img_path = os.path.join(output_dir, f'test_{i}.png')
            img.save(img_path)
            samples.append({'image_path': img_path, 'ground_truth': gt})
        return samples
