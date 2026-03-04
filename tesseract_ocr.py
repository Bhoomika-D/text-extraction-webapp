"""
Tesseract OCR Module - Optimized for Automatic Farsi (Persian) Extraction

Uses streamlined preprocessing (Grayscale + Resizing + Otsu's) and 
flexible PSM 3 for improved results, configured exclusively for Farsi.
"""

import cv2
import numpy as np
import pytesseract
from typing import Dict, List
import re
import os
import platform


class TesseractOCR:
    """
    Tesseract OCR with streamlined preprocessing for automatic Farsi extraction.
    """
    
    def __init__(self, lang='fas'):
        """
        Initialize Tesseract OCR and attempt to set the path.
        
        Args:
            lang: Language code (set to 'fas' for Farsi only)
        """
        self.lang = lang
        self.original_h = 0
        self.original_w = 0
        self.scale = 1.0
        
        # --- Tesseract Path Configuration (Windows Specific) ---
        try:
            if platform.system() == 'Windows':
                # Common Windows installation paths
                possible_paths = [
                    r'C:\Program Files\Tesseract-OCR\tesseract.exe',
                    r'C:\Program Files (x86)\Tesseract-OCR\tesseract.exe',
                ]
                for path in possible_paths:
                    if os.path.exists(path):
                        pytesseract.pytesseract.tesseract_cmd = path
                        print(f"✅ Tesseract path set to: {path}")
                        break
                else:
                    print("⚠️ Tesseract path not found. Ensure it's in your PATH or set manually.")
        except Exception as e:
            print(f"Note: Tesseract path setup encountered an error: {e}")
            
    def preprocess_simple(self, image: np.ndarray) -> np.ndarray:
        """
        Streamlined preprocessing: Grayscale + Resizing + Otsu's Binarization.
        Stores scale for bounding box correction.
        """
        # 1. Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        self.original_h, self.original_w = gray.shape

        # 2. Resize if too small (Aims for better Tesseract performance)
        if self.original_w < 600: 
            self.scale = 600 / self.original_w
            gray = cv2.resize(gray, None, fx=self.scale, fy=self.scale, interpolation=cv2.INTER_CUBIC)
        else:
            self.scale = 1.0
        
        # 3. Apply Otsu's threshold for automatic binarization
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Optional: Invert if text is white on black 
        if cv2.countNonZero(binary) < (binary.size * 0.5):
            binary = cv2.bitwise_not(binary)
            
        return binary
    
    def extract_text(self, image: np.ndarray, 
                     preprocess: bool = True,
                     config: str = None) -> Dict:
        """
        Extract text using Tesseract OCR
        """
        
        if preprocess:
            processed_image = self.preprocess_simple(image)
        else:
            # Handle grayscale conversion if no complex preprocessing is used
            if len(image.shape) == 3:
                processed_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                processed_image = image.copy()
            self.scale = 1.0 # Ensure scale is 1.0 if not preprocessed

        
        # --- Default Tesseract Config ---
        if config is None:
            # PSM 3: Fully automatic page segmentation (flexible layout detection)
            config = '--psm 3'
        
        # Extract text with Tesseract
        try:
            # Get detailed data
            data = pytesseract.image_to_data(
                processed_image,
                lang=self.lang,
                config=config,
                output_type=pytesseract.Output.DICT
            )
            
            # Get full text
            full_text = pytesseract.image_to_string(
                processed_image,
                lang=self.lang,
                config=config
            )
            
            # Parse results
            text_list = []
            confidences = []
            boxes = []
            inv_scale = 1.0 / self.scale # Inverse scale for correction
            
            n_boxes = len(data['text'])
            for i in range(n_boxes):
                text = data['text'][i].strip()
                try:
                    conf = int(data['conf'][i])
                except ValueError:
                    conf = 0
                
                # Filter out empty text and very low confidence
                if text and conf > 30: 
                    text_list.append(text)
                    confidences.append(conf / 100.0)
                    
                    # Bounding box coordinates from Tesseract (on scaled image)
                    x, y, w, h = data['left'][i], data['top'][i], data['width'][i], data['height'][i]
                    
                    # Correct coordinates back to original image scale
                    x, y, w, h = (int(val * inv_scale) for val in (x, y, w, h))
                    
                    # Convert to four-point format: [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
                    boxes.append([[x, y], [x+w, y], [x+w, y+h], [x, y+h]])
            
            # Detect languages
            languages = [self._detect_language(text) for text in text_list]
            
            # Separate English and Persian 
            english_texts = [t for t, l in zip(text_list, languages) if l == 'english']
            persian_texts = [t for t, l in zip(text_list, languages) if l == 'persian']
            
            results = {
                'text': text_list,
                'boxes': boxes,
                'confidences': confidences,
                'languages': languages,
                'combined_text': full_text.strip(),
                'english_text': ' '.join(english_texts),
                'persian_text': ' '.join(persian_texts),
                'ocr_method': 'Tesseract OCR (Farsi Only)'
            }
            
            return results
            
        except Exception as e:
            # Tesseract failed (e.g., image input error or Tesseract is not installed)
            print(f"Tesseract extraction failed: {e}")
            return {
                'text': [],
                'boxes': [],
                'confidences': [],
                'languages': [],
                'combined_text': '',
                'english_text': '',
                'persian_text': '',
                'ocr_method': 'Tesseract OCR (Farsi Only)',
                'error': str(e)
            }
    
    def _detect_language(self, text: str) -> str:
        """Simple language detection (Persian vs English)"""
        # Check for Persian/Farsi characters (Arabic script range)
        persian_pattern = re.compile(r'[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\uFB50-\uFDFF\uFE70-\uFEFF]')
        
        if persian_pattern.search(text):
            return 'persian'
        elif re.search(r'[a-zA-Z]', text):
            return 'english'
        else:
            return 'mixed'
    
    def visualize_results(self, image: np.ndarray, results: Dict) -> np.ndarray:
        """Visualize text extraction results on image"""
        if len(image.shape) == 2:
             vis_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        else:
             vis_image = image.copy()
        
        # Use a Farsi-friendly font for visualization if possible, or stick to default
        font = cv2.FONT_HERSHEY_SIMPLEX 
        
        for i, (bbox, text, conf) in enumerate(zip(
            results.get('boxes', []),
            results.get('text', []),
            results.get('confidences', [])
        )):
            if bbox and len(bbox) >= 4:
                # Draw bounding box (Green)
                pts = np.array(bbox, dtype=np.int32)
                cv2.polylines(vis_image, [pts], True, (0, 255, 0), 2)
                
                # Add text label background and text
                x, y = int(bbox[0][0]), int(bbox[0][1])
                text_preview = text[:20] + "..." if len(text) > 20 else text
                label = f"({conf:.1%})" # Only confidence label to avoid font issues
                
                # Calculate size for confidence label
                (text_width, text_height), _ = cv2.getTextSize(
                    label, font, 0.5, 1
                )
                # Position label above the box
                cv2.rectangle(vis_image, (x, y - text_height - 5), 
                              (x + text_width, y), (0, 255, 0), -1)
                cv2.putText(vis_image, label, (x, y - 5),
                            font, 0.5, (0, 0, 0), 1)
        
        return vis_image