"""
Deep Learning and Hybrid Text Extraction Module
Supports EasyOCR, PaddleOCR, and TesseractOCR (Farsi only)
"""

import cv2
import numpy as np
from typing import List, Tuple, Dict, Optional
import re

# --- 1. IMPORT TESSERACT MODULE ---
try:
    from tesseract_ocr import TesseractOCR
except ImportError:
    # Define a placeholder class or handle error if the file is missing
    TesseractOCR = None
    print("Warning: tesseract_ocr.py module not found. Tesseract functionality will be disabled.")


class TextExtractor:
    """
    Text extraction using deep learning OCR and Tesseract (for Farsi)
    """
    
    # --- 2. ADD use_tesseract TO __init__ ---
    def __init__(self, 
                 use_easyocr: bool = True, 
                 use_paddleocr: bool = False,
                 use_tesseract: bool = False):
        """
        Initialize text extractor
        """
        self.easyocr_active = use_easyocr
        self.paddleocr_active = use_paddleocr
        self.tesseract_active = use_tesseract # Store the new flag
        
        self.reader = None
        self.paddle_ocr = None
        self.tesseract_module = None # New attribute
        
        self._initialize_ocr()
    
    def _initialize_ocr(self):
        """Initialize OCR engines"""
        
        # EasyOCR Initialization
        if self.easyocr_active:
            try:
                import easyocr
                self.reader = easyocr.Reader(['en', 'fa'], gpu=False)
                print("EasyOCR initialized with English and Persian support")
            except ImportError:
                print("EasyOCR not installed. EasyOCR disabled.")
                self.easyocr_active = False
        
        # PaddleOCR Initialization
        if self.paddleocr_active:
            try:
                from paddleocr import PaddleOCR
                self.paddle_ocr = PaddleOCR(use_angle_cls=True, lang='en', use_gpu=False)
                print("PaddleOCR initialized")
            except ImportError:
                print("PaddleOCR not installed. PaddleOCR disabled.")
                self.paddleocr_active = False

        # --- 3. INITIALIZE TESSERACT MODULE ---
        if self.tesseract_active and TesseractOCR is not None:
            self.tesseract_module = TesseractOCR(lang='fas')
            print("TesseractOCR (Farsi only) initialized.")
        elif self.tesseract_active and TesseractOCR is None:
             print("Tesseract requested but TesseractOCR module not available. Tesseract disabled.")
             self.tesseract_active = False


    def extract_text(self, image: np.ndarray, 
                     preprocess: bool = True,
                     confidence_threshold: float = 0.5) -> Dict:
        """
        Extract text from image
        """
        
        # --- Handle Preprocessing based on the primary OCR engine ---
        
        processed_image = image
        if preprocess:
            # Tesseract has internal logic for simple preprocess in its method
            if not self.tesseract_active: # Only run complex preprocess if DL engine is active
                if len(image.shape) == 3:
                    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                else:
                    gray = image.copy()
                processed_image = self._preprocess_for_ocr(gray)
        
        # --- End Preprocessing ---

        results = {
            'text': [],
            'boxes': [],
            'confidences': [],
            'languages': []
        }
        
        # --- New: Tesseract Extraction Logic (Highest Priority if active) ---
        if self.tesseract_active and self.tesseract_module:
            print("Running TesseractOCR...")
            tesseract_results = self.tesseract_module.extract_text(
                processed_image, 
                preprocess=preprocess # Pass preprocessing flag to Tesseract module
            )
            # Tesseract is often used alone, so we'll treat its output as the final one
            # for the current implementation.
            
            # Note: We return Tesseract results immediately here to avoid merging issues 
            # with deep learning models, as Tesseract's preprocessing is incompatible 
            # with the output of the 2D filters in the pipeline.
            tesseract_results['ocr_method'] = 'Tesseract OCR (Farsi only)'
            return tesseract_results
        
        # --- Deep Learning Extraction Logic (If Tesseract is NOT active) ---
        
        if self.easyocr_active and self.reader:
            easyocr_results = self._extract_with_easyocr(processed_image, confidence_threshold)
            results['text'].extend(easyocr_results['text'])
            results['boxes'].extend(easyocr_results['boxes'])
            results['confidences'].extend(easyocr_results['confidences'])
            results['languages'].extend(easyocr_results['languages'])
        
        if self.paddleocr_active and self.paddle_ocr:
            paddle_results = self._extract_with_paddleocr(processed_image, confidence_threshold)
            # Merge results (avoid duplicates is complex, simple extend for now)
            results['text'].extend(paddle_results['text'])
            results['boxes'].extend(paddle_results['boxes'])
            results['confidences'].extend(paddle_results['confidences'])
            results['languages'].extend(paddle_results['languages'])
        
        # Combine all text
        results['combined_text'] = ' '.join(results['text'])
        results['english_text'] = self._extract_english_only(results['text'])
        results['persian_text'] = self._extract_persian_only(results['text'])
        results['ocr_method'] = 'EasyOCR / PaddleOCR'
        
        return results
    
    # (rest of utility methods remain largely the same, but ensure visualize_results is robust)
    
    def _preprocess_for_ocr(self, gray: np.ndarray) -> np.ndarray:
        """Preprocess image for better OCR results"""
        # Enhance contrast
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        
        # Denoise
        denoised = cv2.fastNlMeansDenoising(enhanced, h=10)
        
        # Sharpen
        kernel = np.array([[-1, -1, -1],
                           [-1,  9, -1],
                           [-1, -1, -1]])
        sharpened = cv2.filter2D(denoised, -1, kernel)
        
        return sharpened
    
    def _extract_with_easyocr(self, image: np.ndarray, 
                              confidence_threshold: float) -> Dict:
        """Extract text using EasyOCR"""
        results = {
            'text': [],
            'boxes': [],
            'confidences': [],
            'languages': []
        }
        
        try:
            detections = self.reader.readtext(image)
            
            for detection in detections:
                bbox, text, confidence = detection
                
                if confidence >= confidence_threshold:
                    results['text'].append(text)
                    results['boxes'].append(bbox)
                    results['confidences'].append(confidence)
                    
                    lang = self._detect_language(text)
                    results['languages'].append(lang)
        
        except Exception as e:
            print(f"EasyOCR error: {e}")
        
        return results
    
    def _extract_with_paddleocr(self, image: np.ndarray,
                               confidence_threshold: float) -> Dict:
        """Extract text using PaddleOCR"""
        results = {
            'text': [],
            'boxes': [],
            'confidences': [],
            'languages': []
        }
        
        try:
            ocr_results = self.paddle_ocr.ocr(image, cls=True)
            
            if ocr_results and ocr_results[0]:
                for line in ocr_results[0]:
                    if line:
                        bbox, (text, confidence) = line
                        
                        if confidence >= confidence_threshold:
                            results['text'].append(text)
                            results['boxes'].append(bbox)
                            results['confidences'].append(confidence)
                            
                            lang = self._detect_language(text)
                            results['languages'].append(lang)
        
        except Exception as e:
            print(f"PaddleOCR error: {e}")
        
        return results
    
    def _detect_language(self, text: str) -> str:
        """Simple language detection (English vs Persian)"""
        persian_pattern = re.compile(r'[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\uFB50-\uFDFF\uFE70-\uFEFF]')
        
        if persian_pattern.search(text):
            return 'persian'
        elif re.search(r'[a-zA-Z]', text):
            return 'english'
        else:
            return 'mixed'
    
    def _extract_english_only(self, texts: List[str]) -> str:
        """Extract only English text"""
        english_texts = []
        for text in texts:
            english_only = re.sub(r'[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\uFB50-\uFDFF\uFE70-\uFEFF]', '', text)
            english_only = english_only.strip()
            if english_only:
                english_texts.append(english_only)
        return ' '.join(english_texts)
    
    def _extract_persian_only(self, texts: List[str]) -> str:
        """Extract only Persian text"""
        persian_texts = []
        persian_pattern = re.compile(r'[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\uFB50-\uFDFF\uFE70-\uFEFF]+')
        
        for text in texts:
            matches = persian_pattern.findall(text)
            if matches:
                persian_texts.extend(matches)
        
        return ' '.join(persian_texts)
    
    def visualize_results(self, image: np.ndarray, results: Dict) -> np.ndarray:
        """
        Visualize text extraction results on image
        """
        vis_image = image.copy()
        
        for i, (bbox, text, conf) in enumerate(zip(
            results['boxes'], 
            results['text'], 
            results['confidences']
        )):
            # Draw bounding box
            if isinstance(bbox, list) and len(bbox) >= 4:
                # Tesseract/PaddleOCR format: list of 4 points [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
                if isinstance(bbox[0], (list, tuple)) and len(bbox) == 4:
                    pts = np.array(bbox, dtype=np.int32)
                    cv2.polylines(vis_image, [pts], True, (0, 255, 0), 2)
                    
                    # Determine top-left corner for text label
                    x, y = int(bbox[0][0]), int(bbox[0][1])
                else:
                    # Fallback for unexpected format (e.g., Simple Tesseract: [x, y, w, h])
                    x, y = 10, 30 * (i + 1)
            else:
                x, y = 10, 30 * (i + 1)

            # Add text label (only if a valid position was found)
            if x != 10 or y != 30 * (i + 1):
                text_preview = f"{text[:20]} ({conf:.2f})"
                (text_width, text_height), baseline = cv2.getTextSize(
                    text_preview, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
                )
                
                # Draw text background
                cv2.rectangle(vis_image, (x, y - text_height - baseline - 5), 
                              (x + text_width, y), (0, 255, 0), -1)
                cv2.putText(vis_image, text_preview, (x, y - baseline - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        
        return vis_image