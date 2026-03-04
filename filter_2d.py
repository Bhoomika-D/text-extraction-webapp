"""
2D Filtering Module for Text Extraction from Complex Backgrounds
Implements various filtering techniques to enhance text visibility
"""

import cv2
import numpy as np
from scipy import ndimage
from skimage import filters, morphology, restoration
from typing import Tuple, Optional


class Filter2D:
    """
    A comprehensive 2D filtering class for preprocessing images
    with complex backgrounds and varied illuminations
    """
    
    def __init__(self):
        self.filter_methods = {
            'adaptive_threshold': self._adaptive_threshold,
            'morphological': self._morphological_filter,
            'top_hat': self._top_hat_filter,
            'bottom_hat': self._bottom_hat_filter,
            'clahe': self._clahe_filter,
            'unsharp_mask': self._unsharp_mask,
            'wiener': self._wiener_filter,
            'gaussian_adaptive': self._gaussian_adaptive,
            'mser_preprocessing': self._mser_preprocessing,
            'combined': self._combined_filter
        }
    
    def apply_filter(self, image: np.ndarray, method: str = 'combined', **kwargs) -> np.ndarray:
        """
        Apply specified filter method to the image
        
        Args:
            image: Input image (BGR or grayscale)
            method: Filter method name
            **kwargs: Additional parameters for specific filters
            
        Returns:
            Filtered image
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        if method in self.filter_methods:
            return self.filter_methods[method](gray, **kwargs)
        else:
            raise ValueError(f"Unknown filter method: {method}")
    
    def _adaptive_threshold(self, gray: np.ndarray, block_size: int = 11, C: int = 2) -> np.ndarray:
        """Adaptive thresholding for varying illumination"""
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        # Adaptive threshold
        thresh = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, block_size, C
        )
        return thresh
    
    def _morphological_filter(self, gray: np.ndarray, 
                            kernel_size: int = 3, 
                            operation: str = 'opening') -> np.ndarray:
        """Morphological operations to enhance text"""
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
        
        if operation == 'opening':
            result = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel)
        elif operation == 'closing':
            result = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
        elif operation == 'gradient':
            result = cv2.morphologyEx(gray, cv2.MORPH_GRADIENT, kernel)
        else:
            result = gray
        
        return result
    
    def _top_hat_filter(self, gray: np.ndarray, kernel_size: int = 9) -> np.ndarray:
        """Top-hat filter to extract bright text on dark background"""
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
        tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel)
        return tophat
    
    def _bottom_hat_filter(self, gray: np.ndarray, kernel_size: int = 9) -> np.ndarray:
        """Bottom-hat filter to extract dark text on bright background"""
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
        blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
        return blackhat
    
    def _clahe_filter(self, gray: np.ndarray, 
                     clip_limit: float = 2.0, 
                     tile_grid_size: Tuple[int, int] = (8, 8)) -> np.ndarray:
        """Contrast Limited Adaptive Histogram Equalization"""
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        enhanced = clahe.apply(gray)
        return enhanced
    
    def _unsharp_mask(self, gray: np.ndarray, 
                     sigma: float = 1.0, 
                     strength: float = 1.5) -> np.ndarray:
        """Unsharp masking for edge enhancement"""
        blurred = cv2.GaussianBlur(gray, (0, 0), sigma)
        sharpened = cv2.addWeighted(gray, 1.0 + strength, blurred, -strength, 0)
        return np.clip(sharpened, 0, 255).astype(np.uint8)
    
    def _wiener_filter(self, gray: np.ndarray) -> np.ndarray:
        """Wiener filter for noise reduction"""
        # Convert to float for processing
        img_float = gray.astype(np.float64) / 255.0
        # Apply Wiener filter
        denoised = restoration.wiener(img_float, np.ones((3, 3)) / 9)
        return (np.clip(denoised * 255, 0, 255)).astype(np.uint8)
    
    def _gaussian_adaptive(self, gray: np.ndarray, 
                          sigma: float = 1.0) -> np.ndarray:
        """Gaussian adaptive filtering"""
        # Estimate local mean
        local_mean = cv2.GaussianBlur(gray.astype(np.float32), (0, 0), sigma)
        # Subtract mean to get text regions
        result = gray.astype(np.float32) - local_mean + 128
        return np.clip(result, 0, 255).astype(np.uint8)
    
    def _mser_preprocessing(self, gray: np.ndarray) -> np.ndarray:
        """Preprocessing optimized for MSER-based text detection"""
        # Enhance contrast
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        # Apply bilateral filter to preserve edges
        filtered = cv2.bilateralFilter(enhanced, 9, 75, 75)
        return filtered
    
    def _combined_filter(self, gray: np.ndarray, 
                        use_clahe: bool = True,
                        use_morphology: bool = True,
                        use_unsharp: bool = True) -> np.ndarray:
        """
        Combined filtering approach for complex backgrounds
        This is the recommended method for embossed text
        """
        result = gray.copy()
        
        # Step 1: CLAHE for illumination normalization
        if use_clahe:
            result = self._clahe_filter(result, clip_limit=2.0, tile_grid_size=(8, 8))
        
        # Step 2: Bilateral filter to reduce noise while preserving edges
        result = cv2.bilateralFilter(result, 9, 75, 75)
        
        # Step 3: Unsharp masking for edge enhancement
        if use_unsharp:
            result = self._unsharp_mask(result, sigma=1.0, strength=1.5)
        
        # Step 4: Morphological operations to enhance text strokes
        if use_morphology:
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
            result = cv2.morphologyEx(result, cv2.MORPH_CLOSE, kernel)
        
        return result
    
    def multi_scale_filter(self, image: np.ndarray, 
                          scales: list = [1.0, 0.75, 0.5]) -> list:
        """
        Apply filtering at multiple scales and return results
        Useful for text at different sizes
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        results = []
        for scale in scales:
            if scale != 1.0:
                h, w = gray.shape
                scaled = cv2.resize(gray, (int(w * scale), int(h * scale)))
            else:
                scaled = gray.copy()
            
            filtered = self._combined_filter(scaled)
            results.append(filtered)
        
        return results
    
    def create_text_mask(self, image: np.ndarray, 
                        method: str = 'adaptive') -> np.ndarray:
        """
        Create a binary mask highlighting text regions
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        if method == 'adaptive':
            # Use adaptive thresholding
            mask = self._adaptive_threshold(gray, block_size=11, C=2)
        elif method == 'otsu':
            # Use Otsu's thresholding
            _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        elif method == 'tophat':
            # Use top-hat for bright text
            mask = self._top_hat_filter(gray)
            _, mask = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        else:
            mask = gray
        
        return mask

