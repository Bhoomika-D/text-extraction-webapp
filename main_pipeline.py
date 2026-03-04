"""
Main Pipeline: 2D Filtering + Deep Learning Text Extraction
Combines preprocessing and OCR for complex background images
"""

import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional
import json
from datetime import datetime

# Assuming these modules exist and TextExtractor handles the OCR choice (Tesseract, EasyOCR, etc.)
from filter_2d import Filter2D
from text_extraction import TextExtractor # This module must be able to call Tesseract
#from data_augmentation import DataAugmentation


class TextExtractionPipeline:
    """
    Complete pipeline for text extraction from complex backgrounds
    """
    
    def __init__(self, 
                 filter_method: str = 'combined',
                 use_easyocr: bool = True,
                 #use_paddleocr: bool = False,
                 use_tesseract: bool = False):
        """
        Initialize pipeline
        
        Args:
            filter_method: 2D filter method to use
            use_easyocr: Use EasyOCR for text extraction
            use_paddleocr: Use PaddleOCR for text extraction
            use_tesseract: Use Tesseract for text extraction
        """
        self.filter_2d = Filter2D()
        # Initialize TextExtractor, passing the OCR choice flags
        self.text_extractor = TextExtractor(
            use_easyocr=use_easyocr,
            #use_paddleocr=use_paddleocr,
            use_tesseract=use_tesseract 
        )
       
        self.filter_method = filter_method
        # Determine the primary OCR engine being used
        self.primary_ocr_engine = self._determine_primary_ocr()

    def _determine_primary_ocr(self):
        # Assumes TextExtractor has these flags set from initialization
        if self.text_extractor.tesseract_active:
             return 'Tesseract'
        elif self.text_extractor.easyocr_active:
             return 'EasyOCR'
        #elif self.text_extractor.paddleocr_active:
            # return 'PaddleOCR'
        return 'Unknown'
        
    # --- REMOVED: _tesseract_simple_preprocess is now handled internally by TesseractOCR class ---

    def process_image(self, 
                      image_path: str,
                      apply_filter: bool = True,
                      save_filtered: bool = False,
                      save_visualization: bool = True,
                      output_dir: str = 'outputs') -> Dict:
        """
        Process a single image through the complete pipeline
        """
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        # --- CONDITIONAL PREPROCESSING LOGIC ---
        
        # Default settings for the extractor call
        image_to_send_to_ocr = image
        applied_filter_method = 'none'
        run_ocr_preprocess = True # Default for Tesseract/EasyOCR/PaddleOCR if no 2D filter is used
        
        if self.primary_ocr_engine == 'Tesseract':
            # Case 1: Tesseract is the primary engine. 
            # Send the raw image; let TesseractOCR's internal method handle binarization/scaling.
            print("Using Tesseract: Relying on internal simple preprocessing (Grayscale+Otsu).")
            # The TesseractOCR class handles the preprocessing internally when preprocess=True (default)
            applied_filter_method = 'tesseract_internal' 
        
        elif apply_filter and self.primary_ocr_engine != 'Unknown':
            # Case 2: Deep Learning OCR is the primary engine AND filter is requested.
            # Apply the 2D filter first, then tell the OCR module NOT to preprocess again.
            print(f"Applying {self.filter_method} filter for {self.primary_ocr_engine}...")
            image_to_send_to_ocr = self.filter_2d.apply_filter(image, method=self.filter_method)
            applied_filter_method = self.filter_method
            run_ocr_preprocess = False # The image is already filtered
        
        else:
            # Case 3: No filter applied or engine unknown. Send raw image.
            print("Applying no external filter (using raw image).")
        
        # Step 2: Extract text 
        print(f"Extracting text using {self.primary_ocr_engine}...")
        results = self.text_extractor.extract_text(
            image_to_send_to_ocr,
            # Pass the instruction to the extractor based on the above logic
            preprocess=run_ocr_preprocess, 
            confidence_threshold=0.5
        )
        
        # --- OUTPUT SAVING & METADATA ---
        
        # Step 3: Create visualization
        if save_visualization:
            # Use original image for visualization to overlay bounding boxes clearly
            vis_image = self.text_extractor.visualize_results(image, results)
            
            # Save outputs
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            
            image_name = Path(image_path).stem
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Save filtered image (only if an external 2D filter was applied)
            if save_filtered and applied_filter_method not in ('none', 'tesseract_internal'):
                filtered_path = f"{output_dir}/{image_name}_filtered_{applied_filter_method}_{timestamp}.jpg"
                # Need to check if the filtered image is color (BGR) or single channel (grayscale) for saving
                if len(image_to_send_to_ocr.shape) == 2:
                    cv2.imwrite(filtered_path, image_to_send_to_ocr)
                elif len(image_to_send_to_ocr.shape) == 3:
                    cv2.imwrite(filtered_path, image_to_send_to_ocr)

                results['filtered_image_path'] = filtered_path
            
            # Save visualization
            vis_path = f"{output_dir}/{image_name}_result_{timestamp}.jpg"
            cv2.imwrite(vis_path, vis_image)
            results['visualization_path'] = vis_path
        
        # Add metadata
        results['input_image'] = image_path
        results['filter_method'] = applied_filter_method
        results['ocr_engine'] = self.primary_ocr_engine
        results['timestamp'] = datetime.now().isoformat()
        
        return results
    
    


def main():
    """Example usage"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Text Extraction Pipeline')
    parser.add_argument('--image', type=str, help='Path to input image')
    parser.add_argument('--image_dir', type=str, help='Directory with images')
    parser.add_argument('--filter', type=str, default='combined',
                        help='Filter method to use')
    parser.add_argument('--output', type=str, default='outputs',
                        help='Output directory')
    parser.add_argument('--augment', action='store_true',
                        help='Generate augmented dataset')
    parser.add_argument('--compare', action='store_true',
                        help='Compare different filter methods')
    # Added argument to specify Tesseract usage
    parser.add_argument('--use-tesseract', action='store_true',
                        help='Use Tesseract OCR (bypasses complex 2D filters)')
    
    args = parser.parse_args()
    
    # Initialize pipeline
    # Pass Tesseract flag to initialization
    pipeline = TextExtractionPipeline(
        filter_method=args.filter, 
        use_tesseract=args.use_tesseract,
        use_easyocr=not args.use_tesseract, # Default to EasyOCR if Tesseract isn't forced
        use_paddleocr=False # Keeping PaddleOCR off by default for simplicity
    )
    
    #if args.augment and args.image_dir:
        # Generate augmented dataset
     #   pipeline.prepare_training_data(
      #      args.image_dir,
       #     output_dir=args.output,
        #    augmentations_per_image=10
       # )
    
    if args.compare and args.image:
        # Compare filter methods
        results = pipeline.compare_filters(args.image, output_dir=args.output)
        print("\nComparison Results:")
        for method, result in results.items():
            print(f"\n{method}:")
            print(f"  Text: {result['text'][:100]}...")
            print(f"  Detections: {result['num_detections']}")
            print(f"  Avg Confidence: {result['avg_confidence']:.2f}")
    
    elif args.image:
        # Process single image
        results = pipeline.process_image(
            args.image,
            output_dir=args.output
        )
        print("\nExtraction Results:")
        print(f"OCR Engine Used: {results['ocr_engine']}")
        print(f"Filter Method: {results['filter_method']}")
        print(f"Combined Text: {results['combined_text']}")
        print(f"English Text: {results['english_text']}")
        print(f"Persian Text: {results['persian_text']}")
        print(f"Visualization saved to: {results['visualization_path']}")
    
    elif args.image_dir:
        # Process batch
        image_paths = list(Path(args.image_dir).glob('*.jpg')) + \
                      list(Path(args.image_dir).glob('*.png'))
        pipeline.process_batch(
            [str(p) for p in image_paths],
            output_dir=args.output
        )
    
    else:
        print("Please provide --image or --image_dir argument")
        parser.print_help()


if __name__ == '__main__':
    main()