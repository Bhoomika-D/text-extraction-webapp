"""
Configuration file for text extraction pipeline
Adjust parameters here for your specific use case
"""

# Filter Configuration
FILTER_CONFIG = {
    'default_method': 'combined',  # Recommended: 'combined'
    'adaptive_threshold': {
        'block_size': 11,
        'C': 2
    },
    'morphological': {
        'kernel_size': 3,
        'operation': 'opening'  # 'opening', 'closing', 'gradient'
    },
    'top_hat': {
        'kernel_size': 9
    },
    'bottom_hat': {
        'kernel_size': 9
    },
    'clahe': {
        'clip_limit': 2.0,
        'tile_grid_size': (8, 8)
    },
    'unsharp_mask': {
        'sigma': 1.0,
        'strength': 1.5
    },
    'combined': {
        'use_clahe': True,
        'use_morphology': True,
        'use_unsharp': True
    }
}

# OCR Configuration
OCR_CONFIG = {
    'use_easyocr': True,  # Recommended: True (supports Persian)
    'use_paddleocr': False,  # Alternative OCR engine
    'confidence_threshold': 0.5,  # Lower (0.3-0.4) for difficult images
    'preprocess_before_ocr': True,  # Apply additional preprocessing
    'languages': ['en', 'fa']  # English and Persian (Farsi)
}

# Data Augmentation Configuration
AUGMENTATION_CONFIG = {
    'augmentations_per_image': 10,  # 100 images → 1100 images
    'methods': [
        'brightness',
        'contrast',
        'rotation',
        'blur',
        'noise',
        'perspective',
        'scale',
        'flip',
        'elastic'
    ],
    'brightness_range': (0.5, 1.5),
    'contrast_range': (0.7, 1.3),
    'rotation_range': (-15, 15),
    'blur_kernel_range': (1, 3),
    'noise_factor': 0.05,
    'perspective_max_shift': 0.1,
    'scale_range': (0.8, 1.2)
}

# Pipeline Configuration
PIPELINE_CONFIG = {
    'apply_filter': True,  # Apply 2D filtering before OCR
    'save_filtered': True,  # Save filtered images
    'save_visualization': True,  # Save images with bounding boxes
    'save_results_json': True,  # Save results as JSON
    'output_dir': 'outputs',
    'batch_size': None  # None = process all at once
}

# Filter Comparison Configuration
COMPARISON_CONFIG = {
    'methods_to_compare': [
        'combined',
        'adaptive_threshold',
        'clahe',
        'top_hat',
        'morphological'
    ]
}

# Multi-scale Configuration
MULTISCALE_CONFIG = {
    'enabled': False,  # Set to True for text at different sizes
    'scales': [1.0, 0.75, 0.5]  # Different scales to process
}

# Text Processing Configuration
TEXT_CONFIG = {
    'min_confidence': 0.3,  # Minimum confidence for text detection
    'separate_languages': True,  # Separate English and Persian text
    'remove_duplicates': True,  # Remove duplicate text detections
    'post_process': False  # Apply post-processing (future feature)
}

