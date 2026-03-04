# Installation Guide

## Prerequisites
- Python 3.8 or higher
- Internet connection (for downloading OCR models)

## Step 1: Install Python Dependencies

```bash
pip install -r requirements.txt
```

This will install:
- Flask (web framework)
- OpenCV (image processing)
- EasyOCR (deep learning OCR)
- Tesseract Python wrapper
- And other dependencies

## Step 2: Install Tesseract OCR (Optional)

Tesseract is optional. If not installed, use "Complex Image" option which uses Deep Learning OCR.

### Windows:
1. Download from: https://github.com/UB-Mannheim/tesseract/wiki
2. Run installer
3. Select languages: English (eng) and Persian (fas)

### Linux:
```bash
sudo apt-get update
sudo apt-get install tesseract-ocr
sudo apt-get install tesseract-ocr-fas  # For Persian
```

### macOS:
```bash
brew install tesseract
brew install tesseract-lang
```

## Step 3: Run the Application

```bash
python start_webapp.py
```

Or:

```bash
python app.py
```

## Step 4: Open in Browser

Go to: **http://localhost:5000**

## First Run Notes

- EasyOCR will download models on first use (~500MB)
- This may take a few minutes
- Subsequent runs will be faster

## Troubleshooting

**Port 5000 already in use:**
- Change port in `app.py`: `app.run(port=5001)`

**Tesseract not found:**
- Use "Complex Image" option (works without Tesseract)
- Or install Tesseract (see Step 2)

**Module not found:**
- Run: `pip install -r requirements.txt`

**No text detected:**
- Try different filter methods
- Lower confidence threshold
- Try both Simple and Complex options
