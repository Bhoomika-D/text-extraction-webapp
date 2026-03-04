"""
Startup script for the Text Extraction Web Application
"""

import os
import sys
import webbrowser
from pathlib import Path
import time

def check_dependencies():
    """Check if required dependencies are installed"""
    required = ['flask', 'cv2', 'numpy', 'easyocr']
    missing = []
    
    for module in required:
        try:
            if module == 'cv2':
                import cv2
            elif module == 'flask':
                import flask
            elif module == 'numpy':
                import numpy
            elif module == 'easyocr':
                import easyocr    
        except ImportError:
            missing.append(module)
    
    if missing:
        print("❌ Missing dependencies:")
        for m in missing:
            print(f"   - {m}")
        print("\nInstall with:")
        print("   pip install flask flask-cors opencv-python numpy scipy scikit-image easyocr")
        return False
    
    return True

def check_files():
    """Check if required files exist"""
    required_files = [
        'app.py',
        'main_pipeline.py',
        'filter_2d.py',
        'text_extraction.py',
        'templates/index.html',
        'static/style.css',
        'static/script.js'
    ]
    
    missing = []
    for file in required_files:
        if not Path(file).exists():
            missing.append(file)
    
    if missing:
        print("❌ Missing files:")
        for f in missing:
            print(f"   - {f}")
        return False
    
    return True

def main():
    print("=" * 70)
    print("Text Extraction Web Application - Startup")
    print("=" * 70)
    
    # Check dependencies
   
    
    # Check files
    print("\n🔍 Checking files...")
    if not check_files():
        sys.exit(1)
    print("✓ All files present")
    
    # Create necessary directories
    print("\n📁 Creating directories...")
    Path('uploads').mkdir(exist_ok=True)
    Path('static/outputs').mkdir(parents=True, exist_ok=True)
    print("✓ Directories ready")
    
    # Start server
    print("\n" + "=" * 70)
    print("Starting Flask server...")
    print("=" * 70)
    print("\n🌐 Server will start at: http://localhost:5000")
    print("📝 Press Ctrl+C to stop the server")
    print("\n⏳ Opening browser in 3 seconds...")
    print("=" * 70 + "\n")
    
    # Open browser after delay
    def open_browser():
        time.sleep(3)
        webbrowser.open('http://localhost:5000')
    
    import threading
    browser_thread = threading.Thread(target=open_browser)
    browser_thread.daemon = True
    browser_thread.start()
    
    # Import and run app
    try:
        from app import app
        port = int(os.environ.get("PORT", 5000))
        app.run(debug=False, host="0.0.0.0", port=port)
    
    except KeyboardInterrupt:
        print("\n\n👋 Server stopped by user")
    except Exception as e:
        print(f"\n❌ Error starting server: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()

