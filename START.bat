@echo off
echo ========================================
echo Text Extraction Web Application
echo ========================================
echo.
echo Installing dependencies...
pip install -r requirements.txt
echo.
echo Starting server...
echo Open your browser to: http://localhost:5000
echo.
python start_webapp.py
pause
