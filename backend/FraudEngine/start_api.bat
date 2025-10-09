@echo off
REM Startup script for Fraud Detection API (Windows)

echo.
echo 🚀 Starting Fraud Detection API...
echo.
echo Note: Models will train on startup (takes 30-60 seconds)
echo       Please wait for 'API is now accepting requests' message
echo.

REM Check if data file exists
if not exist ".\data\SAML-D.csv" (
    echo ❌ ERROR: Data file not found at ..\..\data\SAML-D.csv
    echo    Please ensure the dataset is in the correct location
    pause
    exit /b 1
)

REM Start the API
python main.py
pause

