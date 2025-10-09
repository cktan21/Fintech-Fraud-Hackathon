#!/bin/bash
# Startup script for Fraud Detection API

echo "🚀 Starting Fraud Detection API..."
echo ""
echo "Note: Models will train on startup (takes 30-60 seconds)"
echo "      Please wait for 'API is now accepting requests' message"
echo ""

# Check if data file exists
DATA_FILE="../../data/SAML-D.csv"
if [ ! -f "$DATA_FILE" ]; then
    echo "❌ ERROR: Data file not found at $DATA_FILE"
    echo "   Please ensure the dataset is in the correct location"
    exit 1
fi

# Start the API
python main.py

