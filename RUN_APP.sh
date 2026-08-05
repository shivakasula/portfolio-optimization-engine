#!/bin/bash
# Quick start script for the Streamlit app

set -e

echo "🚀 Portfolio Optimization Engine - Streamlit App"
echo "=============================================="
echo ""

# Check if in correct directory
if [ ! -f "app.py" ]; then
    echo "❌ Error: app.py not found. Please run from project root."
    exit 1
fi

# Check if venv exists
if [ ! -d "venv" ]; then
    echo "❌ Error: Virtual environment not found. Please set up venv first."
    exit 1
fi

# Activate venv and run streamlit
echo "📦 Activating virtual environment..."
source venv/bin/activate

echo "✅ Virtual environment activated"
echo ""
echo "🌐 Starting Streamlit app..."
echo "📍 App will open at: http://localhost:8501"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

streamlit run app.py --logger.level=warning
