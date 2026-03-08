#!/bin/bash

# Setup script for Semantic Search System
# This script sets up the virtual environment and prepares the data

set -e  # Exit on error

echo "================================================"
echo "Semantic Search System - Setup"
echo "================================================"
echo ""

# Check Python version
echo "🐍 Checking Python version..."
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "   Found Python $python_version"
echo ""

# Create virtual environment
echo "📦 Creating virtual environment..."
if [ -d "venv" ]; then
    echo "   Virtual environment already exists"
else
    python3 -m venv venv
    echo "   ✅ Virtual environment created"
fi
echo ""

# Activate virtual environment
echo "🔌 Activating virtual environment..."
source venv/bin/activate
echo "   ✅ Virtual environment activated"
echo ""

# Upgrade pip
echo "⬆️  Upgrading pip..."
pip install --upgrade pip setuptools wheel
echo ""

# Install dependencies
echo "📚 Installing dependencies..."
pip install -r requirements.txt
echo "   ✅ Dependencies installed"
echo ""

# Create data directories
echo "📁 Creating data directories..."
mkdir -p data/raw
mkdir -p data/processed
mkdir -p data/embeddings
echo "   ✅ Directories created"
echo ""

# Prepare data
echo "================================================"
echo "🚀 Preparing Dataset"
echo "================================================"
echo ""
echo "This will:"
echo "  1. Download 20 Newsgroups dataset (~20k documents)"
echo "  2. Clean and preprocess text"
echo "  3. Generate embeddings (takes ~5-10 minutes)"
echo "  4. Build FAISS index"
echo "  5. Train fuzzy clustering model"
echo ""
read -p "Proceed with data preparation? (y/n) " -n 1 -r
echo ""

if [[ $REPLY =~ ^[Yy]$ ]]
then
    echo ""
    python scripts/prepare_data.py
    echo ""
    echo "✅ Data preparation complete!"
else
    echo ""
    echo "⚠️  Skipped data preparation"
    echo "   Run manually: python scripts/prepare_data.py"
fi

echo ""
echo "================================================"
echo "✨ Setup Complete!"
echo "================================================"
echo ""
echo "Next steps:"
echo ""
echo "1. Activate the virtual environment:"
echo "   source venv/bin/activate"
echo ""
echo "2. Start the API server:"
echo "   uvicorn app.main:app --host 0.0.0.0 --port 8000"
echo ""
echo "3. Visit the API documentation:"
echo "   http://localhost:8000/docs"
echo ""
echo "4. Run tests:"
echo "   pytest tests/ -v"
echo ""
echo "================================================"
