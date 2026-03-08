#!/bin/bash

# Dependency Fix Script
set -e

echo "🔧 Fixing Dependencies"
echo ""

# Remove old venv
rm -rf venv

# Create fresh venv
python3 -m venv venv
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip setuptools wheel

echo "📥 Installing in correct order..."

# Install PyTorch first
pip install torch==2.1.2

# Install transformers and huggingface
pip install transformers==4.36.2 huggingface-hub==0.20.3

# Install sentence-transformers
pip install sentence-transformers==2.3.1

# Install sklearn (use older version for Python 3.11 compatibility)
pip install scikit-learn==1.3.2

# Install numpy and scipy
pip install numpy==1.24.3 scipy==1.11.4

# Install FAISS
pip install faiss-cpu==1.7.4

# Install FastAPI
pip install fastapi==0.104.1 uvicorn[standard]==0.24.0

# Install remaining deps
pip install pydantic==2.5.0 pandas==2.1.3 python-multipart==0.0.6 \
    pytest==7.4.3 pytest-asyncio==0.21.1 pytest-cov==4.1.0 \
    httpx==0.25.1 tqdm==4.66.1 python-dotenv==1.0.0

echo "✅ Installation Complete!"
echo ""
echo "Run: python scripts/prepare_data.py"
