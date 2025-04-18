#!/bin/bash

# Create a virtual environment
python -m venv venv

# Activate the virtual environment
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Create models directory if it doesn't exist
mkdir -p models

echo "Environment setup complete!"
echo "To activate the virtual environment, run: source venv/bin/activate"
echo "Don't forget to place the model file (transformer_final.pth.tar) in the models directory"
echo "To start the server, run: uvicorn main:app --reload"