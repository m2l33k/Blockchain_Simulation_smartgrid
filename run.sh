#!/bin/bash

# This script runs the FULL pipeline:
# 1. Generates data with anomalies.
# 2. Pre-processes the data into a labeled CSV.
# 3. Trains the LSTM model on the new data.

# Activate the virtual environment
if [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
fi

echo "=============================================="
echo "⚡ SMART GRID - FULL TRAINING PIPELINE ⚡"
echo "=============================================="

# --- Step 1: Generate Data ---
echo "STEP 1: Starting 10-minute simulation to generate training data..."
# Pass the 'generate' mode argument to main.py
python main.py generate \
    --prosumers 10 \
    --consumers 20 \
    --difficulty 3 \
    --duration 600 

echo "----------------------------------------------"

# --- Step 2: Process Data ---
echo "STEP 2: Processing log file into featurized CSV..."
python data_preprocessor.py

echo "----------------------------------------------"

# --- Step 3: Train Model ---
echo "STEP 3: Training the LSTM model on the new data..."
python train_model.py

echo "=============================================="
echo "✅ Pipeline finished."
echo "   A new model has been saved to 'saved_models/'"
echo "   You can now run './run_live.sh' to test it."
echo "=============================================="