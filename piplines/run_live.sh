#!/bin/bash

# Activate the virtual environment if it exists
if [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
fi

# Define file paths
SIM_LOG_FILE="live_detection_run.log"
LATENCY_LOG_FILE="latency_log.csv"

echo "=============================================="
echo "⚡ SMART GRID - LIVE DETECTION & ANALYSIS ⚡"
echo "=============================================="
echo "🧹 Clearing previous log files..."
> "$SIM_LOG_FILE"
> "$LATENCY_LOG_FILE"  # <-- ADDED: Clear the old latency log

echo "🚀 Starting simulation..."
echo "   The AnomalyInjector will inject attacks."
echo "   The FraudDetector will try to catch them."
echo "   Latency events will be recorded to '$LATENCY_LOG_FILE'."
echo "----------------------------------------------"

# Run the main simulation
python main.py detect \
    --prosumers 10 \
    --consumers 20 \
    --difficulty 3 \
    --duration 300

echo "=============================================="
echo "✅ Live detection simulation finished."
echo "   Check the main log file: $SIM_LOG_FILE"

# --- ADDED: Automatically run the latency analysis ---
echo "📊 Generating latency analysis plot..."
python plot_latency.py

echo "📈 Plot generation complete."
echo "   Check the output image: detection_latency_curve.png"
echo "=============================================="