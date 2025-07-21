#!/bin/bash


if [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
fi

LOG_FILE="live_detection_run.log"

echo "=============================================="
echo "⚡ SMART GRID - LIVE DETECTION MODE ⚡"
echo "=============================================="
echo "🧹 Clearing previous live detection log file..."
> "$LOG_FILE"

echo "🚀 Starting simulation..."
echo "   The AnomalyInjector will inject attacks."
echo "   The FraudDetector will try to catch them."
echo "   Watch the logs for '!!! ANOMALY' and '!!! LIVE ALERT' messages."
echo "----------------------------------------------"


python main.py detect \
    --prosumers 10 \
    --consumers 20 \
    --difficulty 3 \
    --duration 300

echo "=============================================="
echo "✅ Live detection simulation finished."
echo "   Check the log file: $LOG_FILE"
echo "=============================================="