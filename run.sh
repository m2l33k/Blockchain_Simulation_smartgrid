#!/bin/bash

# Activate the virtual environment if it exists
if [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
fi

# Set a variable for the log file to easily clear it
LOG_FILE="smartgrid_simulation.log"

# Clear the log file before each run for a clean slate
# This makes debugging much easier
echo "Clearing previous log file..."
# > "$LOG_FILE"  # Commented out to allow logs to accumulate across runs


echo "Starting simulation..."
python main.py \
    --prosumers 5 \
    --consumers 10 \
    --difficulty 4 \
    --duration 120
echo "Simulation completed. Check $LOG_FILE for details."
# deactivate