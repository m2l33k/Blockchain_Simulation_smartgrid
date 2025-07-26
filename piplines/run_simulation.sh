#!/bin/bash

# ==============================================================================
#  Run Blockchain Energy Grid Simulation
# ==============================================================================
# This script runs the simple, fast energy grid simulation and saves ALL
# output (both standard output and errors) to a timestamped log file.
#
# USAGE:
#   ./run_simulation.sh
#   (You can customize the parameters in the CONFIGURATION section below)
# ==============================================================================

# --- CONFIGURATION ---
# Adjust these parameters to change the simulation behavior.
SIMULATION_DURATION=300   # How long to run the simulation (in seconds).
DIFFICULTY=2            # Mining difficulty. Lower is faster (e.g., 2 or 3). Higher is slower.
PROSUMERS=10            # Number of energy-producing nodes.
CONSUMERS=20            # Number of energy-consuming nodes.

# --- SCRIPT LOGIC ---
# Do not change the lines below unless you know what you are doing.

# 1. Define the log file name with a timestamp for uniqueness.
TIMESTAMP=$(date +"%Y-%m-%d_%H-%M-%S")
LOG_FILE="simulation_logs/simulation_run_${TIMESTAMP}.log"

# 2. Set the name of the Python virtual environment directory.
VENV_NAME="venv"

echo "======================================="
echo "  STARTING BLOCKCHAIN SIMULATION"
echo "======================================="
echo "  - Duration:   ${SIMULATION_DURATION} seconds"
echo "  - Difficulty: ${DIFFICULTY}"
echo "  - Log File:   ${LOG_FILE}"
echo "---------------------------------------"

# 3. Activate the virtual environment if it exists.
if [ -d "$VENV_NAME" ]; then
    echo "Activating Python virtual environment..."
    source "$VENV_NAME/bin/activate"
else
    echo "Warning: Virtual environment '$VENV_NAME' not found. Using system Python."
fi

# 4. Construct the Python command with all arguments.
COMMAND="python3 simulation.py \
    --difficulty $DIFFICULTY \
    --duration $SIMULATION_DURATION \
    --prosumers $PROSUMERS \
    --consumers $CONSUMERS"

# 5. Execute the command and redirect ALL output to the log file.
#    - '2>&1' redirects stderr (errors) to stdout (standard output).
#    - '| tee' sends the output to both the log file AND your terminal screen.
echo "Running command: $COMMAND"
echo "Output will be saved to '$LOG_FILE' and displayed on screen."
echo "Press Ctrl+C to stop the simulation early."
echo "---------------------------------------"

$COMMAND 2>&1 | tee "$LOG_FILE"

# The 'set -e' command is not used here to ensure the final message
# is printed even if the python script fails.
EXIT_CODE=${PIPESTATUS[0]}

echo "---------------------------------------"
if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ Simulation finished successfully."
else
    echo "❌ Simulation finished with an error (Exit Code: $EXIT_CODE)."
fi
echo "Log file saved at: $(pwd)/$LOG_FILE"
echo "======================================="