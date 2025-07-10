#!/bin/bash

# Activate virtual environment if exists
if [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
fi

LOG_FILE="smartgrid_simulation.log"
RESULT_FILE="results.log"

# Define colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[1;34m'
CYAN='\033[1;36m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Visual Start
echo -e "${BLUE}=============================================="
echo -e "⚡ ${CYAN}SMART GRID SIMULATION - RUNNING SCRIPT${BLUE} ⚡"
echo -e "==============================================${NC}"

echo -e "${YELLOW}🧹 Clearing previous smartgrid_simulation.log file...${NC}"
rm -f "$LOG_FILE"

echo -e "${GREEN}🚀 Starting simulation with parameters:${NC}"
echo -e "${CYAN}   ➤ Prosumers : 5"
echo -e "   ➤ Consumers : 10"
echo -e "   ➤ Difficulty: 4"
echo -e "   ➤ Duration  : 120s${NC}"

echo -e "${BLUE}----------------------------------------------${NC}"

# Run simulation and save to both logs
python main.py \
    --prosumers 5 \
    --consumers 10 \
    --difficulty 4 \
    --duration 120 | tee -a "$LOG_FILE" "$RESULT_FILE"

echo -e "${BLUE}----------------------------------------------${NC}"
echo -e "${GREEN}✅ Simulation completed successfully!${NC}"
echo -e "${YELLOW}📄 Check the log files:${NC}"
echo -e "${CYAN}   - $LOG_FILE (reset each run)"
echo -e "   - $RESULT_FILE (cumulative)${NC}"
echo -e "${BLUE}==============================================${NC}"

# deactivate