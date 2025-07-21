#!/bin/bash


if [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
    echo -e "\n✅ Virtual environment activated."
fi

SIM_DURATION=600  
SIM_PROSUMERS=10
SIM_CONSUMERS=20
SIM_DIFFICULTY=3


GREEN='\033[1;32m'
YELLOW='\033[1;33m'
BLUE='\033[1;34m'
CYAN='\033[1;36m'
RED='\033[0;31m'
NC='\033[0m' 

print_header() {
    echo -e "\n${BLUE}==============================================================${NC}"
    echo -e "${CYAN} $1 ${NC}"
    echo -e "${BLUE}==============================================================${NC}"
}


check_success() {
    if [ $? -ne 0 ]; then
        echo -e "\n${RED}❌ ERROR: The last step failed. Aborting pipeline.${NC}\n"
        exit 1
    fi
    echo -e "${GREEN}✅ Success!${NC}"
}


print_header "STEP 0: CLEANING UP PREVIOUS RUNS"
echo -e "${YELLOW}🧹 Clearing previous log, data, and model files...${NC}"
#rm -f smartgrid_simulation.log
#rm -rf data/
#rm -rf saved_models/
echo -e "${GREEN}✅ Cleanup complete.${NC}"

print_header "STEP 1: RUNNING SIMULATION TO GENERATE LOG DATA"
echo "   - Duration  : ${SIM_DURATION} seconds"
echo "   - Prosumers : ${SIM_PROSUMERS}"
echo "   - Consumers : ${SIM_CONSUMERS}"
echo "   - Difficulty: ${SIM_DIFFICULTY}"
echo -e "${YELLOW}🚀 Starting simulation... (This will take ${SIM_DURATION} seconds)${NC}"

python main.py \
    --prosumers ${SIM_PROSUMERS} \
    --consumers ${SIM_CONSUMERS} \
    --difficulty ${SIM_DIFFICULTY} \
    --duration ${SIM_DURATION}

check_success


print_header "STEP 2: PREPROCESSING LOG DATA"
echo -e "${YELLOW}⚙️ Parsing 'data_generation_run.log' to create training CSV...${NC}"
python data_preprocessor.py
check_success

print_header "STEP 3: TRAINING THE ANOMALY DETECTION MODEL"
echo -e "${YELLOW}🧠 Training model using data from 'data/featurized_labeled_data.csv'...${NC}"
echo "   (This may take several minutes depending on your hardware)"
python train_model.py
check_success


print_header "PIPELINE COMPLETED SUCCESSFULLY"
echo -e "A new, trained anomaly detection model is now available in the ${GREEN}'saved_models/'${NC} directory."
echo -e "You can now run the live detection script against a running simulation."
echo -e "${BLUE}==============================================================${NC}\n"