# Smart Grid Blockchain Simulation

A decentralized energy trading platform that simulates a smart grid system using blockchain technology to securely record and verify energy transactions between different types of nodes.

## Overview

This simulation creates a virtual power grid with prosumers, consumers, and a grid operator that interact to trade energy in a decentralized marketplace. Energy transactions are recorded on a blockchain, providing a transparent and immutable ledger.

## Features

- **Decentralized Energy Trading**: Peer-to-peer energy transactions between grid participants
- **Blockchain Implementation**: Secure transaction verification with customizable mining difficulty
- **Dynamic Node Behavior**: Prosumers and consumers with realistic energy production/consumption patterns
- **Grid Management**: Central grid operator that facilitates trades and maintains grid stability
- **Real-time Simulation**: Time-accelerated simulation with configurable duration
- **Comprehensive Logging**: Detailed transaction and system state logging

## System Components

- **Prosumers**: Entities that both produce and consume energy (e.g., houses with solar panels)
- **Consumers**: Entities that only consume energy from the grid
- **Grid Operator**: Central entity that manages the grid and facilitates energy trading
- **Blockchain**: Distributed ledger that records all energy transactions

## Running the Simulation

Use the included shell script to run the simulation with default parameters:

```bash
./run.sh
```

Or run manually with custom parameters:

```bash
python main.py --prosumers 5 --consumers 10 --difficulty 4 --duration 360
```

### Parameters

- `--prosumers`: Number of prosumer nodes (default: 5)
- `--consumers`: Number of consumer nodes (default: 10)
- `--difficulty`: Blockchain mining difficulty (default: 4)
- `--duration`: Simulation duration in seconds (default: 120)

## Logging

All simulation activities are logged to both the console and `smartgrid_simulation.log` file. The log includes:

- Blockchain operations (block mining, transaction validation)
- Energy trades between nodes
- Node status updates
- System-wide statistics

## Development

### Requirements

- Python 3.8+
- Required packages listed in `requirements.txt`

### Project Structure

- `main.py`: Entry point and simulation orchestration
- `models/`: Core system models
  - `blockchain.py`: Blockchain implementation
  - `grid_nodes.py`: Node implementations (Prosumer, Consumer, GridOperator)
- `utils/`: Utility functions
- `run.sh`: Convenience script for running the simulation

## License

This project is licensed under the MIT License - see the LICENSE file for details.