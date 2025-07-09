# Smart Grid Blockchain Simulation

A dynamic command-line application for simulating blockchain-based smart grid energy trading environments. This simulation models interactions between prosumers (who both produce and consume energy), consumers, and grid operators within a microgrid environment.

## Features

- **Configurable Node Architecture**: Specify the number of prosumers, consumers, and grid operators
- **Adjustable Mining Parameters**: Set Proof of Work difficulty and toggle GPU acceleration
- **Simulation Control**: Define simulation duration and runtime parameters
- **Database Integration**: Save simulation results to MySQL database
- **Real-time Statistics**: Monitor energy trading and blockchain metrics during simulation
- **Energy Market Simulation**: Includes price dynamics, energy trading, and grid balancing

## Installation

1. Clone this repository:
   ```
   git clone <repository-url>
   cd BlockchainNode
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

   For GPU acceleration (optional):
   ```
   pip install cupy-cuda12x
   ```

## Usage

Run the simulation using command-line arguments:

```bash
python main.py [OPTIONS]
```

### Command Line Options

| Option | Description | Default |
|--------|-------------|---------|
| `--prosumers N` | Number of prosumer nodes | 3 |
| `--consumers N` | Number of consumer nodes | 5 |
| `--operators N` | Number of grid operator nodes (fixed at 1) | 1 |
| `--difficulty N` | Blockchain mining difficulty (number of leading zeros) | 4 |
| `--duration N` | Simulation duration in seconds (0 for infinite) | 300 |
| `--use-gpu` | Use GPU for mining if available | False |
| `--db-host HOST` | MySQL database host | '' (disabled) |
| `--db-port PORT` | MySQL database port | 3306 |
| `--db-user USER` | MySQL database user | 'root' |
| `--db-password PASS` | MySQL database password | '' |
| `--db-name NAME` | MySQL database name | 'smartgrid' |

### Examples

1. Basic simulation with default parameters (3 prosumers, 5 consumers, 5-minute duration):
   ```bash
   python main.py
   ```

2. Extended simulation with more nodes and longer duration:
   ```bash
   python main.py --prosumers 10 --consumers 20 --duration 600
   ```

3. High-difficulty simulation with GPU acceleration:
   ```bash
   python main.py --difficulty 6 --use-gpu
   ```

4. Simulation with database storage:
   ```bash
   python main.py --db-host localhost --db-user myuser --db-password mypassword
   ```

## Architecture

The simulation consists of the following components:

1. **Blockchain Core** (`models/blockchain.py`): Implementation of the blockchain with Proof of Work consensus and optional GPU acceleration.

2. **Smart Grid Nodes** (`models/grid_nodes.py`):
   - `Prosumer`: Entities that both produce and consume energy
   - `Consumer`: Entities that only consume energy
   - `GridOperator`: Entities that manage the grid and facilitate energy trading

3. **Database Utilities** (`utils/db_utils.py`): Tools for storing simulation data in a MySQL database.

4. **Main Simulation** (`main.py`): Orchestrates the simulation and provides the command-line interface.

## Simulation Workflow

1. The simulation initializes a blockchain and creates the specified number of nodes.
2. Prosumer nodes generate energy and may have surplus or deficit.
3. Consumer nodes always need to purchase energy.
4. The grid operator matches energy offers with requests and facilitates trades.
5. Energy transactions are recorded on the blockchain.
6. Statistics are collected and optionally stored in a database.

## Database Schema

When using database integration, the following tables are created:

1. `blocks`: Stores blockchain blocks
2. `transactions`: Stores individual energy and financial transactions
3. `nodes`: Stores information about simulation nodes
4. `simulation_stats`: Stores aggregate statistics about the simulation

## License

This project is licensed under the MIT License - see the LICENSE file for details.