# 1_🏠_Overview.py (Final Version: Automatically finds the latest log)

import streamlit as st
import os
import re
from streamlit_agraph import agraph, Node, Edge, Config
import glob

# --- Page Configuration ---
st.set_page_config(
    layout="wide",
    page_title="Smart Grid Dashboard",
    page_icon="⚡️"
)

# --- Constants ---
LOG_DIRECTORY = 'simulation_logs'

# --- Helper Functions ---
def find_latest_log_file(directory: str) -> str | None:
    """Finds the most recent simulation log file in the specified directory."""
    if not os.path.isdir(directory):
        st.error(f"Log directory not found: '{directory}'")
        return None
    
    # Use glob to find all files matching the simulation run pattern
    log_files = glob.glob(os.path.join(directory, 'simulation_run_*.log'))
    
    if not log_files:
        return None
        
    # Find the most recently modified file from the list
    latest_file = max(log_files, key=os.path.getmtime)
    return latest_file

@st.cache_data
def load_all_grid_nodes_from_log(log_path: str) -> list | None:
    """
    Parses a specific simulation log file to find all registered nodes.
    """
    if not log_path or not os.path.exists(log_path):
        return None
    
    node_pattern = re.compile(r"Node (CON-\d+|PRO-\d+|GRID-OP-\d+) registered")
    registered_nodes = set()
    
    try:
        with open(log_path, 'r', encoding='utf-8') as f:
            for line in f:
                match = node_pattern.search(line)
                if match:
                    registered_nodes.add(match.group(1))
    except Exception as e:
        st.error(f"Error reading log file '{os.path.basename(log_path)}': {e}")
        return None

    return sorted(list(registered_nodes))

def create_agraph_data(nodes):
    """Creates the nodes and edges for the streamlit-agraph component."""
    if not nodes:
        return [], []

    graph_nodes = []
    graph_edges = []
    grid_op_node_id = next((node for node in nodes if 'GRID-OP' in node), None)

    for node_id in nodes:
        if 'GRID-OP' in node_id:
            graph_nodes.append(Node(id=node_id, label=node_id, color='#f0ad4e', size=25, shape="star", title="Grid Operator"))
        elif 'PRO' in node_id:
            graph_nodes.append(Node(id=node_id, label=node_id, color='#5bc0de', size=15, title="Prosumer"))
        elif 'CON' in node_id:
            graph_nodes.append(Node(id=node_id, label=node_id, color='#d9534f', size=10, title="Consumer"))

    if grid_op_node_id:
        for node_id in nodes:
            if node_id != grid_op_node_id:
                graph_edges.append(Edge(source=node_id, target=grid_op_node_id, color='lightgray'))
    
    return graph_nodes, graph_edges

# --- Main Page ---
st.title("⚡️ Smart Grid Overview & Topology")
st.markdown("Welcome to the central dashboard for monitoring and analyzing the smart grid simulation.")

# --- Find and Load Data from the LATEST Log ---
latest_log_file = find_latest_log_file(LOG_DIRECTORY)

if latest_log_file is None:
    st.error(f"No simulation log files found in the '{LOG_DIRECTORY}' directory. Please run a `generate` simulation first.")
    st.stop()

# Display which file is being used for transparency
st.info(f"Displaying data from the latest simulation log: **{os.path.basename(latest_log_file)}**")

grid_nodes = load_all_grid_nodes_from_log(latest_log_file)

if not grid_nodes:
    st.warning("Could not find any registered nodes in the latest log file. The simulation might not have initialized correctly.")
    st.stop()

# --- Display Key Metrics ---
st.header("Grid Statistics")
consumers = [node for node in grid_nodes if 'CON' in node]
prosumers = [node for node in grid_nodes if 'PRO' in node]
operators = [node for node in grid_nodes if 'GRID-OP' in node]

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Nodes", len(grid_nodes))
col2.metric("Consumers", len(consumers))
col3.metric("Prosumers", len(prosumers))
col4.metric("Grid Operators", len(operators))

# --- Display Interactive Graph ---
st.header("Interactive Grid Topology")
st.markdown("This graph shows the connections between all registered nodes in the simulation. Pan, zoom, and drag nodes to explore the network.")
nodes, edges = create_agraph_data(grid_nodes)
config = Config(width=1200, height=600, directed=False, physics=True, hierarchical=False)
agraph(nodes=nodes, edges=edges, config=config)