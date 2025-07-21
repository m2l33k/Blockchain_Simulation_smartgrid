import re
import json
from datetime import datetime, timedelta
from collections import defaultdict

def parse_log_file(log_file_path):
    """
    Parses the simulation log file to extract anomalies, detections, and blockchain data.
    """
    log_pattern = re.compile(
        r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3}) - ([\w.-]+) - (\w+) - (.*)'
    )

    anomalies = []
    detections = []
    blocks = []
    
    last_detection_check = {}

    with open(log_file_path, 'r') as f:
        for line in f:
            match = log_pattern.match(line)
            if not match:
                continue

            timestamp_str, logger, level, message = match.groups()
            timestamp = datetime.strptime(timestamp_str, '%Y-%m-%d %H:%M:%S,%f')

            if '!!! ANOMALY: Injecting' in message:
                anomaly_type = "Unknown"
                details = {}
                if 'NODE BREAKDOWN' in message:
                    anomaly_type = "Node Breakdown"
                    m = re.search(r'on ([\w-]+)\.', message)
                    if m: details['node'] = m.group(1)
                elif 'METER TAMPERING' in message:
                    anomaly_type = "Meter Tampering"
                    m = re.search(r'on ([\w-]+)', message)
                    if m: details['node'] = m.group(1)
                elif 'COORDINATED INAUTHENTIC TRADING' in message:
                    anomaly_type = "Wash Trading"
                    m = re.search(r'between ([\w-]+) and ([\w-]+)', message)
                    if m: details['nodes'] = [m.group(1), m.group(2)]
                anomalies.append({'type': anomaly_type, 'start_time': timestamp, 'details': details, 'end_time': None})

            elif '!!! ANOMALY:' in message:
                anomaly_type = "Unknown"
                details = {}
                if 'DoS ATTACK' in message:
                    anomaly_type = "DoS Attack"
                    m = re.search(r'([\w-]+) beginning', message)
                    if m: details['node'] = m.group(1)
                elif 'THEFT' in message:
                    anomaly_type = "Theft"
                    m = re.search(r'([\w-]+) attempting THEFT of \$([\d.]+) from ([\w-]+)', message)
                    if m: details = {'attacker': m.group(1), 'amount': float(m.group(2)), 'victim': m.group(3)}
                anomalies.append({'type': anomaly_type, 'start_time': timestamp, 'details': details, 'end_time': None})
            
            elif '--- ANOMALY END:' in message:
                ended_anomaly_node = None
                if 'DoS attack from' in message:
                    m = re.search(r'from ([\w-]+)', message)
                    if m: ended_anomaly_node = m.group(1)
                elif 'Coordinated trading between' in message:
                    m = re.search(r'between ([\w-]+) and ([\w-]+)', message)
                    if m: ended_anomaly_node = sorted([m.group(1), m.group(2)])

                for anomaly in reversed(anomalies):
                    if anomaly['end_time'] is None:
                        nodes_in_anomaly = []
                        if 'node' in anomaly['details']: nodes_in_anomaly = [anomaly['details']['node']]
                        if 'nodes' in anomaly['details']: nodes_in_anomaly = sorted(anomaly['details']['nodes'])
                        
                        if ended_anomaly_node and nodes_in_anomaly and (ended_anomaly_node == nodes_in_anomaly or ended_anomaly_node == nodes_in_anomaly[0]):
                             anomaly['end_time'] = timestamp
                             break

            elif 'Detector check on Block' in message:
                m = re.search(r'Block #(\d+): Score = ([\d.]+)', message)
                if m:
                    block_id, score = m.groups()
                    last_detection_check[int(block_id)] = float(score)

            elif '!!! LIVE ALERT: Anomaly detected' in message:
                m = re.search(r'in Block #(\d+)', message)
                if m:
                    block_id = int(m.group(1))
                    score = last_detection_check.get(block_id, 'N/A')
                    detections.append({
                        'timestamp': timestamp,
                        'block_id': block_id,
                        'score': score
                    })

            elif message.startswith('MINED_BLOCK:'):
                try:
                    json_str = message.replace('MINED_BLOCK: ', '')
                    block_data = json.loads(json_str)
                    blocks.append(block_data)
                except json.JSONDecodeError:
                    print(f"Warning: Could not parse JSON from line: {line.strip()}")

    return anomalies, detections, blocks

# <-- CHANGE: The function now accepts an output_filename
def analyze_results(anomalies, detections, blocks, output_filename):
    """
    Analyzes the parsed data and generates a report string.
    """
    report_lines = [] # <-- CHANGE: We build a list of lines instead of printing

    report_lines.append("="*80)
    report_lines.append("Simulation Log Analysis Report")
    report_lines.append("="*80)

    detected_anomaly_indices = set()

    for i, anomaly in enumerate(anomalies):
        anomaly_start = anomaly['start_time']
        anomaly_end = anomaly['end_time'] or (anomaly_start + timedelta(seconds=15))

        transaction_type_map = {
            "Theft": "fraudulent_payment",
            "Wash Trading": "wash_trade_payment",
            "DoS Attack": "spam_peak"
        }
        anomaly_tx_type = transaction_type_map.get(anomaly['type'])
        
        for detection in detections:
            if anomaly_start <= detection['timestamp'] <= anomaly_end:
                 detected_anomaly_indices.add(i)
                 anomaly['detection_time'] = detection['timestamp']
                 anomaly['detection_latency'] = detection['timestamp'] - anomaly['start_time']
                 break
        
        if i not in detected_anomaly_indices and anomaly_tx_type:
             for block in blocks:
                 for tx in block.get('transactions', []):
                     if tx.get('type') == anomaly_tx_type:
                         for detection in detections:
                            if detection['block_id'] == block['index']:
                                detected_anomaly_indices.add(i)
                                anomaly['detection_time'] = detection['timestamp']
                                anomaly['detection_latency'] = detection['timestamp'] - anomaly['start_time']
                                break
                 if i in detected_anomaly_indices:
                     break

    report_lines.append("\n--- Overall Performance ---\n")
    total_anomalies = len(anomalies)
    report_lines.append(f"Total Anomalies Injected: {total_anomalies}")
    report_lines.append(f"Anomalies Detected: {len(detected_anomaly_indices)}")
    if total_anomalies > 0:
        detection_rate = (len(detected_anomaly_indices) / total_anomalies) * 100
        report_lines.append(f"Overall Detection Rate: {detection_rate:.2f}%")
    
    latencies = [a['detection_latency'].total_seconds() for a in anomalies if 'detection_latency' in a]
    if latencies:
        avg_latency = sum(latencies) / len(latencies)
        report_lines.append(f"Average Detection Latency: {avg_latency:.2f} seconds")

    report_lines.append("\n\n" + "--- Anomaly Injection Details ---" + "\n")
    anomaly_counts = defaultdict(int)
    for i, anomaly in enumerate(anomalies):
        anomaly_counts[anomaly['type']] += 1
        status = "DETECTED" if i in detected_anomaly_indices else "MISSED"
        duration_str = ""
        if anomaly['end_time']:
            duration = (anomaly['end_time'] - anomaly['start_time']).total_seconds()
            duration_str = f" (Duration: {duration:.2f}s)"
        
        latency_str = ""
        if 'detection_latency' in anomaly:
            latency_str = f" (Latency: {anomaly['detection_latency'].total_seconds():.2f}s)"

        report_lines.append(f"[{status}] {anomaly['start_time'].strftime('%H:%M:%S')} - {anomaly['type']}{duration_str}{latency_str}")
        details = anomaly.get('details', {})
        if 'node' in details: report_lines.append(f"    - Target Node: {details['node']}")
        if 'nodes' in details: report_lines.append(f"    - Involved Nodes: {', '.join(details['nodes'])}")
        if 'attacker' in details: report_lines.append(f"    - Details: Attacker {details['attacker']}, Victim {details['victim']}, Amount ${details['amount']:.2f}")

    report_lines.append("\nAnomaly Counts by Type:")
    for type, count in anomaly_counts.items():
        report_lines.append(f"- {type}: {count}")

    report_lines.append("\n\n" + "--- Detection Alert Details ---" + "\n")
    if not detections:
        report_lines.append("No detection alerts were triggered.")
    else:
        report_lines.append(f"{'Time':<10} {'Block ID':<10} {'Confidence Score':<20}")
        report_lines.append("-"*45)
        for detection in detections:
            score_str = f"{detection['score']:.2%}" if isinstance(detection['score'], float) else 'N/A'
            report_lines.append(f"{detection['timestamp'].strftime('%H:%M:%S'):<10} {detection['block_id']:<10} {score_str:<20}")

    report_lines.append("\n" + "="*80)

    # <-- CHANGE: Join all lines and write to the specified file
    final_report = "\n".join(report_lines)
    with open(output_filename, 'w') as f:
        f.write(final_report)
    
    # <-- CHANGE: Print a confirmation message to the console
    print(f"Analysis complete. Report saved to '{output_filename}'")


if __name__ == "__main__":
    LOG_FILE = 'live_detection_run.log'
    REPORT_FILE = 'analysis_report.txt' # <-- CHANGE: Define the output filename

    try:
        anomalies, detections, blocks = parse_log_file(LOG_FILE)
        # <-- CHANGE: Pass the filename to the function
        analyze_results(anomalies, detections, blocks, output_filename=REPORT_FILE) 
    except FileNotFoundError:
        print(f"Error: The log file '{LOG_FILE}' was not found.")
        print("Please save the log content to this file and run the script again.")