import os
import re
import glob
import json
from collections import defaultdict

runs = ["C1", "C2", "C3"]
log_dir = "/home/pedro.vidal/facerec_flower/face_rec_fl/face_rec_fl/logs"
output_file = "/home/pedro.vidal/facerec_flower/face_rec_fl/metrics.json"

final_metrics = {}

for run in runs:
    run_path = os.path.join(log_dir, run)
    server_log = os.path.join(run_path, "server.log")
    
    # 1. Parse evaluation metrics from server.log
    eval_agedb = []
    eval_cfpfp = []
    eval_lfw = []
    
    if os.path.exists(server_log):
        with open(server_log, "r", errors='ignore') as f:
            for line in f:
                if "agedb_30 Accuracy:" in line:
                    m = re.search(r"Accuracy: ([\d\.]+)", line)
                    if m: eval_agedb.append(float(m.group(1)))
                if "cfp_fp Accuracy:" in line:
                    m = re.search(r"Accuracy: ([\d\.]+)", line)
                    if m: eval_cfpfp.append(float(m.group(1)))
                if "lfw Accuracy:" in line:
                    m = re.search(r"Accuracy: ([\d\.]+)", line)
                    if m: eval_lfw.append(float(m.group(1)))
                    
    # 2. Parse training metrics from client logs
    client_logs = glob.glob(os.path.join(run_path, "client_*.log"))
    
    round_losses = defaultdict(list)
    round_accs = defaultdict(list)
    max_round_trained = 0
    
    for clog in client_logs:
        if "eval" in clog: continue
        
        current_round = None
        with open(clog, "r", errors='ignore') as f:
            for line in f:
                m_round = re.search(r"for round (\d+)\.\.\.", line)
                if m_round:
                    current_round = int(m_round.group(1))
                    max_round_trained = max(max_round_trained, current_round)
                
                m_metrics = re.search(r"Training finished\. Loss: ([\d\.]+), Accuracy: ([\d\.]+)", line)
                if m_metrics and current_round is not None:
                    loss = float(m_metrics.group(1))
                    acc = float(m_metrics.group(2))
                    round_losses[current_round].append(loss)
                    round_accs[current_round].append(acc)
                    current_round = None

    avg_losses = []
    avg_accs = []
    rounds_sorted = sorted(round_losses.keys())
    
    for r in rounds_sorted:
        avg_losses.append(sum(round_losses[r]) / len(round_losses[r]))
        avg_accs.append(sum(round_accs[r]) / len(round_accs[r]))
        
    # Truncate evaluation metrics to align with the maximum rounds trained 
    # (to match plot_metrics.py logic for early stopping)
    if max_round_trained > 0:
        eval_agedb = eval_agedb[:max_round_trained]
        eval_cfpfp = eval_cfpfp[:max_round_trained]
        eval_lfw = eval_lfw[:max_round_trained]

    final_metrics[run] = {
        "rounds": rounds_sorted,
        "train_loss": avg_losses,
        "train_acc": avg_accs,
        "eval_agedb": eval_agedb,
        "eval_cfpfp": eval_cfpfp,
        "eval_lfw": eval_lfw
    }

with open(output_file, "w") as f:
    json.dump(final_metrics, f, indent=4)

print(f"Metrics successfully exported to {output_file}")
