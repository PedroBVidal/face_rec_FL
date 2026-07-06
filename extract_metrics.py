import os
import re
import glob

runs = ["C1", "C2", "C3"]
log_dir = "/home/pedro.vidal/facerec_flower/face_rec_fl/face_rec_fl/logs"

global_metrics = {}
client_accuracies = {}

for run in runs:
    run_path = os.path.join(log_dir, run)
    server_log = os.path.join(run_path, "server.log")
    
    global_metrics[run] = {"rounds": [], "loss": [], "acc": [], "agedb": [], "cfpfp": [], "lfw": []}
    
    if os.path.exists(server_log):
        with open(server_log, "r", errors='ignore') as f:
            for line in f:
                if "Aggregated Client Metrics" in line:
                    loss_match = re.search(r"Loss: ([\d\.]+)", line)
                    acc_match = re.search(r"Accuracy: ([\d\.]+)", line)
                    if loss_match and acc_match:
                        global_metrics[run]["loss"].append(float(loss_match.group(1)))
                        global_metrics[run]["acc"].append(float(acc_match.group(1)))
                
                if "agedb_30 Accuracy:" in line:
                    m = re.search(r"Accuracy: ([\d\.]+)", line)
                    if m: global_metrics[run]["agedb"].append(float(m.group(1)))
                if "cfp_fp Accuracy:" in line:
                    m = re.search(r"Accuracy: ([\d\.]+)", line)
                    if m: global_metrics[run]["cfpfp"].append(float(m.group(1)))
                if "lfw Accuracy:" in line:
                    m = re.search(r"Accuracy: ([\d\.]+)", line)
                    if m: global_metrics[run]["lfw"].append(float(m.group(1)))

    client_logs = glob.glob(os.path.join(run_path, "client_*.log"))
    client_accuracies[run] = {}
    
    for clog in client_logs:
        if "eval" in clog: continue
        
        match = re.search(r"client_(\d+)\.log", clog)
        if match:
            pid = int(match.group(1))
            last_acc = None
            with open(clog, "r", errors='ignore') as f:
                for line in f:
                    if "Accuracy:" in line:
                        acc_match = re.search(r"Accuracy: ([\d\.]+)", line)
                        if acc_match:
                            last_acc = float(acc_match.group(1))
            if last_acc is not None:
                client_accuracies[run][pid] = last_acc

print("# Global Models Metrics (Last Round)")
print("| Run | Final Train Loss | Final Train Acc | AgeDB-30 | CFP-FP | LFW |")
print("|---|---|---|---|---|---|")
for run in runs:
    gm = global_metrics[run]
    loss = f"{gm['loss'][-1]:.4f}" if gm['loss'] else "N/A"
    tacc = f"{gm['acc'][-1]:.4f}" if gm['acc'] else "N/A"
    agedb = f"{gm['agedb'][-1]:.4f}" if gm['agedb'] else "N/A"
    cfpfp = f"{gm['cfpfp'][-1]:.4f}" if gm['cfpfp'] else "N/A"
    lfw = f"{gm['lfw'][-1]:.4f}" if gm['lfw'] else "N/A"
    print(f"| {run} | {loss} | {tacc} | {agedb} | {cfpfp} | {lfw} |")

print("\n# Sorted Client Classification Accuracies (Top 5 & Bottom 5 per run)")
for run in runs:
    print(f"\n### Run: {run}")
    ca = client_accuracies[run]
    if not ca:
        print("No client evaluation logs found.")
        continue
    
    sorted_clients = sorted(ca.items(), key=lambda item: item[1], reverse=True)
    
    print("| Rank | Client ID | Classification Accuracy |")
    print("|---|---|---|")
    
    for rank, (pid, acc) in enumerate(sorted_clients[:5], 1):
        print(f"| {rank} | Client {pid} | {acc:.4f} |")
        
    print("| ... | ... | ... |")
    
    for rank, (pid, acc) in enumerate(sorted_clients[-5:], len(sorted_clients) - 4):
        print(f"| {rank} | Client {pid} | {acc:.4f} |")
