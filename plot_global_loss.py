import os
import re
import glob
import matplotlib.pyplot as plt
from collections import defaultdict
import numpy as np

runs = ["C1", "C2", "C3"]
log_dir = "/home/pedro.vidal/facerec_flower/face_rec_fl/face_rec_fl/logs"
output_dir = "/home/pedro.vidal/facerec_flower/face_rec_fl/analysis_plots"
os.makedirs(output_dir, exist_ok=True)

global_reconstructed = {}

for run in runs:
    run_path = os.path.join(log_dir, run)
    client_logs = glob.glob(os.path.join(run_path, "client_*.log"))
    
    round_losses = defaultdict(list)
    round_accs = defaultdict(list)
    
    for clog in client_logs:
        if "eval" in clog: continue
        
        current_round = None
        
        with open(clog, "r", errors='ignore') as f:
            for line in f:
                m_round = re.search(r"for round (\d+)\.\.\.", line)
                if m_round:
                    current_round = int(m_round.group(1))
                
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
    
    # Calculate means
    for r in rounds_sorted:
        avg_losses.append(np.mean(round_losses[r]))
        avg_accs.append(np.mean(round_accs[r]))
        
    global_reconstructed[run] = {
        "rounds": rounds_sorted,
        "loss": avg_losses,
        "acc": avg_accs
    }

colors = {"C1": "royalblue", "C2": "darkorange", "C3": "forestgreen"}

# --- Plot the Reconstructed Global Training Loss ---
plt.figure(figsize=(10, 6))

for run in runs:
    data = global_reconstructed[run]
    if not data["rounds"]: continue
    plt.plot(data["rounds"], data["loss"], label=f"{run} Training Loss", color=colors[run], linewidth=2.5)

plt.title("Global Training Loss", fontsize=14, fontweight='bold')
plt.xlabel("Communication Round", fontsize=12)
plt.ylabel("Average Training Loss", fontsize=12)
plt.xlim(1, 100) # Force X axis to 100
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()

loss_plot_path = os.path.join(output_dir, "global_training_loss.png")
plt.savefig(loss_plot_path, dpi=300)
plt.close()
print(f"Successfully saved {loss_plot_path}")


# --- Plot the Reconstructed Global Training Accuracy ---
plt.figure(figsize=(10, 6))

for run in runs:
    data = global_reconstructed[run]
    if not data["rounds"]: continue
    plt.plot(data["rounds"], data["acc"], label=f"{run} Training Accuracy", color=colors[run], linewidth=2.5)

plt.title("Global Training Accuracy", fontsize=14, fontweight='bold')
plt.xlabel("Communication Round", fontsize=12)
plt.ylabel("Average Training Accuracy", fontsize=12)
plt.xlim(1, 100) # Force X axis to 100
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()

acc_plot_path = os.path.join(output_dir, "global_training_accuracy.png")
plt.savefig(acc_plot_path, dpi=300)
plt.close()
print(f"Successfully saved {acc_plot_path}")
