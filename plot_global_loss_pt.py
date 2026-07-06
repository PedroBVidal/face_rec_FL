import os
import re
import glob
import matplotlib.pyplot as plt
from collections import defaultdict
import numpy as np

runs = ["C1", "C2", "C3"]
log_dir = "/home/pedro.vidal/facerec_flower/face_rec_fl/face_rec_fl/logs"
output_dir = "/home/pedro.vidal/facerec_flower/face_rec_fl/analysis_plots_pt"
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
    plt.plot(data["rounds"], data["loss"], label=f"Perda de Treinamento {run}", color=colors[run], linewidth=2.5)

plt.title("Perda de Treinamento Global", fontsize=14, fontweight='bold')
plt.xlabel("Rodada de Comunicação", fontsize=12)
plt.ylabel("Perda de Treinamento Média", fontsize=12)
plt.xlim(1, 100) # Force X axis to 100
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()

loss_plot_path = os.path.join(output_dir, "global_training_loss.png")
plt.savefig(loss_plot_path, dpi=300)
plt.close()
print(f"Successfully saved {loss_plot_path}")

# --- Parse Global Verification Accuracy from server.log ---
global_verif_acc = {}

for run in runs:
    run_path = os.path.join(log_dir, run)
    server_log = os.path.join(run_path, "server.log")

    agedb = []
    cfpfp = []
    lfw = []

    if os.path.exists(server_log):
        with open(server_log, "r", errors='ignore') as f:
            for line in f:
                if "agedb_30 Accuracy:" in line:
                    m = re.search(r"Accuracy: ([\d\.]+)", line)
                    if m: agedb.append(float(m.group(1)))
                if "cfp_fp Accuracy:" in line:
                    m = re.search(r"Accuracy: ([\d\.]+)", line)
                    if m: cfpfp.append(float(m.group(1)))
                if "lfw Accuracy:" in line:
                    m = re.search(r"Accuracy: ([\d\.]+)", line)
                    if m: lfw.append(float(m.group(1)))

    # Truncate to max_round_trained (from loss parsing above)
    max_round = max(global_reconstructed[run]["rounds"]) if global_reconstructed[run]["rounds"] else 0
    if max_round > 0:
        agedb = agedb[:max_round]
        cfpfp = cfpfp[:max_round]
        lfw = lfw[:max_round]

    # Average the three benchmarks per round
    n_rounds = min(len(agedb), len(cfpfp), len(lfw))
    avg_verif = [np.mean([agedb[i], cfpfp[i], lfw[i]]) for i in range(n_rounds)]

    global_verif_acc[run] = {
        "rounds": list(range(1, n_rounds + 1)),
        "acc": avg_verif,
    }

# --- Plot the Global Verification Accuracy ---
plt.figure(figsize=(10, 6))

for run in runs:
    data = global_verif_acc[run]
    if not data["rounds"]: continue
    plt.plot(data["rounds"], data["acc"], label=f"Acurácia de Verificação {run}", color=colors[run], linewidth=2.5)

plt.title("Acurácia de Verificação Global (Média AgeDB-30, CFP-FP, LFW)", fontsize=14, fontweight='bold')
plt.xlabel("Rodada de Comunicação", fontsize=12)
plt.ylabel("Acurácia de Verificação Média", fontsize=12)
plt.xlim(1, 100) # Force X axis to 100
plt.ylim(0, 1.0)  # Fixed Y axis for all runs
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()

acc_plot_path = os.path.join(output_dir, "global_training_accuracy.png")
plt.savefig(acc_plot_path, dpi=300)
plt.close()
print(f"Successfully saved {acc_plot_path}")
