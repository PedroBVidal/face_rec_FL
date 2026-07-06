import os
import re
import glob
import matplotlib.pyplot as plt

runs = ["C1", "C2", "C3"]
log_dir = "/home/pedro.vidal/facerec_flower/face_rec_fl/face_rec_fl/logs"
output_dir = "/home/pedro.vidal/facerec_flower/face_rec_fl/analysis_plots_pt"
os.makedirs(output_dir, exist_ok=True)

global_metrics = {}
client_accuracies = {}

# --- 1. Parse Logs ---
for run in runs:
    run_path = os.path.join(log_dir, run)
    server_log = os.path.join(run_path, "server.log")
    
    global_metrics[run] = {"agedb": [], "cfpfp": [], "lfw": []}
    
    if os.path.exists(server_log):
        with open(server_log, "r", errors='ignore') as f:
            for line in f:
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
    client_accuracies[run] = []
    max_round_trained = 0
    
    for clog in client_logs:
        if "eval" in clog: continue
        
        last_acc = None
        with open(clog, "r", errors='ignore') as f:
            for line in f:
                if "Accuracy:" in line:
                    acc_match = re.search(r"Accuracy: ([\d\.]+)", line)
                    if acc_match:
                        last_acc = float(acc_match.group(1))
                m_round = re.search(r"for round (\d+)\.\.\.", line)
                if m_round:
                    max_round_trained = max(max_round_trained, int(m_round.group(1)))
                    
        if last_acc is not None:
            client_accuracies[run].append(last_acc)
            
    # Truncate global evaluation metrics to the round where early stopping kicked in
    if max_round_trained > 0:
        for key in ["agedb", "cfpfp", "lfw"]:
            global_metrics[run][key] = global_metrics[run][key][:max_round_trained]
            
    client_accuracies[run].sort(reverse=True)


# --- 2. Plot Global Model Metrics Across Epochs/Rounds SEPARATELY ---
datasets = [("AgeDB-30", "agedb"), ("CFP-FP", "cfpfp"), ("LFW", "lfw")]
colors = {"C1": "royalblue", "C2": "darkorange", "C3": "forestgreen"}

for ds_name, ds_key in datasets:
    plt.figure(figsize=(10, 6))
    for run in runs:
        y_data = global_metrics[run][ds_key]
        if not y_data: continue
        x_data = list(range(1, len(y_data) + 1))
        plt.plot(x_data, y_data, label=run, color=colors[run], linewidth=2.5)

    plt.title(f"Acurácia do Modelo Global em {ds_name}", fontsize=14, fontweight='bold')
    plt.xlabel("Rodada de Comunicação", fontsize=12)
    plt.ylabel("Acurácia", fontsize=12)
    plt.xlim(1, 100) # Force X axis to 100
    plt.ylim(0, 1.0)  # Fixed Y axis for all runs
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.tight_layout()

    out_path = os.path.join(output_dir, f"global_accuracy_{ds_key}.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Successfully saved {out_path}")

# --- 3. Plot Sorted Client Final Epoch Accuracies ---
plt.figure(figsize=(10, 6))

for run in runs:
    y_data = client_accuracies[run]
    if not y_data: continue
    x_data = [i / len(y_data) * 100 for i in range(len(y_data))]
    plt.plot(x_data, y_data, label=f"{run} ({len(y_data)} clientes)", color=colors[run], linewidth=2.5)

plt.title("Acurácias Finais de Classificação dos Clientes Ordenadas", fontsize=14, fontweight='bold')
plt.xlabel("Percentil de Cliente (Ordenado do Melhor para o Pior)", fontsize=12)
plt.ylabel("Acurácia Final de Classificação Local", fontsize=12)
plt.ylim(0, 1.0)  # Fixed Y axis for all runs
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()

client_plot_path = os.path.join(output_dir, "sorted_client_accuracies.png")
plt.savefig(client_plot_path, dpi=300)
plt.close()
print(f"Successfully saved {client_plot_path}")
