import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt

if len(sys.argv) < 2:
    print("Usage: python visualize_drift.py <run_name>")
    exit(1)

run_name = sys.argv[1]
ckpt_dir = f"./checkpoints/{run_name}"

rounds = sorted([int(f.split('_')[-1].split('.')[0]) for f in os.listdir(ckpt_dir) if f.startswith("checkpoint_round_")])

if len(rounds) < 2:
    print(f"[{run_name}] Need at least 2 checkpoints to measure drift.")
    exit(1)

drift_l2 = []
round_labels = []

prev_weights = None

for r in rounds:
    ckpt_path = os.path.join(ckpt_dir, f"checkpoint_round_{r}.pt")
    state = torch.load(ckpt_path, map_location="cpu")["backbone_state_dict"]
    
    current_weights = np.concatenate([v.numpy().flatten() for k, v in state.items() if "weight" in k])
    
    if prev_weights is not None:
        l2_dist = np.linalg.norm(current_weights - prev_weights)
        drift_l2.append(l2_dist)
        round_labels.append(r)
        
    prev_weights = current_weights

plt.figure(figsize=(10, 5))
plt.plot(round_labels, drift_l2, marker='o', linestyle='-', color='purple', linewidth=2)
plt.title(f"Divergência de Peso do Modelo Global (Desvio do Cliente) - {run_name}", fontsize=14, fontweight='bold')
plt.xlabel("Rodada de Comunicação", fontsize=12)
plt.ylabel("Norma L2 da Atualização Global $||w^{(t)} - w^{(t-1)}||_2$", fontsize=12)
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()

os.makedirs("analysis_plots_pt", exist_ok=True)
output_image = f"analysis_plots_pt/{run_name}_global_weight_drift.png"
plt.savefig(output_image, dpi=300)
print(f"[{run_name}] Plot saved as '{output_image}'")
