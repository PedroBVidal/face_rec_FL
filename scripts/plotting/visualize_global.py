import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

if len(sys.argv) < 2:
    print("Usage: python visualize_global.py <run_name>")
    exit(1)

run_name = sys.argv[1]
ckpt_dir = f"./checkpoints/{run_name}"

if not os.path.exists(ckpt_dir):
    print(f"Error: checkpoint directory '{ckpt_dir}' not found.")
    exit(1)

rounds = sorted([int(f.split('_')[-1].split('.')[0]) for f in os.listdir(ckpt_dir) if f.startswith("checkpoint_round_")])

if not rounds:
    print(f"No checkpoint files found in '{ckpt_dir}' matching 'checkpoint_round_*.pt'")
    exit(1)

weights = []
valid_rounds = []

print(f"[{run_name}] Found {len(rounds)} checkpoints. Loading...")
for r in rounds:
    ckpt_path = os.path.join(ckpt_dir, f"checkpoint_round_{r}.pt")
    try:
        state = torch.load(ckpt_path, map_location="cpu")["backbone_state_dict"]
        layer_keys = [k for k in state.keys() if "weight" in k and len(state[k].shape) == 4]
        layer_key = layer_keys[-1] if layer_keys else list(state.keys())[0]
            
        flat_w = state[layer_key].numpy().flatten()
        weights.append(flat_w)
        valid_rounds.append(r)
    except Exception as e:
        pass

if len(weights) < 2:
    print(f"[{run_name}] Error: Need at least 2 valid checkpoints to run PCA.")
    exit(1)

X = np.stack(weights)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

plt.figure(figsize=(10, 6))
plt.plot(X_pca[:, 0], X_pca[:, 1], '-o', color='royalblue', label=f'{run_name} Path', linewidth=1.5, markersize=5)
plt.scatter(X_pca[0, 0], X_pca[0, 1], color='green', s=150, zorder=5, label=f'Start (Round {valid_rounds[0]})')
plt.scatter(X_pca[-1, 0], X_pca[-1, 1], color='red', s=150, zorder=5, label=f'End (Round {valid_rounds[-1]})')

for idx, r in enumerate(valid_rounds):
    if r % 5 == 0 or r == valid_rounds[0] or r == valid_rounds[-1]:
        plt.annotate(f"R{r}", (X_pca[idx, 0], X_pca[idx, 1]), textcoords="offset points", xytext=(0, 10), ha='center', fontsize=8, fontweight='bold')

plt.title(f"PCA Global Trajectory - {run_name}", fontsize=14, fontweight='bold')
plt.xlabel("Principal Component 1", fontsize=12)
plt.ylabel("Principal Component 2", fontsize=12)
plt.legend(loc='best')
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()

os.makedirs("analysis_plots", exist_ok=True)
output_image = f"analysis_plots/{run_name}_global_trajectory_pca.png"
plt.savefig(output_image, dpi=300)
print(f"[{run_name}] Plot saved as '{output_image}'")
