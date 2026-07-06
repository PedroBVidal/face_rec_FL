import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt

if len(sys.argv) < 2:
    print("Usage: python visualize_layer_updates.py <run_name>")
    exit(1)

run_name = sys.argv[1]
ckpt_dir = f"./checkpoints/{run_name}"

rounds = sorted([int(f.split('_')[-1].split('.')[0]) for f in os.listdir(ckpt_dir) if f.startswith("checkpoint_round_")])

if len(rounds) < 2:
    print(f"[{run_name}] Need at least 2 checkpoints.")
    exit(1)

start_round = rounds[0]
end_round = rounds[-1]

state_start = torch.load(os.path.join(ckpt_dir, f"checkpoint_round_{start_round}.pt"), map_location="cpu")["backbone_state_dict"]
state_end = torch.load(os.path.join(ckpt_dir, f"checkpoint_round_{end_round}.pt"), map_location="cpu")["backbone_state_dict"]

layer_names = []
update_norms = []

weight_keys = [k for k in state_start.keys() if "weight" in k and len(state_start[k].shape) >= 2]
step = max(1, len(weight_keys) // 20)
sampled_keys = weight_keys[::step]

for k in sampled_keys:
    w_start = state_start[k].numpy()
    w_end = state_end[k].numpy()
    
    diff_norm = np.linalg.norm(w_end - w_start)
    base_norm = np.linalg.norm(w_start) + 1e-8
    
    normalized_update = diff_norm / base_norm
    short_name = k.replace('.weight', '').replace('features.', '')
    
    layer_names.append(short_name)
    update_norms.append(normalized_update)

plt.figure(figsize=(12, 6))
bars = plt.bar(layer_names, update_norms, color='teal', alpha=0.7)
plt.xticks(rotation=45, ha='right', fontsize=9)
plt.title(f"Adaptação de Parâmetros por Camada - {run_name} (Rodada {start_round} $\\rightarrow$ {end_round})", fontsize=14, fontweight='bold')
plt.ylabel("Norma de Mudança Relativa $||\\Delta W|| / ||W||$", fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.tight_layout()

os.makedirs("analysis_plots_pt", exist_ok=True)
output_image = f"analysis_plots_pt/{run_name}_layer_wise_updates.png"
plt.savefig(output_image, dpi=300)
print(f"[{run_name}] Plot saved as '{output_image}'")
