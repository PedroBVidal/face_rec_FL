import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

# Configure paths
# In a real run, you might want to point this to a specific round
round_num = 10 
client_updates_dir = f"./checkpoints/client_updates/round_{round_num}"
global_ckpt_path = f"./checkpoints/faces_emore_C1/checkpoint_round_{round_num-1}.pt"

if not os.path.exists(client_updates_dir):
    print(f"Warning: {client_updates_dir} not found. You need to run training first to generate these.")
    exit(0)

# Load the previous global model to compute the pseudo-gradient (update)
print(f"Loading previous global model from round {round_num-1}...")
global_state = torch.load(global_ckpt_path, map_location="cpu")["backbone_state_dict"]

# Choose a representative layer to analyze (e.g., last block)
layer_keys = [k for k in global_state.keys() if "weight" in k and len(global_state[k].shape) == 4]
layer_key = layer_keys[-1]
global_w = global_state[layer_key].numpy().flatten()

client_files = [f for f in os.listdir(client_updates_dir) if f.endswith(".pt")]
if not client_files:
    print(f"No client files found in {client_updates_dir}")
    exit(1)

gradients = []
client_ids = []

print(f"Loading {len(client_files)} client updates...")
for cf in client_files:
    client_id = cf.split('_')[1].split('.')[0]
    ckpt_path = os.path.join(client_updates_dir, cf)
    
    try:
        client_state = torch.load(ckpt_path, map_location="cpu")
        client_w = client_state[layer_key].numpy().flatten()
        
        # Pseudo-gradient: client_weight - global_weight
        grad = client_w - global_w
        gradients.append(grad)
        client_ids.append(client_id)
    except Exception as e:
        print(f"Failed to load {cf}: {e}")

if len(gradients) < 2:
    print("Need at least 2 clients for t-SNE.")
    exit(1)

X = np.stack(gradients)
print(f"Data matrix shape for t-SNE: {X.shape}")

# Apply t-SNE
# Adjust perplexity to be less than the number of samples
perplexity = min(30, len(gradients) - 1)
print(f"Computing t-SNE with perplexity {perplexity}...")
tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
X_tsne = tsne.fit_transform(X)

# Plot t-SNE
plt.figure(figsize=(10, 8))
plt.scatter(X_tsne[:, 0], X_tsne[:, 1], alpha=0.7, c='coral', s=100, edgecolors='w')

# Annotate points with Client IDs
for i, txt in enumerate(client_ids):
    plt.annotate(f"C{txt}", (X_tsne[i, 0], X_tsne[i, 1]), fontsize=8, ha='center', xytext=(0, 5), textcoords='offset points')

plt.title(f"t-SNE of Client Pseudo-Gradients (Round {round_num})", fontsize=14, fontweight='bold')
plt.xlabel("t-SNE Dimension 1")
plt.ylabel("t-SNE Dimension 2")
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()

output_image = f"client_tsne_round_{round_num}.png"
plt.savefig(output_image, dpi=300)
print(f"Plot saved successfully as '{output_image}'")
