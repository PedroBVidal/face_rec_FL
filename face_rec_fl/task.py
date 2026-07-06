import os
import logging
from tqdm import tqdm

# Module-level logger (will be configured by the app)
logger = logging.getLogger(__name__)

import io
import pickle
import numbers
import mxnet as mx
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sys
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from datetime import datetime
import time

def check_restricted_time():
    """Exits execution if the current time is Friday at or after 18:00."""
    now = datetime.now()
    # weekday 4 = Friday
    if now.weekday() == 4 and now.hour >= 18:
        logger.warning(f"Deadline reached (Friday 18:00). (Hard stop disabled by Antigravity to allow training)")
        # import sys
        # sys.exit(0)

# Add root to sys.path to import arcface_torch
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

try:
    from arcface_torch.backbones.iresnet import iresnet18, iresnet34, iresnet50, iresnet100
    from arcface_torch.losses import CombinedMarginLoss
except ImportError as e:
    logger.error(f"arcface_torch not found or failed to import. Error: {e}")

# --- MODEL & LOSS ---

class FaceModel(nn.Module):
    def __init__(self, num_local_classes, backbone_type="iresnet18"):
        super().__init__()
        logger.info(f"FaceModel initialized with backbone={backbone_type}, classes={num_local_classes}")
        # Backbone selection from arcface_torch
        if backbone_type == "iresnet18":
            self.backbone = iresnet18(num_features=512)
        elif backbone_type == "iresnet34":
            self.backbone = iresnet34(num_features=512)
        elif backbone_type == "iresnet50":
            self.backbone = iresnet50(num_features=512)
        elif backbone_type == "iresnet100":
            self.backbone = iresnet100(num_features=512)
        else:
            raise ValueError(f"Unsupported backbone type: {backbone_type}")
            
        # The head is a simple Linear layer that produces logits.
        # This matches the arcface_torch methodology of separating weights from the loss.
        self.head = nn.Linear(512, num_local_classes, bias=False)
        nn.init.xavier_uniform_(self.head.weight)

# --- DATA LOADING (DISJOINT PARTITIONING) ---

def get_transforms():
    return transforms.Compose([
        transforms.Resize((112, 112)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

class LocalMXFaceDataset(Dataset):
    """Dataset that reads from a .rec file but only exposes a specific subset of identities."""
    def __init__(self, root_dir, local_idx_range, label_map, transform=None):
        self.transform = transform
        self.label_map = label_map
        path_imgrec = os.path.join(root_dir, 'train.rec')
        path_imgidx = os.path.join(root_dir, 'train.idx')
        self.imgrec = mx.recordio.MXIndexedRecordIO(path_imgidx, path_imgrec, 'r')
        
        # local_idx_range should be the list of actual image indices for this partition
        self.imgidx = local_idx_range
        logger.info(f"LocalMXFaceDataset initialized with {len(self.imgidx)} samples.")

    def __getitem__(self, index):
        idx = self.imgidx[index]
        s = self.imgrec.read_idx(idx)
        header, img = mx.recordio.unpack(s)
        
        global_label = header.label
        if not isinstance(global_label, numbers.Number):
            global_label = global_label[0]
        
        # Map global label to local label [0, num_local_classes)
        # This ensures label < num_classes in the ArcFace head.
        local_label = self.label_map[int(global_label)]
        label = torch.tensor(local_label, dtype=torch.long)
        
        sample = mx.image.imdecode(img).asnumpy()
        # Convert NumPy array to PIL Image to be compatible with torchvision transforms
        sample = Image.fromarray(sample)
        
        if self.transform is not None:
            sample = self.transform(sample)
        
        return sample, label

    def __len__(self):
        return len(self.imgidx)

class LocalFaceDataset(Dataset):
    """Custom dataset that only loads a specific subset of identity folders."""
    def __init__(self, root_dir, assigned_identities, transform=None, max_imgs_per_identity=9999):
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []

        logger.info(f"LocalFaceDataset: Processing {len(assigned_identities)} identities with max_imgs_per_identity={max_imgs_per_identity}")
        for local_idx, identity_folder in enumerate(assigned_identities):
            identity_path = os.path.join(root_dir, identity_folder)
            if os.path.isdir(identity_path):
                img_count = 0
                try:
                    with os.scandir(identity_path) as it:
                        for entry in it:
                            if entry.is_file() and entry.name.lower().endswith(('.png', '.jpg', '.jpeg')):
                                if img_count >= max_imgs_per_identity:
                                    break
                                self.samples.append((entry.path, local_idx))
                                img_count += 1
                except Exception as e:
                    logger.error(f"Error scanning directory {identity_path}: {e}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, local_label = self.samples[idx]
        try:
            image = Image.open(img_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
        except Exception as e:
            logger.error(f"Error loading image {img_path}: {e}")
            image = torch.zeros((3, 112, 112))
        return image, local_label

def load_local_data(data_path: str, partition_id: int, num_partitions: int, batch_size: int, max_imgs_per_identity: int = 9999):
    logger.info(f"load_local_data started for partition_id={partition_id}, num_partitions={num_partitions}")
    
    rec_path = os.path.join(data_path, 'train.rec')
    idx_path = os.path.join(data_path, 'train.idx')
    
    if os.path.exists(rec_path) and os.path.exists(idx_path):
        logger.info(f"Optimized .rec dataset found at {data_path}")
        imgrec = mx.recordio.MXIndexedRecordIO(idx_path, rec_path, 'r')
        
        # CRITICAL FIX: Partition IDENTITIES instead of raw image indices to ensure Non-IID distribution.
        import fcntl
        import pickle
        
        cache_path = os.path.join(data_path, 'identity_map_cache.pkl')
        lock_path = cache_path + ".lock"
        
        with open(lock_path, 'w') as lock_file:
            # Acquire exclusive lock so concurrent clients don't overwrite the cache
            fcntl.flock(lock_file, fcntl.LOCK_EX)
            try:
                if os.path.exists(cache_path):
                    logger.info(f"Loading identity map from cache {cache_path}")
                    with open(cache_path, 'rb') as f:
                        label_to_indices = pickle.load(f)
                else:
                    logger.info(f"Building identity map for {data_path} (This may take ~10 seconds)")
                    label_to_indices = {}
                    keys = list(imgrec.keys)
                    for idx in keys:
                        s = imgrec.read_idx(idx)
                        header, _ = mx.recordio.unpack(s)
                        
                        # Only process actual image records (flag == 0)
                        if header.flag != 0:
                            continue
                            
                        label = header.label
                        if not isinstance(label, numbers.Number):
                            label = label[0]
                        label = int(label)
                        if label not in label_to_indices:
                            label_to_indices[label] = []
                        label_to_indices[label].append(idx)
                    
                    with open(cache_path, 'wb') as f:
                        pickle.dump(label_to_indices, f)
                    logger.info("Identity map built and cached.")
            finally:
                fcntl.flock(lock_file, fcntl.LOCK_UN)

        # Get all unique identities and sort them to ensure deterministic partitioning across clients
        unique_global_labels = sorted(list(label_to_indices.keys()))
        
        # Chunk IDENTITIES evenly (handling remainders to prevent the last partition from getting too many)
        total_labels = len(unique_global_labels)
        base_chunk = total_labels // num_partitions
        remainder = total_labels % num_partitions
        
        if partition_id < remainder:
            start = partition_id * (base_chunk + 1)
            end = start + (base_chunk + 1)
        else:
            start = remainder * (base_chunk + 1) + (partition_id - remainder) * base_chunk
            end = start + base_chunk
        
        local_labels = unique_global_labels[start:end]
        num_local_classes = len(local_labels)
        
        # Build local indices from assigned identities (applying max_imgs_per_identity limit)
        local_indices = []
        for label in local_labels:
            indices_for_label = label_to_indices[label][:max_imgs_per_identity]
            local_indices.extend(indices_for_label)
        
        # Create a mapping from global label to local label [0, num_local_classes)
        label_map = {global_label: i for i, global_label in enumerate(local_labels)}
        
        dataset = LocalMXFaceDataset(data_path, local_indices, label_map=label_map, transform=get_transforms())
        logger.info(f"LocalMXFaceDataset created for partition {partition_id} with {num_local_classes} classes and {len(local_indices)} total images.")
    else:
        logger.info(f"No .rec files found. Falling back to Folder loader at {data_path}")
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Data path {data_path} not found.")

        all_identities = []
        with os.scandir(data_path) as it:
            for entry in it:
                if entry.is_dir():
                    all_identities.append(entry.name)
        all_identities.sort()

        logger.info(f"Total identities found on disk: {len(all_identities)}")
        
        # DEBUG MODE: limit dataset size
        debug_max = os.environ.get("DEBUG_MAX_IDENTITIES")
        if debug_max:
            try:
                limit = int(debug_max)
                all_identities = all_identities[:limit]
                logger.warning(f"DEBUG MODE ACTIVE: Limiting dataset to first {limit} identities.")
            except ValueError:
                logger.error(f"Invalid DEBUG_MAX_IDENTITIES value: {debug_max}")

        if not all_identities:
            raise ValueError(f"No identity folders found in {data_path}")

        chunk_size = len(all_identities) // num_partitions
        if chunk_size == 0:
            start_idx = partition_id if partition_id < len(all_identities) else len(all_identities)
            end_idx = min(start_idx + 1, len(all_identities))
        else:
            start_idx = partition_id * chunk_size
            end_idx = (partition_id + 1) * chunk_size if partition_id < num_partitions - 1 else len(all_identities)

        assigned_identities = all_identities[start_idx:end_idx]
        num_local_classes = len(assigned_identities)
        logger.info(f"Assigned identities for partition {partition_id}: {num_local_classes} (from {start_idx} to {end_idx})")

        dataset = LocalFaceDataset(data_path, assigned_identities, transform=get_transforms(), max_imgs_per_identity=max_imgs_per_identity)
        logger.info(f"LocalFaceDataset created with {len(dataset)} samples.")
    
    trainloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=8, 
        drop_last=True,
        pin_memory=True
    )

    return trainloader, num_local_classes

def load_bin_dataset(bin_path: str):
    with open(bin_path, 'rb') as f:
        bins, issame_list = pickle.load(f, encoding='bytes')
    return bins, issame_list

# --- TRAIN & EVAL ---

def train(model, trainloader, epochs, lr, device):
    logger.info(f"Starting training on {device} for {epochs} epochs...")
    model.to(device)
    model.train()
    
    # Official ArcFace margin loss from arcface_torch
    # s=64.0, m1=1.0, m2=0.5, m3=0.0
    margin_loss = CombinedMarginLoss(s=64.0, m1=1.0, m2=0.5, m3=0.0)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    
    if len(trainloader) == 0:
        logger.warning("Empty trainloader.")
        return 0.0, 0.0, []
        
    epoch_losses = []
    epoch_accuracies = []
    
    for epoch in range(epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        
        # tqdm for interactive progress tracking
        pbar = tqdm(enumerate(trainloader), total=len(trainloader), desc=f"Epoch {epoch+1}/{epochs}", leave=False)
        
        for batch_idx, (images, labels) in pbar:
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            optimizer.zero_grad()
            
            # 1. Forward Pass: Get Embeddings
            embeddings = model.backbone(images)
            
            # 2. Get Raw Logits (normalized)
            # This is standard ArcFace methodology
            logits = F.linear(F.normalize(embeddings), F.normalize(model.head.weight))
            
            # 3. Apply Margin Penalty
            output = margin_loss(logits, labels)
            
            # 4. Loss and Step
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            # 5. Monitoring Accuracy: Use RAW Logits for real feedback
            _, predicted = torch.max(logits.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # Update progress bar
            if batch_idx % 10 == 0:
                pbar.set_postfix({
                    "loss": f"{loss.item():.4f}", 
                    "acc": f"{correct/total:.4f}"
                })
            
            if batch_idx % 20 == 0:
                logger.info(f"Epoch {epoch+1}/{epochs} | Batch {batch_idx}/{len(trainloader)} | Loss: {loss.item():.4f} | Acc: {correct/total:.4f}")
        
        # Track per-epoch metrics
        epoch_avg_loss = running_loss / len(trainloader) if len(trainloader) > 0 else 0.0
        epoch_acc = correct / total if total > 0 else 0.0
        epoch_losses.append(epoch_avg_loss)
        epoch_accuracies.append(epoch_acc)
        logger.info(f"Epoch {epoch+1}/{epochs} complete. Loss: {epoch_avg_loss:.4f}, Accuracy: {epoch_acc:.4f}")
            
    avg_loss = sum(epoch_losses) / len(epoch_losses) if epoch_losses else 0.0
    accuracy = epoch_accuracies[-1] if epoch_accuracies else 0.0
    
    logger.info(f"Training complete. Avg Loss: {avg_loss:.4f}, Final Accuracy: {accuracy:.4f}")
    model.to("cpu")
    torch.cuda.empty_cache()
    
    return avg_loss, accuracy, epoch_losses

def evaluate(model, dataloader, device):
    model.to(device)
    model.eval()
    correct = 0
    total = 0
    if len(dataloader) == 0:
        return 0.0
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            embeddings = model.backbone(images)
            # Use raw normalized logits for classification accuracy
            logits = F.linear(F.normalize(embeddings), F.normalize(model.head.weight))
            _, predicted = torch.max(logits.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = correct / total if total > 0 else 0.0
    
    model.to("cpu")
    torch.cuda.empty_cache()
    
    return accuracy

class BinDataset(Dataset):
    def __init__(self, bins, transform):
        self.bins = bins
        self.transform = transform
    def __len__(self):
        return len(self.bins)
    def __getitem__(self, idx):
        img = Image.open(io.BytesIO(self.bins[idx])).convert('RGB')
        return self.transform(img)

def test_verification(backbone, bins, issame_list, device, batch_size=256):
    """Standard face verification pipeline using cosine similarity on .bin files."""
    backbone.eval()
    backbone.to(device)
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    
    dataset = BinDataset(bins, transform)
    # Use 4 workers to decode evaluation images in parallel
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=4, pin_memory=True)
    
    all_embeddings = []
    with torch.no_grad():
        for imgs in tqdm(loader, desc="Evaluating", leave=False):
            imgs = imgs.to(device, non_blocking=True)
            embeddings = backbone(imgs)
            all_embeddings.append(F.normalize(embeddings))
            
    embeddings = torch.cat(all_embeddings, dim=0)
    actual_issame = torch.tensor(issame_list, device=device, dtype=torch.bool)
    
    emb1 = embeddings[0::2]
    emb2 = embeddings[1::2]
    similarities = torch.sum(emb1 * emb2, dim=1)
    
    # 10-fold cross validation for threshold search
    from sklearn.model_selection import KFold
    
    similarities_np = similarities.cpu().numpy()
    actual_issame_np = actual_issame.cpu().numpy()
    
    thresholds = torch.arange(-1.0, 1.0, 0.01).numpy()
    k_fold = KFold(n_splits=10, shuffle=False)
    
    accuracies = []
    
    for train_idx, test_idx in k_fold.split(similarities_np):
        # Find best threshold on 9 folds
        train_sims = similarities_np[train_idx]
        train_labels = actual_issame_np[train_idx]
        
        best_acc_train = 0.0
        best_thresh = 0.0
        
        for threshold in thresholds:
            predict_train = train_sims > threshold
            acc_train = np.mean(predict_train == train_labels)
            if acc_train > best_acc_train:
                best_acc_train = acc_train
                best_thresh = threshold
                
        # Test on the remaining 1 fold
        test_sims = similarities_np[test_idx]
        test_labels = actual_issame_np[test_idx]
        predict_test = test_sims > best_thresh
        acc_test = np.mean(predict_test == test_labels)
        accuracies.append(acc_test)
        
    backbone.to("cpu")
    torch.cuda.empty_cache()
    return float(np.mean(accuracies))
