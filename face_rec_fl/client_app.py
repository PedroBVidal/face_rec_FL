import os
import torch
from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp
from .task import FaceModel, load_local_data, train, evaluate, check_restricted_time
from .logging_utils import setup_logger

app = ClientApp()

@app.train()
def train_client(msg: Message, context: Context):
    check_restricted_time()
    partition_id = context.node_config["partition-id"]
    
    # Use server-sent config to determine per-run log directory
    config = msg.content["config"]
    run_name = config.get("run-name", "")
    if run_name:
        log_file = f"face_rec_fl/logs/{run_name}/client_{partition_id}.log"
    else:
        log_file = f"face_rec_fl/logs/client_{partition_id}.log"
    logger = setup_logger(f"Client_{partition_id}", log_file)
    
    logger.info("="*50)
    logger.info(f"train_client() started for partition {partition_id}.")
    
    num_partitions = context.node_config["num-partitions"]
    
    # Use server-sent config if available, otherwise fallback to local run_config
    batch_size = int(config.get("batch-size", context.run_config.get("batch-size", 32)))
    local_epochs = int(config.get("local-epochs", context.run_config.get("local-epochs", 1)))
    lr = float(config.get("lr", context.run_config.get("learning-rate", 0.1)))
    backbone_type = config.get("backbone-type", context.run_config.get("backbone-type", "iresnet18"))
    max_imgs_per_identity = int(config.get("max-imgs-per-identity", context.run_config.get("max-imgs-per-identity", 9999)))
    
    data_path = context.run_config.get("data-path", "/home/pedro.vidal/datasets/faces_emore")
    
    logger.info(f"Configuration: Batch size: {batch_size}, Epochs: {local_epochs}, LR: {lr}, Backbone: {backbone_type}")
    
    # 1. Load Custom Disjoint Data
    logger.info("Loading local data...")
    trainloader, num_local_classes = load_local_data(
        data_path, partition_id, num_partitions, batch_size, max_imgs_per_identity=max_imgs_per_identity
    )
    logger.info(f"Local data loaded. {num_local_classes} classes.")

    # 2. Initialize Model with backbone type
    logger.info("Initializing model...")
    model = FaceModel(num_local_classes=num_local_classes, backbone_type=backbone_type)
    
    # 3. Load GLOBAL Backbone Weights from Server
    logger.info("Loading global backbone weights from server...")
    backbone_weights = msg.content["arrays"].to_torch_state_dict()
    model.backbone.load_state_dict(backbone_weights)
    
    # 4. Load LOCAL Head Weights from Disk
    run_name = config.get("run-name", "default_run")
    local_state_dir = f"/home/pedro.vidal/facerec_flower/face_rec_fl/local_state/{run_name}"
    os.makedirs(local_state_dir, exist_ok=True)
    
    head_current_path = os.path.join(local_state_dir, f"client_{partition_id}_head_current.pt")
    head_best_path = os.path.join(local_state_dir, f"client_{partition_id}_head_best.pt")
    
    # Get round number from server config
    server_round = int(config.get("server-round", 1))
    best_round = int(config.get("best-round", 0))
    
    import shutil
    # If the previous round was the best round, save the current head as best
    if server_round > 1 and (server_round - 1) == best_round:
        if os.path.exists(head_current_path):
            logger.info(f"Round {server_round-1} was the best. Saving head as best.")
            shutil.copy2(head_current_path, head_best_path)
    
    # Only load existing head if it exists (allows resuming across restarts)
    if os.path.exists(head_current_path):
        try:
            logger.info(f"Checking local head weights at {head_current_path} for round {server_round}...")
            loaded_state = torch.load(head_current_path, map_location="cpu")
            current_state = model.head.state_dict()
            
            # Prevent crashes if total identities changed (e.g. Debug vs Full mode)
            match = True
            for key in current_state:
                if key in loaded_state:
                    if current_state[key].shape != loaded_state[key].shape:
                        logger.warning(f"Shape mismatch for {key}: {current_state[key].shape} vs {loaded_state[key].shape}. Ignoring local weights.")
                        match = False
                        break
                else:
                    match = False
                    break
            
            if match:
                model.head.load_state_dict(loaded_state)
                logger.info("Local head weights loaded successfully.")
            else:
                logger.warning("Local head weights incompatible. Using initialized weights.")
        except Exception as e:
            logger.warning(f"Failed to load local head weights: {e}. Using initialized weights.")
    else:
        logger.info("No local head weights found, using initialized weights.")
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # 5. Train locally
    logger.info("Starting local training loop...")
    train_loss, train_accuracy, epoch_losses = train(
        model, trainloader, 
        local_epochs, 
        lr, device
    )
    logger.info(f"Training finished. Loss: {train_loss:.8f}, Accuracy: {train_accuracy:.8f}")
    logger.info(f"Per-epoch losses: {[f'{l:.4f}' for l in epoch_losses]}")

    # 6. Save updated LOCAL Head Weights back to Disk
    logger.info("Saving local head weights...")
    torch.save(model.head.state_dict(), head_current_path)

    # 7. Return ONLY the updated GLOBAL Backbone Weights to the Server
    logger.info("Preparing return message...")
    backbone_record = ArrayRecord(model.backbone.to("cpu").state_dict())
    
    metrics = {
        "train_loss": train_loss, 
        "train_accuracy": train_accuracy,
        "num-examples": len(trainloader.dataset),
        "server_round": float(server_round),
        "partition_id": float(partition_id),
    }
    # Include per-epoch losses for granular TensorBoard logging
    for i, ep_loss in enumerate(epoch_losses):
        metrics[f"epoch_{i}_loss"] = ep_loss
    metrics["num_epochs"] = float(len(epoch_losses))
    
    metric_record = MetricRecord(metrics)
    
    content = RecordDict({"arrays": backbone_record, "metrics": metric_record})
    logger.info("Returning result to server.")
    logger.info("="*50 + "\n")
    return Message(content=content, reply_to=msg)

@app.evaluate()
def evaluate_client(msg: Message, context: Context):
    check_restricted_time()
    partition_id = context.node_config["partition-id"]
    
    # Use run-name from run_config for per-run log directory
    run_name = context.run_config.get("run-name", "")
    if run_name:
        log_file = f"face_rec_fl/logs/{run_name}/client_{partition_id}_eval.log"
    else:
        log_file = f"face_rec_fl/logs/client_{partition_id}_eval.log"
    logger = setup_logger(f"Client_{partition_id}_Eval", log_file)
    
    logger.info(f"evaluate_client() started for partition {partition_id}.")
    
    num_partitions = context.node_config["num-partitions"]
    batch_size = context.run_config["batch-size"]
    backbone_type = context.run_config.get("backbone-type", "iresnet18")
    data_path = context.run_config.get("data-path", "/home/pedro.vidal/datasets/faces_emore")
    max_imgs_per_identity = int(context.run_config.get("max-imgs-per-identity", 9999))
    
    # 1. Load Custom Disjoint Data for evaluation
    evalloader, num_local_classes = load_local_data(
        data_path, partition_id, num_partitions, batch_size, max_imgs_per_identity=max_imgs_per_identity
    )

    # 2. Initialize Model
    model = FaceModel(num_local_classes=num_local_classes, backbone_type=backbone_type)
    
    # 3. Load GLOBAL Backbone Weights from Server
    backbone_weights = msg.content["arrays"].to_torch_state_dict()
    model.backbone.load_state_dict(backbone_weights)
    
    # 4. Load LOCAL Head Weights
    run_name = context.run_config.get("run-name", "default_run")
    local_state_dir = f"/home/pedro.vidal/facerec_flower/face_rec_fl/local_state/{run_name}"
    head_current_path = os.path.join(local_state_dir, f"client_{partition_id}_head_current.pt")
    if os.path.exists(head_current_path):
        try:
            logger.info(f"Checking local head weights at {head_current_path}...")
            loaded_state = torch.load(head_current_path, map_location="cpu")
            current_state = model.head.state_dict()
            
            match = True
            for key in current_state:
                if key in loaded_state:
                    if current_state[key].shape != loaded_state[key].shape:
                        match = False
                        break
                else:
                    match = False
                    break
            
            if match:
                model.head.load_state_dict(loaded_state)
                logger.info("Local head weights loaded.")
            else:
                logger.warning("Local head weights incompatible for evaluation. Using initialized weights.")
        except Exception:
            logger.warning("Could not load local head weights for evaluation.")
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # 5. Evaluate locally
    accuracy = evaluate(model, evalloader, device)
    logger.info(f"Evaluation accuracy: {accuracy:.4f}")

    # 6. Return evaluation metrics
    metrics = {"accuracy": accuracy, "num_examples": len(evalloader.dataset)}
    metric_record = MetricRecord(metrics)
    
    content = RecordDict({"metrics": metric_record})
    return Message(content=content, reply_to=msg)
