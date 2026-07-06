import os
import shutil
import torch
from torch.utils.tensorboard import SummaryWriter
from flwr.app import ArrayRecord, ConfigRecord, Context, MetricRecord
from flwr.serverapp import Grid, ServerApp
from .custom_strategy import CustomFedAvg
from .task import load_bin_dataset, test_verification, iresnet18, iresnet34, iresnet50, iresnet100, check_restricted_time
from .logging_utils import setup_logger

# Initialize server logger (console-only at module level; reconfigured per-run in main())
logger = setup_logger("Server")

# Global state
best_avg_acc = 0.0
best_round = 0
early_stop_flag = False
patience_counter = 0
CHECKPOINT_DIR = "/home/pedro.vidal/facerec_flower/face_rec_fl/checkpoints"
writer = None # Global TensorBoard writer
backbone_type = "iresnet18"

app = ServerApp()

@app.main()
def main(grid: Grid, context: Context) -> None:
    global writer, logger, CHECKPOINT_DIR, best_avg_acc, best_round, early_stop_flag, patience_counter, backbone_type
    
    # Reset global state for each run (critical when scheduler reuses process)
    best_avg_acc = 0.0
    best_round = 0
    early_stop_flag = False
    patience_counter = 0
    
    num_rounds: int = context.run_config["num-server-rounds"]
    lr: float = context.run_config["learning-rate"]
    batch_size = context.run_config["batch-size"]
    local_epochs = context.run_config["local-epochs"]
    backbone_type = context.run_config.get("backbone-type", "iresnet18")
    
    # Build run name from configuration
    run_name = context.run_config.get("run-name", "")
    if not run_name:
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        is_debug = os.environ.get("DEBUG_MAX_IDENTITIES") is not None
        mode_str = f"debug_{os.environ.get('DEBUG_MAX_IDENTITIES')}" if is_debug else "prod"
        run_name = f"run_{mode_str}_{backbone_type}_lr{lr}_bs{batch_size}_ep{local_epochs}_r{num_rounds}_{timestamp}"
    
    # Create per-run log directory for all logs (server + clients)
    run_log_dir = os.path.join("face_rec_fl/logs", run_name)
    os.makedirs(run_log_dir, exist_ok=True)
    
    # Update CHECKPOINT_DIR to be run-specific
    CHECKPOINT_DIR = os.path.join(CHECKPOINT_DIR, run_name)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    
    # Re-initialize logger to write to per-run directory
    logger = setup_logger("Server", os.path.join(run_log_dir, "server.log"))
    
    tb_log_dir = os.path.join("face_rec_fl/logs/tensorboard", run_name)
    writer = SummaryWriter(log_dir=tb_log_dir)
    
    logger.info("="*50)
    logger.info("ServerApp main() execution started.")
    logger.info(f"Run: {run_name}")
    logger.info(f"TensorBoard initialized at {tb_log_dir}")
    logger.info(f"Text logs at {run_log_dir}")
    logger.info(f"Config: rounds={num_rounds}, lr={lr}, backbone={backbone_type}")

    if backbone_type == "iresnet18":
        global_backbone = iresnet18(num_features=512)
    elif backbone_type == "iresnet34":
        global_backbone = iresnet34(num_features=512)
    elif backbone_type == "iresnet50":
        global_backbone = iresnet50(num_features=512)
    elif backbone_type == "iresnet100":
        global_backbone = iresnet100(num_features=512)
    else:
        raise ValueError(f"Unsupported backbone type: {backbone_type}")
        
    logger.info("Global backbone initialized.")
    arrays = ArrayRecord(global_backbone.to("cpu").state_dict())

    # --- Checkpoint resume logic ---
    resume_from = str(context.run_config.get("resume-from", ""))
    resumed_round = 0
    
    if not resume_from:
        auto_resume_path = os.path.join(CHECKPOINT_DIR, "latest_checkpoint.pt")
        if os.path.isfile(auto_resume_path):
            resume_from = auto_resume_path

    if resume_from:
        if os.path.isfile(resume_from):
            checkpoint = torch.load(resume_from, map_location="cpu", weights_only=True)
            global_backbone.load_state_dict(checkpoint["backbone_state_dict"])
            arrays = ArrayRecord(global_backbone.to("cpu").state_dict())
            best_avg_acc = checkpoint.get("best_avg_acc", 0.0)
            best_round = checkpoint.get("best_round", 0)
            resumed_round = int(checkpoint.get("round", 0))
            logger.info(f"Resumed from checkpoint: {resume_from} (round {resumed_round}, best_avg_acc={best_avg_acc:.8f})")
        else:
            logger.warning(f"resume-from path '{resume_from}' not found — starting from scratch.")

    # LR decay interval: configurable for different training regimes
    lr_decay_interval = int(context.run_config.get("lr-decay-interval", 10))
    # Calculate remaining rounds
    remaining_rounds = max(0, num_rounds - resumed_round)
    logger.info(f"Total target rounds: {num_rounds}. Resumed from round: {resumed_round}. Remaining rounds to run: {remaining_rounds}.")
    
    strategy = CustomFedAvg(
        fraction_train=context.run_config["fraction-train"],
        fraction_evaluate=0.0,
        lr_decay_interval=lr_decay_interval,
        initial_server_round=resumed_round,
    )
    strategy.set_writer(writer)  # Inject TensorBoard writer into strategy
    logger.info(f"Strategy initialized. LR decay every {lr_decay_interval} rounds.")

    result = strategy.start(
        grid=grid,
        initial_arrays=arrays,
        train_config=ConfigRecord({
            "lr": lr, 
            "batch-size": context.run_config["batch-size"], 
            "local-epochs": context.run_config["local-epochs"],
            "backbone-type": backbone_type,
            "max-imgs-per-identity": context.run_config.get("max-imgs-per-identity", 9999),
            "server-round": 0, # This will be updated by strategy
            "run-name": run_name, # For per-run client log directories
        }),
        num_rounds=remaining_rounds,
        evaluate_fn=global_evaluate,
        timeout=86400, # 24-hour timeout
    )

    logger.info("Training complete. Saving final global backbone...")
    save_path = os.path.join(CHECKPOINT_DIR, "final_global_backbone.pt")
    torch.save(result.arrays.to_torch_state_dict(), save_path)
    
    if writer:
        writer.close()
    logger.info("ServerApp main() execution finished.")
    logger.info("="*50)

def global_evaluate(server_round: int, arrays: ArrayRecord) -> MetricRecord:
    global best_avg_acc, best_round, early_stop_flag, patience_counter, writer, backbone_type
    
    check_restricted_time()
    
    # Skip Round 0 evaluation
    if server_round == 0:
        logger.info("Round 0 - Initial evaluation skipped.")
        return MetricRecord({})

    enable_eval = os.environ.get("ENABLE_GLOBAL_EVAL", "true").lower() == "true"
    if not enable_eval:
        logger.info(f"Round {server_round} - Evaluation SKIPPED.")
        return MetricRecord({"skipped": 1.0})

    logger.info(f"--- [GLOBAL EVALUATION] Round {server_round} ---")
    if backbone_type == "iresnet18":
        backbone = iresnet18(num_features=512)
    elif backbone_type == "iresnet34":
        backbone = iresnet34(num_features=512)
    elif backbone_type == "iresnet50":
        backbone = iresnet50(num_features=512)
    elif backbone_type == "iresnet100":
        backbone = iresnet100(num_features=512)
    else:
        raise ValueError(f"Unsupported backbone type: {backbone_type}")
    backbone.load_state_dict(arrays.to_torch_state_dict())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    eval_dir = "/work/pedro.vidal/eval_data/"
    import glob
    bin_files = glob.glob(os.path.join(eval_dir, "*.bin"))
    datasets_to_eval = {os.path.splitext(os.path.basename(f))[0]: f for f in bin_files}

    metrics = {}
    total_acc = 0.0
    count = 0
    
    for name, path in datasets_to_eval.items():
        try:
            bins, issame_list = load_bin_dataset(path)
            acc = test_verification(backbone, bins, issame_list, device)
            metrics[f"eval/{name}_acc"] = acc
            total_acc += acc
            count += 1
            logger.info(f"{name} Accuracy: {acc:.8f}")
            if writer:
                writer.add_scalar(f"eval/{name}_acc", acc, server_round)
        except FileNotFoundError:
            logger.warning(f"Eval dataset {name} not found.")

    # Log GPU memory usage
    try:
        import pynvml
        pynvml.nvmlInit()
        gpu_id = int(os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(',')[0])
        handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_id)
        info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        mem_used_mb = info.used / (1024 * 1024)
        logger.info(f"GPU {gpu_id} Memory Used: {mem_used_mb:.1f} MB")
        if writer:
            writer.add_scalar("system/gpu_memory_used_MB", mem_used_mb, server_round)
    except Exception as e:
        logger.warning(f"Could not read GPU memory: {e}")

    # --- Save checkpoint after every round ---
    _save_checkpoint(server_round, arrays, best_avg_acc)

    if count > 0:
        avg_acc = total_acc / count
        metrics["eval/avg_acc"] = avg_acc
        logger.info(f"Average Evaluation Accuracy: {avg_acc:.8f}")
        
        # Log to TensorBoard
        if writer:
            writer.add_scalar("eval/avg_acc", avg_acc, server_round)
            writer.flush()
            
        min_delta = 0.002
        PATIENCE = 15
            
        if avg_acc > best_avg_acc + min_delta:
            best_avg_acc = avg_acc
            best_round = server_round
            patience_counter = 0
            best_save_path = os.path.join(CHECKPOINT_DIR, "best_global_backbone.pt")
            torch.save({
                "round": server_round,
                "backbone_state_dict": arrays.to_torch_state_dict(),
                "best_avg_acc": best_avg_acc,
                "best_round": best_round,
            }, best_save_path)
            logger.info(f"New BEST model saved (round {server_round}) with average accuracy {best_avg_acc:.8f}")
        else:
            patience_counter += 1
            logger.info(f"No improvement for {patience_counter} rounds (best: {best_avg_acc:.8f} at round {best_round}).")
            if patience_counter >= PATIENCE:
                logger.info(f"Early stopping triggered! Patience ({PATIENCE}) reached.")
                early_stop_flag = True

    # Pass early_stop_flag through metrics so strategy can read it reliably
    if early_stop_flag:
        metrics["early_stop"] = 1.0

    del backbone
    torch.cuda.empty_cache()
    return MetricRecord(metrics)


def _save_checkpoint(server_round: int, arrays: ArrayRecord, best_avg_acc: float) -> None:
    """Save a checkpoint after each round and maintain a latest symlink."""
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    ckpt_data = {
        "round": server_round,
        "backbone_state_dict": arrays.to_torch_state_dict(),
        "best_avg_acc": best_avg_acc,
    }
    round_path = os.path.join(CHECKPOINT_DIR, f"checkpoint_round_{server_round}.pt")
    torch.save(ckpt_data, round_path)

    latest_path = os.path.join(CHECKPOINT_DIR, "latest_checkpoint.pt")
    shutil.copy2(round_path, latest_path)

    logger.info(f"Checkpoint saved: {round_path} (also copied to {latest_path})")
