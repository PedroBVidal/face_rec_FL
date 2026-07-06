import time
import math
from typing import Iterable, Optional, Tuple, List
import logging
from flwr.serverapp import Grid
from flwr.serverapp.strategy import FedAvg
from flwr.app import ArrayRecord, ConfigRecord, Message

logger = logging.getLogger(__name__)

class CustomFedAvg(FedAvg):
    def __init__(self, *args, lr_decay_interval=10, initial_server_round=0, **kwargs):
        super().__init__(*args, **kwargs)
        self.initial_server_round = initial_server_round
        self.total_aggregations = 0
        self.start_time = None
        self.cumulative_comm_bytes = 0
        self.lr_decay_interval = lr_decay_interval
        self._early_stop = False  # Instance-level flag, set from evaluate metrics
        self.writer = None  # TensorBoard SummaryWriter — set via set_writer()

    def set_writer(self, writer):
        """Inject the TensorBoard SummaryWriter from server_app."""
        self.writer = writer

    def start(self, *args, evaluate_fn=None, **kwargs):
        """Override start to wrap evaluate_fn and break on early stopping."""
        self.start_time = time.time()
        self._early_stop = False

        if evaluate_fn is not None:
            original_evaluate_fn = evaluate_fn

            def wrapped_evaluate_fn(server_round, arrays):
                true_round = server_round + self.initial_server_round
                result = original_evaluate_fn(true_round, arrays)
                # Check if evaluate set the early_stop flag via MetricRecord
                if result is not None and result.get("early_stop", 0.0) >= 1.0:
                    self._early_stop = True
                    logger.info(f"Early stop detected from evaluate_fn at round {server_round}. Will stop after this round.")
                return result

            kwargs["evaluate_fn"] = wrapped_evaluate_fn
        
        return super().start(*args, **kwargs)

    def aggregate_train(self, *args, **kwargs):
        """Aggregate results and log client info with detailed TensorBoard metrics."""
        server_round = None
        results = None
        failures = None
        
        # Determine round and results from args or kwargs
        if len(args) >= 2:
            server_round = args[0]
            results = args[1]
            failures = args[2] if len(args) > 2 else []
        else:
            server_round = kwargs.get("server_round")
            results = kwargs.get("results")
            failures = kwargs.get("failures", [])

        if server_round is not None:
            server_round += self.initial_server_round

        round_end_time = time.time()
        round_duration = 0
        if self.start_time:
            round_duration = round_end_time - self.start_time
        self.start_time = round_end_time # Reset for next round

        if results:
            self.total_aggregations += len(results)
            
            # Extract metrics for logging
            train_losses = []
            train_accs = []
            raw_losses = []  # Unweighted, for std/min/max
            total_examples = 0
            round_comm_bytes = 0
            
            # Per-epoch loss collection: epoch_idx -> list of (loss, num_examples)
            epoch_losses_by_idx = {}
            
            try:
                ids = []
                client_metrics = []
                for msg in results:
                    ids.append(msg.metadata.src_node_id)
                    
                    # 1. Calculate Communication Cost (Payload size)
                    arrays = msg.content.get("arrays")
                    if arrays:
                        msg_bytes = sum(len(a.data) for a in arrays.values())
                        round_comm_bytes += msg_bytes
                        
                        # Save the client update for analysis (e.g., PCA, t-SNE, Divergence)
                        try:
                            import os
                            import torch
                            partition_id = int(msg.content.get("metrics", {}).get("partition_id", msg.metadata.src_node_id))
                            save_dir = f"/home/pedro.vidal/facerec_flower/face_rec_fl/checkpoints/client_updates/round_{server_round}"
                            os.makedirs(save_dir, exist_ok=True)
                            torch.save(arrays.to_torch_state_dict(), os.path.join(save_dir, f"client_{partition_id}.pt"))
                        except Exception as e:
                            logger.error(f"Failed to save client checkpoint: {e}")
                    
                    # 2. Extract Training Metrics
                    metrics = msg.content.get("metrics")
                    if metrics:
                        loss = metrics.get("train_loss", 0.0)
                        acc = metrics.get("train_accuracy", 0.0)
                        n = metrics.get("num-examples", 1)
                        partition_id = int(metrics.get("partition_id", msg.metadata.src_node_id))
                        client_metrics.append((partition_id, loss, acc))
                        
                        train_losses.append(loss * n)
                        train_accs.append(acc * n)
                        raw_losses.append(loss)
                        total_examples += n
                        
                        # Collect per-epoch losses
                        num_epochs = int(metrics.get("num_epochs", 0))
                        for ep_idx in range(num_epochs):
                            ep_loss = metrics.get(f"epoch_{ep_idx}_loss", None)
                            if ep_loss is not None:
                                if ep_idx not in epoch_losses_by_idx:
                                    epoch_losses_by_idx[ep_idx] = []
                                epoch_losses_by_idx[ep_idx].append((ep_loss, n))
                
                self.cumulative_comm_bytes += round_comm_bytes
                
                logger.info(f"\n>>> Round {server_round}: Aggregating {len(ids)} clients. IDs: {ids}")
                
                if total_examples > 0:
                    avg_loss = sum(train_losses) / total_examples
                    avg_acc = sum(train_accs) / total_examples
                    
                    # Compute loss statistics across clients
                    loss_std = 0.0
                    min_loss = 0.0
                    max_loss = 0.0
                    if len(raw_losses) > 1:
                        mean_loss = sum(raw_losses) / len(raw_losses)
                        variance = sum((l - mean_loss) ** 2 for l in raw_losses) / (len(raw_losses) - 1)
                        loss_std = math.sqrt(variance)
                        min_loss = min(raw_losses)
                        max_loss = max(raw_losses)
                    elif len(raw_losses) == 1:
                        min_loss = raw_losses[0]
                        max_loss = raw_losses[0]
                    
                    # Log Summary
                    logger.info(f">>> Aggregated Client Metrics - Loss: {avg_loss:.4f} (std: {loss_std:.4f}, min: {min_loss:.4f}, max: {max_loss:.4f}), Accuracy: {avg_acc:.4f}")
                    logger.info(f">>> Round Duration: {round_duration:.2f}s | Comm: {round_comm_bytes / (1024**2):.2f} MB")
                    
                    # Log per-epoch aggregated losses
                    for ep_idx in sorted(epoch_losses_by_idx.keys()):
                        ep_data = epoch_losses_by_idx[ep_idx]
                        ep_total_n = sum(n for _, n in ep_data)
                        ep_weighted_avg = sum(l * n for l, n in ep_data) / ep_total_n if ep_total_n > 0 else 0.0
                        logger.info(f">>>   Epoch {ep_idx} avg loss: {ep_weighted_avg:.4f} (from {len(ep_data)} clients)")
                    
                    # Log to TensorBoard
                    _w = self.writer
                    if _w:
                        # Quality Metrics
                        _w.add_scalar("train/avg_loss", avg_loss, server_round)
                        _w.add_scalar("train/avg_acc", avg_acc, server_round)

                        # Per-Client Metrics (skip for large runs to avoid TB bloat)
                        if len(client_metrics) <= 50:
                            for pid, p_loss, p_acc in client_metrics:
                                _w.add_scalar(f"train/client_{pid}_loss", p_loss, server_round)
                                _w.add_scalar(f"train/client_{pid}_acc", p_acc, server_round)

                        # Loss Distribution (measures client heterogeneity)
                        _w.add_scalar("train/loss_std", loss_std, server_round)
                        _w.add_scalar("train/min_loss", min_loss, server_round)
                        _w.add_scalar("train/max_loss", max_loss, server_round)

                        # Per-Epoch Aggregated Losses
                        for ep_idx in sorted(epoch_losses_by_idx.keys()):
                            ep_data = epoch_losses_by_idx[ep_idx]
                            ep_total_n = sum(n for _, n in ep_data)
                            ep_weighted_avg = sum(l * n for l, n in ep_data) / ep_total_n if ep_total_n > 0 else 0.0
                            _w.add_scalar(f"train/epoch_{ep_idx}_avg_loss", ep_weighted_avg, server_round)

                        # Performance & Cost Metrics
                        _w.add_scalar("perf/round_duration_sec", round_duration, server_round)
                        _w.add_scalar("perf/comm_round_mb", round_comm_bytes / (1024**2), server_round)
                        _w.add_scalar("perf/comm_cumulative_gb", self.cumulative_comm_bytes / (1024**3), server_round)

                        # Participation Metrics
                        _w.add_scalar("participation/success_count", len(results), server_round)
                        _w.add_scalar("participation/failure_count", len(failures) if failures else 0, server_round)

                        _w.flush()
                    else:
                        logger.warning("TensorBoard writer not set on strategy — call set_writer() in server_app.")
                        
            except Exception as e:
                logger.error(f"Error during custom metric aggregation: {e}")
            
            logger.info(f">>> Total successful aggregations so far: {self.total_aggregations}")
        
        return super().aggregate_train(*args, **kwargs)


    def configure_train(
        self, server_round: int, arrays: ArrayRecord, config: ConfigRecord, grid: Grid
    ) -> Iterable[Message]:
        """Configure the next round of federated training and log config."""
        server_round += self.initial_server_round
        
        # Check early stop — both instance flag (from evaluate) and module global (fallback)
        if self._early_stop:
            logger.info(f"Early stop flag detected at round {server_round}. Skipping training.")
            return []
        
        try:
            from . import server_app as _sa
            if getattr(_sa, "early_stop_flag", False):
                logger.info(f"Early stop flag (module global) detected at round {server_round}. Skipping training.")
                self._early_stop = True
                return []
            config["best-round"] = getattr(_sa, "best_round", 0)
        except Exception as e:
            logger.error(f"Error checking early stop flag: {e}")
            config["best-round"] = 0

        # Decrease learning rate by a configurable factor every N rounds
        if server_round % self.lr_decay_interval == 0 and server_round > 0:
            config["lr"] *= 0.5
            logger.info(f"LR decreased to: {config['lr']} (decay every {self.lr_decay_interval} rounds)")
        
        # Log Learning Rate to TensorBoard
        _w = self.writer
        if _w:
            _w.add_scalar("config/learning_rate", config["lr"], server_round)
            _w.flush()

        # Pass the current round number to the client
        config["server-round"] = server_round
        
        return super().configure_train(server_round, arrays, config, grid)
