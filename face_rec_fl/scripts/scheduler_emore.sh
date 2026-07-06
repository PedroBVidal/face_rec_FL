#!/bin/bash
# =============================================================================
# EMORE BENCHMARK SCHEDULER — Simple sequential runner
#
# Runs each benchmark one after another. No polling, no timers.
# Kill the script (or the current flwr run) whenever you want.
# Checkpoints are saved per-round, so you can always resume.
#
# Usage:
#   nohup bash face_rec_fl/scripts/scheduler_emore.sh > face_rec_fl/logs/scheduler_logs/nohup_v4.out 2>&1 &
# =============================================================================

PROJECT_DIR="/home/pedro.vidal/facerec_flower/face_rec_fl"
LOG_DIR="$PROJECT_DIR/face_rec_fl/logs/scheduler_logs"
GPU_DEVICE="3"

cd "$PROJECT_DIR"

# Activate conda env
source /home/pedro.vidal/miniconda3/bin/activate facerec

# Environment
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES="$GPU_DEVICE"
export RAY_USAGE_STATS_ENABLED=0
export RAY_DISABLE_METRICS_EXPORT=1
export ENABLE_GLOBAL_EVAL=true
export FLWR_LOG_LEVEL=DEBUG

mkdir -p "$LOG_DIR"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

# Kill any leftover flower/ray from previous runs before starting
log "Initial cleanup of old processes..."
pkill -9 -u "$(whoami)" -f "flower-superlink|flower-superexe|flwr-simulation|flower-simulation" 2>/dev/null || true
pkill -9 -u "$(whoami)" -f "ray::" 2>/dev/null || true
rm -rf ~/.flwr/local-superlink
sleep 5

# Clean up flower/ray processes before and between runs
cleanup() {
    log "Cleaning up flower/ray processes..."
    pkill -9 -u "$(whoami)" -f "flower-superlink|flower-superexe|flwr-simulation|flower-simulation" 2>/dev/null || true
    pkill -9 -u "$(whoami)" -f "ray::" 2>/dev/null || true
    rm -rf ~/.flwr/local-superlink
    sleep 5
}

# Run a single benchmark. Args: NAME FLWR_ARGS...
run_benchmark() {
    local name="$1"; shift
    local marker="$LOG_DIR/faces_emore_${name}.DONE"

    if [[ -f "$marker" ]]; then
        log "SKIP $name — already DONE"
        return 0
    fi

    cleanup

    log "=========================================="
    log "STARTING: faces_emore_${name}"
    log "=========================================="

    # Run flwr and tee to a log file. flwr run blocks until completion.
    if flwr run . "$@" 2>&1 | tee "$LOG_DIR/faces_emore_${name}_stdout.log"; then
        log "DONE: faces_emore_${name}"
        date > "$marker"
    else
        log "FAILED: faces_emore_${name} (exit code $?)"
        date > "$LOG_DIR/faces_emore_${name}.FAILED"
    fi
}

# =============================================================================
# BENCHMARKS — same configs as before, in order
# =============================================================================

log "============================================================"
log "EMORE Benchmark Scheduler v4 (simple sequential)"
log "GPU: $GPU_DEVICE"
log "FLWR_LOG_LEVEL: $FLWR_LOG_LEVEL"
log "============================================================"

# C_CROSS_SILO: 4 supernodes, 100% participation, 1 epoch (PRIORITY — runs first)
run_benchmark C_CROSS_SILO \
    --stream \
    --run-config \
    'run-name="faces_emore_C_CROSS_SILO" num-server-rounds=100 local-epochs=1 batch-size=64 learning-rate=0.1 fraction-train=0.5 max-imgs-per-identity=50 data-path="/home/pedro.vidal/datasets/faces_emore"' \
    --federation-config \
    'num-supernodes=4 client-resources-num-cpus=8 client-resources-num-gpus=0.25'

# C1: Baseline Full — 857 supernodes, 15% participation, 3 epochs
run_benchmark C1 \
    --stream \
    --run-config \
    'run-name="faces_emore_C1" num-server-rounds=100 local-epochs=3 batch-size=64 learning-rate=0.1 fraction-train=0.15 max-imgs-per-identity=9999 lr-decay-interval=10 data-path="/home/pedro.vidal/datasets/faces_emore"' \
    --federation-config \
    'num-supernodes=857 client-resources-num-cpus=8 client-resources-num-gpus=0.25'

# C2: Device-Sim — 5716 supernodes, 5% participation, 1 epoch
run_benchmark C2 \
    --stream \
    --run-config \
    'run-name="faces_emore_C2" num-server-rounds=100 local-epochs=1 batch-size=32 learning-rate=0.05 fraction-train=0.05 max-imgs-per-identity=9999 lr-decay-interval=20 data-path="/home/pedro.vidal/datasets/faces_emore"' \
    --federation-config \
    'num-supernodes=5716 client-resources-num-cpus=8 client-resources-num-gpus=0.25'

# C3: Few-Shot — 857 supernodes, 100% participation, 2 epochs, 10 imgs/ID
run_benchmark C3 \
    --stream \
    --run-config \
    'run-name="faces_emore_C3" num-server-rounds=100 local-epochs=2 batch-size=32 learning-rate=0.05 fraction-train=1.0 max-imgs-per-identity=10 lr-decay-interval=20 data-path="/home/pedro.vidal/datasets/faces_emore"' \
    --federation-config \
    'num-supernodes=857 client-resources-num-cpus=8 client-resources-num-gpus=0.25'

cleanup
log "============================================================"
log "ALL EMORE BENCHMARKS COMPLETE"
log "============================================================"
