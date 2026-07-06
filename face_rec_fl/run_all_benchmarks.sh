#!/bin/bash
###############################################################################
# Automated DCFace Benchmark Runner using Python Scheduler
#
# Usage:
#   nohup bash face_rec_fl/run_all_benchmarks.sh > benchmark_run.log 2>&1 &
#
# Monitor progress:
#   tail -f benchmark_run.log
#   tensorboard --logdir face_rec_fl/logs/tensorboard --port 6006
###############################################################################

echo "============================================================"
echo "Starting DCFace Benchmarks via Scheduler at $(date)"
echo "============================================================"

# Kill any existing processes (optional clean up)
pkill -9 -u pedro.vidal -f "flwr|ray|flower" 2>/dev/null || true
sleep 5

# Launch the python scheduler
python face_rec_fl/scheduler.py

echo "============================================================"
echo "ALL DCFace BENCHMARKS COMPLETE at $(date)"
echo "============================================================"
echo "Results available at:"
echo "  Logs:        face_rec_fl/logs/"
echo "  TensorBoard: face_rec_fl/logs/tensorboard/"
echo "  Checkpoints: checkpoints/"
echo "============================================================"
