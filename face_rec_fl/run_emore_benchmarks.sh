#!/bin/bash

# Automated Faces Emore Benchmark Runner
# Usage: nohup bash face_rec_fl/run_emore_benchmarks.sh > benchmark_run_emore.log 2>&1 &

cd "$(dirname "$0")"

echo "==========================================================="
echo "Starting faces_emore Benchmarks via Scheduler at $(date)"
echo "==========================================================="

# Activate environment
source /home/pedro.vidal/miniconda3/bin/activate facerec

# Run the new python scheduler
python scheduler_emore.py

echo "==========================================================="
echo "ALL faces_emore BENCHMARKS COMPLETE at $(date)"
echo "==========================================================="
