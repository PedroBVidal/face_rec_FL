import os 
import subprocess
import time
from datetime import datetime

# Configuration
CONFIGS = {
    "C1": ['flwr', 'run', '.', '--stream', '--run-config', 'run-name="C1" num-server-rounds=100 local-epochs=3 batch-size=64 learning-rate=0.1 fraction-train=0.65 max-imgs-per-identity=9999 lr-decay-interval=10', '--federation-config', 'num-supernodes=100 client-resources-num-cpus=8 client-resources-num-gpus=0.1'],
    "C2": ['flwr', 'run', '.', '--stream', '--run-config', 'run-name="C2" num-server-rounds=100 local-epochs=1 batch-size=32 learning-rate=0.05 fraction-train=0.2 max-imgs-per-identity=9999 lr-decay-interval=20', '--federation-config', 'num-supernodes=667 client-resources-num-cpus=8 client-resources-num-gpus=0.1'],
    "C3": ['flwr', 'run', '.', '--stream', '--run-config', 'run-name="C3" num-server-rounds=100 local-epochs=2 batch-size=32 learning-rate=0.05 fraction-train=1.0 max-imgs-per-identity=10 lr-decay-interval=20', '--federation-config', 'num-supernodes=100 client-resources-num-cpus=8 client-resources-num-gpus=0.1'],
    "C4": ['flwr', 'run', '.', '--stream', '--run-config', 'run-name="C4" num-server-rounds=100 local-epochs=1 batch-size=64 learning-rate=0.1 fraction-train=0.2 max-imgs-per-identity=50 lr-decay-interval=10', '--federation-config', 'num-supernodes=50 client-resources-num-cpus=8 client-resources-num-gpus=0.1'],
    "C5": ['flwr', 'run', '.', '--stream', '--run-config', 'run-name="C5" num-server-rounds=100 local-epochs=1 batch-size=32 learning-rate=0.03 fraction-train=0.05 max-imgs-per-identity=20 lr-decay-interval=25', '--federation-config', 'num-supernodes=1000 client-resources-num-cpus=8 client-resources-num-gpus=0.1']
}

ORDER = ["C1", "C2", "C3", "C4", "C5"]
MAX_CONCURRENT = 1  # Only 1 GPU available — must run sequentially
LOG_DIR = "/home/pedro.vidal/facerec_flower/face_rec_fl/face_rec_fl/logs/scheduler_logs"
os.makedirs(LOG_DIR, exist_ok=True)

def is_run_actually_complete(name):
    """Check if the server log indicates the run finished properly."""
    server_log = os.path.join(
        "/home/pedro.vidal/facerec_flower/face_rec_fl/face_rec_fl/logs",
        f"{name}",
        "server.log"
    )
    if os.path.exists(server_log):
        try:
            with open(server_log, "r", errors="ignore") as f:
                content = f.read()
                if "execution finished" in content or "Training complete" in content:
                    return True
                if "Traceback (most recent call" in content or "Exception" in content:
                    print(f"[{datetime.now()}] Run {name} failed with an exception in server log.")
                    return True
        except Exception as e:
            print(f"Error reading server log {server_log}: {e}")
    return False

MAX_WAIT_AFTER_EXIT = 7 * 24 * 3600  # 7 days max wallclock per run

def manage_processes(active_processes):
    still_active = []
    for name, p, launched_time in active_processes:
        if p.poll() is None:
            still_active.append((name, p, launched_time))
        else:
            print(f"[{datetime.now()}] Run {name} process exited with return code {p.returncode}")
            if is_run_actually_complete(name):
                print(f"[{datetime.now()}] Run {name} completed successfully.")
                with open(os.path.join(LOG_DIR, f"{name}.DONE"), "w") as f:
                    f.write("DONE")
            else:
                # Subprocess exited but log is not showing complete/crashed.
                # Check total elapsed time since launch.
                elapsed = time.time() - launched_time
                if elapsed > MAX_WAIT_AFTER_EXIT:
                    print(f"[{datetime.now()}] Run {name} process exited and exceeded max wall-clock ({MAX_WAIT_AFTER_EXIT/3600:.0f}h). Marking as failed.")
                else:
                    # Keep waiting for the server/client logs to update/finish
                    print(f"[{datetime.now()}] Run {name} process exited but no completion marker in server.log yet (elapsed: {elapsed/60:.0f}min). Waiting...")
                    still_active.append((name, p, launched_time))
    return still_active

def cleanup_flower_state():
    import shutil
    print(f"[{datetime.now()}] Cleaning up zombie flower processes and pending queue...")
    os.system("killall -9 flower-superlink 2>/dev/null")
    os.system("killall -9 flwr 2>/dev/null")
    flwr_cache = os.path.expanduser("~/.flwr/local-superlink")
    if os.path.exists(flwr_cache):
        shutil.rmtree(flwr_cache, ignore_errors=True)
    time.sleep(2)

def main():
    cleanup_flower_state()
    active_processes = []
    pending_configs = []
    
    for c in ORDER:
        if not os.path.exists(os.path.join(LOG_DIR, f"{c}.DONE")):
            pending_configs.append(c)
            
    print(f"[{datetime.now()}] Starting scheduler. Pending runs: {pending_configs}")

    while pending_configs or active_processes:
        active_processes = manage_processes(active_processes)
        
        while len(active_processes) < MAX_CONCURRENT and pending_configs:
            next_run = pending_configs.pop(0)
            print(f"[{datetime.now()}] Launching {next_run}...")
            cmd = CONFIGS[next_run]
            
            log_file = open(os.path.join(LOG_DIR, f"{next_run}_stdout.log"), "w")
            
            env = os.environ.copy()
            env["PATH"] = "/home/pedro.vidal/miniconda3/envs/facerec/bin:" + env.get("PATH", "")
            env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
            env["CUDA_VISIBLE_DEVICES"] = "4"
            env["RAY_USAGE_STATS_ENABLED"] = "0"
            env["RAY_DISABLE_METRICS_EXPORT"] = "1"
            env["ENABLE_GLOBAL_EVAL"] = "true"
            # Enable Flower DEBUG-level logging for full verbose output
            env["FLWR_LOG_LEVEL"] = "DEBUG"
            
            flwr_path = "/home/pedro.vidal/miniconda3/envs/facerec/bin/flwr"
            p = subprocess.Popen(cmd, executable=flwr_path, env=env, stdout=log_file, stderr=subprocess.STDOUT, cwd="/home/pedro.vidal/facerec_flower/face_rec_fl")
            active_processes.append((next_run, p, time.time()))
            
            time.sleep(10) # Stagger launches slightly
            
        time.sleep(10)
        
    print(f"[{datetime.now()}] All benchmarks completed.")

if __name__ == '__main__':
    main()
