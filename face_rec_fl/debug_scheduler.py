import os
import subprocess
import time
import shutil
import signal
from datetime import datetime

# ============================================================================
# DEBUG SCHEDULER — Fast validation of the scheduler logic
#
# Uses DEBUG_MAX_IDENTITIES=100 so each run finishes in ~1 minute.
# All configs use 2 rounds, 1 epoch, 4 supernodes.
#
# Logs are fully separated from production runs:
#   - Scheduler log:  logs/debug_runs/scheduler.log       (high-level flow)
#   - Verbose log:    logs/debug_runs/verbose.log          (behind-the-scenes)
#   - Run stdout:     logs/debug_runs/debug_C1_stdout.log  (flwr output)
#   - Server logs:    logs/debug_C1/server.log             (separate run-names)
#   - Markers:        logs/debug_runs/debug_C1.DONE|FAILED
# ============================================================================

CONFIGS = {
    "C1": [
        'flwr', 'run', '.', '--stream',
        '--run-config',
        'run-name="debug_C1" num-server-rounds=2 local-epochs=1 '
        'batch-size=64 learning-rate=0.1 fraction-train=1.0 '
        'max-imgs-per-identity=9999 lr-decay-interval=10 '
        'data-path="/home/pedro.vidal/datasets/faces_emore"',
        '--federation-config',
        'num-supernodes=4 client-resources-num-cpus=8 '
        'client-resources-num-gpus=0.25'
    ],
    "C2": [
        'flwr', 'run', '.', '--stream',
        '--run-config',
        'run-name="debug_C2" num-server-rounds=2 local-epochs=1 '
        'batch-size=32 learning-rate=0.05 fraction-train=1.0 '
        'max-imgs-per-identity=9999 lr-decay-interval=20 '
        'data-path="/home/pedro.vidal/datasets/faces_emore"',
        '--federation-config',
        'num-supernodes=4 client-resources-num-cpus=8 '
        'client-resources-num-gpus=0.25'
    ],
    "C3": [
        'flwr', 'run', '.', '--stream',
        '--run-config',
        'run-name="debug_C3" num-server-rounds=2 local-epochs=1 '
        'batch-size=32 learning-rate=0.05 fraction-train=1.0 '
        'max-imgs-per-identity=10 lr-decay-interval=20 '
        'data-path="/home/pedro.vidal/datasets/faces_emore"',
        '--federation-config',
        'num-supernodes=4 client-resources-num-cpus=8 '
        'client-resources-num-gpus=0.25'
    ],
    "C_CROSS_SILO": [
        'flwr', 'run', '.', '--stream',
        '--run-config',
        'run-name="debug_C_CROSS_SILO" num-server-rounds=2 '
        'local-epochs=1 batch-size=64 learning-rate=0.1 fraction-train=1.0 '
        'max-imgs-per-identity=50 '
        'data-path="/home/pedro.vidal/datasets/faces_emore"',
        '--federation-config',
        'num-supernodes=4 client-resources-num-cpus=8 '
        'client-resources-num-gpus=0.25'
    ],
}

ORDER = ["C1", "C2", "C3", "C_CROSS_SILO"]
GPU_DEVICE = "3"
LOG_DIR = "/home/pedro.vidal/facerec_flower/face_rec_fl/face_rec_fl/logs/debug_runs"
PROJECT_DIR = "/home/pedro.vidal/facerec_flower/face_rec_fl"

os.makedirs(LOG_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Logging — two separate files:
#   scheduler.log  →  what you see in the terminal (high-level)
#   verbose.log    →  every detail (process lists, GPU queries, cleanup steps)
# ---------------------------------------------------------------------------
_scheduler_log_file = open(os.path.join(LOG_DIR, "scheduler.log"), "a")
_verbose_log_file = open(os.path.join(LOG_DIR, "verbose.log"), "a")


def log(msg: str):
    """High-level scheduler message → terminal + scheduler.log."""
    line = f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {msg}"
    print(line, flush=True)
    _scheduler_log_file.write(line + "\n")
    _scheduler_log_file.flush()
    # also goes to verbose
    _verbose_log_file.write(line + "\n")
    _verbose_log_file.flush()


def vlog(msg: str):
    """Verbose-only message → verbose.log only (not printed to terminal)."""
    line = f"[{datetime.now():%Y-%m-%d %H:%M:%S}] [VERBOSE] {msg}"
    _verbose_log_file.write(line + "\n")
    _verbose_log_file.flush()


def run_shell(cmd: str) -> str:
    """Run a shell command and return its output. Logs to verbose."""
    vlog(f"  $ {cmd}")
    try:
        result = subprocess.run(
            cmd, shell=True, capture_output=True, text=True, timeout=30
        )
        output = (result.stdout + result.stderr).strip()
        if output:
            for line in output.splitlines():
                vlog(f"    | {line}")
        return output
    except subprocess.TimeoutExpired:
        vlog(f"    | (command timed out after 30s)")
        return ""
    except Exception as e:
        vlog(f"    | (error: {e})")
        return ""


def log_gpu_state():
    """Query nvidia-smi for GPU 3 and write to verbose log."""
    vlog("--- GPU 3 state ---")
    run_shell(f"nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv -i {GPU_DEVICE}")
    run_shell(f"nvidia-smi --query-gpu=memory.used,memory.free,memory.total --format=csv -i {GPU_DEVICE}")
    vlog("--- end GPU 3 state ---")


def log_flower_processes():
    """List all flower/ray processes and write to verbose log."""
    vlog("--- Flower/Ray processes ---")
    run_shell("ps -u $(whoami) -o pid,rss,vsz,stat,start,time,cmd | grep -E 'flwr|flower|ray::' | grep -v grep")
    vlog("--- end Flower/Ray processes ---")


# ---------------------------------------------------------------------------
# Process cleanup
# ---------------------------------------------------------------------------

def _terminate_process_tree(proc: subprocess.Popen, name: str):
    """Kill the subprocess and ALL its children (SuperLink, simulation, Ray)."""
    log(f"  Cleaning up all processes for debug_{name}...")

    vlog(f"  Main proc PID={proc.pid}, poll={proc.poll()}, returncode={proc.returncode}")

    if proc.poll() is None:
        try:
            vlog(f"  Sending SIGTERM to PID {proc.pid}")
            proc.terminate()
            try:
                proc.wait(timeout=10)
                vlog(f"  PID {proc.pid} terminated gracefully")
            except subprocess.TimeoutExpired:
                vlog(f"  SIGTERM timed out, sending SIGKILL to PID {proc.pid}")
                proc.kill()
                proc.wait(timeout=5)
                vlog(f"  PID {proc.pid} killed")
        except Exception as e:
            vlog(f"  Error terminating PID {proc.pid}: {e}")

    _kill_project_flower_processes()

    flwr_cache = os.path.expanduser("~/.flwr/local-superlink")
    if os.path.exists(flwr_cache):
        vlog(f"  Removing SuperLink cache: {flwr_cache}")
        shutil.rmtree(flwr_cache, ignore_errors=True)

    vlog("  Post-cleanup process check:")
    log_flower_processes()
    log_gpu_state()

    time.sleep(3)


def _kill_project_flower_processes():
    """Kill ALL flower/ray processes associated with this project directory."""
    current_uid = os.getuid()
    my_pid = os.getpid()
    killed = 0

    if not os.path.exists("/proc"):
        return

    FLOWER_BINARIES = [
        'flower-superlink',
        'flower-superexec',
        'flwr-simulation',
        'flower-simulation',
        'ray',
    ]

    for pid_str in os.listdir("/proc"):
        if not pid_str.isdigit():
            continue
        pid = int(pid_str)

        if pid == my_pid:
            continue

        try:
            stat_info = os.stat(f"/proc/{pid}")
            if stat_info.st_uid != current_uid:
                continue

            cmdline_path = f"/proc/{pid}/cmdline"
            if not os.path.exists(cmdline_path):
                continue
            with open(cmdline_path, "r", errors="ignore") as f:
                cmdline = f.read().replace("\x00", " ")

            if not cmdline.strip():
                try:
                    with open(f"/proc/{pid}/stat", "r") as sf:
                        stat_content = sf.read()
                    if " Z " in stat_content:
                        comm_path = f"/proc/{pid}/comm"
                        if os.path.exists(comm_path):
                            with open(comm_path, "r") as cf:
                                comm = cf.read().strip()
                            if any(b in comm for b in FLOWER_BINARIES + ['flwr']):
                                vlog(f"  Found zombie PID {pid} ({comm}), killing")
                                log(f"  Killing zombie PID {pid} ({comm})")
                                os.kill(pid, signal.SIGKILL)
                                killed += 1
                except (OSError, ValueError):
                    pass
                continue

            cwd_path = f"/proc/{pid}/cwd"
            try:
                cwd = os.readlink(cwd_path)
            except OSError:
                cwd = ""

            is_flower = any(binary_name in cmdline for binary_name in FLOWER_BINARIES)
            is_our_dir = cwd.startswith(PROJECT_DIR)

            if is_flower and is_our_dir:
                vlog(f"  Found flower process PID {pid}, cwd={cwd}, cmd={cmdline.strip()[:200]}")
                log(f"  Killing PID {pid}: {cmdline.strip()[:100]}")
                try:
                    os.kill(pid, signal.SIGKILL)
                    killed += 1
                except OSError as e:
                    vlog(f"  Failed to kill PID {pid}: {e}")
        except (OSError, ValueError):
            pass

    if killed:
        log(f"  Killed {killed} processes.")
    else:
        log(f"  No flower processes found.")


def cleanup_flower_state():
    """Kill flower processes for this project and clear caches."""
    log("Cleaning up flower processes for this project...")
    vlog("=== cleanup_flower_state() START ===")
    log_flower_processes()
    _kill_project_flower_processes()

    flwr_cache = os.path.expanduser("~/.flwr/local-superlink")
    if os.path.exists(flwr_cache):
        shutil.rmtree(flwr_cache, ignore_errors=True)
        log("  Cleared Flower SuperLink cache.")
        vlog(f"  Removed {flwr_cache}")

    log_gpu_state()
    vlog("=== cleanup_flower_state() END ===")
    time.sleep(3)


# ---------------------------------------------------------------------------
# Launch
# ---------------------------------------------------------------------------

def launch_run(name: str) -> subprocess.Popen:
    """Launch `flwr run . --stream` as a subprocess."""
    log(f"Launching debug_{name}...")

    cmd = CONFIGS[name][:]

    stdout_path = os.path.join(LOG_DIR, f"debug_{name}_stdout.log")
    log_file = open(stdout_path, "w")

    env = os.environ.copy()
    env["PATH"] = "/home/pedro.vidal/miniconda3/envs/facerec/bin:" + env.get("PATH", "")
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    env["DEBUG_MAX_IDENTITIES"] = "100"
    env["CUDA_VISIBLE_DEVICES"] = GPU_DEVICE
    env["RAY_USAGE_STATS_ENABLED"] = "0"
    env["RAY_DISABLE_METRICS_EXPORT"] = "1"
    env["ENABLE_GLOBAL_EVAL"] = "true"

    log(f"  Command: {' '.join(cmd)}")
    log(f"  GPU: CUDA_VISIBLE_DEVICES={GPU_DEVICE}")
    log(f"  DEBUG_MAX_IDENTITIES=100")
    log(f"  Stdout → {stdout_path}")

    vlog(f"  Full environment overrides:")
    vlog(f"    PYTORCH_CUDA_ALLOC_CONF={env['PYTORCH_CUDA_ALLOC_CONF']}")
    vlog(f"    DEBUG_MAX_IDENTITIES={env['DEBUG_MAX_IDENTITIES']}")
    vlog(f"    CUDA_VISIBLE_DEVICES={env['CUDA_VISIBLE_DEVICES']}")
    vlog(f"    RAY_USAGE_STATS_ENABLED={env['RAY_USAGE_STATS_ENABLED']}")
    vlog(f"    RAY_DISABLE_METRICS_EXPORT={env['RAY_DISABLE_METRICS_EXPORT']}")
    vlog(f"    ENABLE_GLOBAL_EVAL={env['ENABLE_GLOBAL_EVAL']}")
    vlog(f"  CWD: {PROJECT_DIR}")

    p = subprocess.Popen(
        cmd,
        env=env,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        cwd=PROJECT_DIR,
    )

    log(f"  Started subprocess PID: {p.pid}")
    vlog(f"  Popen created: pid={p.pid}, args={cmd}")

    # Give it a moment to start and then snapshot the process tree
    time.sleep(5)
    vlog("  Process tree after 5s:")
    log_flower_processes()
    log_gpu_state()

    return p


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    log("=" * 60)
    log("DEBUG SCHEDULER — 100 identities, 2 rounds, 4 supernodes")
    log("=" * 60)
    vlog("=" * 60)
    vlog("DEBUG SCHEDULER VERBOSE LOG")
    vlog(f"Started at {datetime.now()}")
    vlog(f"PID: {os.getpid()}")
    vlog("=" * 60)

    # Step 1: Clean up any leftover processes
    cleanup_flower_state()

    # Step 2: Determine which runs are pending
    pending = []
    for name in ORDER:
        done_marker = os.path.join(LOG_DIR, f"debug_{name}.DONE")
        if os.path.exists(done_marker):
            log(f"  Skipping debug_{name} — already DONE")
        else:
            failed_marker = os.path.join(LOG_DIR, f"debug_{name}.FAILED")
            if os.path.exists(failed_marker):
                log(f"  Clearing previous FAILED marker for debug_{name}")
                os.remove(failed_marker)
            pending.append(name)

    if not pending:
        log("All runs already completed! Nothing to do.")
        return

    log(f"Pending runs: {pending}")
    log(f"Using GPU: {GPU_DEVICE}")
    log("")

    # Step 3: Run each benchmark sequentially
    for i, name in enumerate(pending):
        log(f"{'='*60}")
        log(f"Starting run: debug_{name} ({i+1}/{len(pending)})")
        log(f"{'='*60}")
        vlog(f">>>>>> RUN {name} ({i+1}/{len(pending)}) <<<<<<")

        # Clean state before each run
        cleanup_flower_state()

        # Launch the run
        launched_time = time.time()
        proc = launch_run(name)

        # Wait for the process to finish (no timers — blocks until exit)
        log(f"  Waiting for debug_{name} process to finish...")
        proc.wait()

        elapsed = time.time() - launched_time
        success = (proc.returncode == 0)

        vlog(f"  proc.wait() returned: returncode={proc.returncode}, elapsed={elapsed:.1f}s")

        if success:
            log(f"  ✓ Run debug_{name} completed successfully!")
            with open(os.path.join(LOG_DIR, f"debug_{name}.DONE"), "w") as f:
                f.write(f"DONE at {datetime.now()}\n")
        else:
            log(f"  ✗ Run debug_{name} failed with exit code {proc.returncode}.")
            with open(os.path.join(LOG_DIR, f"debug_{name}.FAILED"), "w") as f:
                f.write(f"FAILED at {datetime.now()} with code {proc.returncode}\n")

        # Check the flwr stdout log for errors
        stdout_path = os.path.join(LOG_DIR, f"debug_{name}_stdout.log")
        vlog(f"  --- Last 20 lines of {stdout_path} ---")
        try:
            with open(stdout_path, "r") as sf:
                lines = sf.readlines()
            for line in lines[-20:]:
                vlog(f"    | {line.rstrip()}")
        except Exception as e:
            vlog(f"    | (could not read: {e})")
        vlog(f"  --- end stdout tail ---")

        _terminate_process_tree(proc, name)

        log(f"  Run debug_{name} finished in {elapsed/60:.1f} minutes "
            f"({'SUCCESS' if success else 'FAILED'})")
        log("")

        # Pause between runs for GPU memory to fully release
        if i < len(pending) - 1:
            log("  Pausing 15s before next run...")
            time.sleep(15)

    log("=" * 60)
    log("All debug benchmarks completed!")
    log("=" * 60)

    # Final verbose summary
    vlog("=" * 60)
    vlog("FINAL STATE")
    vlog("=" * 60)
    log_flower_processes()
    log_gpu_state()
    vlog("Marker files:")
    for name in ORDER:
        done = os.path.exists(os.path.join(LOG_DIR, f"debug_{name}.DONE"))
        failed = os.path.exists(os.path.join(LOG_DIR, f"debug_{name}.FAILED"))
        vlog(f"  {name}: DONE={done}, FAILED={failed}")


if __name__ == '__main__':
    main()
