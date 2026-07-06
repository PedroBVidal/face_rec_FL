# Investigation: Why `faces_emore_C1` Stopped Early

## Summary

**All simulations (C1, C2, C3, C_CROSS_SILO) are stopping early** — not just C1. The root cause is a **race condition between `flwr run --stream` and the scheduler**.

## Root Cause

### Primary Bug: `flwr run --stream` exits before simulation completes

The `flwr run --stream` command (Flower v1.29.0) connects to the superlink via gRPC to stream logs, but it **disconnects and exits with return code 0** while the simulation is still running on the superlink. This causes the scheduler to:

1. See `p.returncode == 0` (line 26 of [scheduler_emore.py](file:///home/pedro.vidal/facerec_flower/face_rec_fl/face_rec_fl/scheduler_emore.py#L26))
2. Mark the run as `.DONE` (line 28)
3. Launch the next run immediately

When the next `flwr run` starts, it creates a **new simulation run on the same superlink**, which supersedes/disrupts the previous in-flight simulation.

### Evidence (Timeline for C1)

| Time | Event |
|------|-------|
| May 25 22:45 | C1 starts (resumed from round 1, 99 remaining rounds) |
| May 26 06:09 | Round 15 evaluation completes |
| May 26 06:25:34 | Superlink log: last `StreamLogs` heartbeat for C1 |
| May 26 06:25:45 | Scheduler writes `faces_emore_C1.DONE` (returncode=0) |
| May 26 06:25:46 | Client 344 still training round 16 (!) |
| May 26 06:26:50 | Superlink: `All logs for run ID returned` |
| May 26 06:27:08 | Superlink: `StartRun` — C2 begins |

> [!CAUTION]
> C1 completed only **15 of 100 rounds**. C2 completed only **37 of 100 rounds**. C3 completed **0 of 100 rounds** before being marked DONE.

### Secondary Bug: `Array.nbytes` AttributeError

In [custom_strategy.py:94](file:///home/pedro.vidal/facerec_flower/face_rec_fl/face_rec_fl/custom_strategy.py#L94):

```python
msg_bytes = sum(a.nbytes for a in arrays.values())
```

Flower v1.29.0 uses `Array` objects (not raw numpy arrays), which don't have an `nbytes` attribute. This error is **caught** by the `try/except` block and only logged — it does **not** crash the simulation, but it means communication cost metrics are never collected.

## Affected Runs

| Run | Rounds Completed | Target | Status |
|-----|-----------------|--------|--------|
| C1 | 15 | 100 | ❌ Stopped early |
| C2 | 37 | 100 | ❌ Stopped early |
| C3 | ~0 | 100 | ❌ Stopped early (just starting round 1) |
| C_CROSS_SILO | Running | 100 | ⏳ Currently active |

## Proposed Fixes

### Fix 1: Replace `flwr run --stream` with a wait-for-completion approach

The `--stream` flag is unreliable for detecting completion. Options:

**Option A**: Remove `--stream` and poll the superlink for run status:
```python
# Launch without --stream, then poll the Flower API for run completion
cmd = ['flwr', 'run', '.', '--run-config', ...]  # no --stream
p = subprocess.Popen(cmd, ...)
p.wait()  # This returns quickly since flwr run without --stream just submits

# Then poll for completion using flwr CLI or API
while not is_run_complete(run_id):
    time.sleep(30)
```

**Option B**: Monitor the server log file for completion instead of relying on `flwr run` exit code:
```python
# Watch for "ServerApp main() execution finished" in the server log
import time
server_log = f"face_rec_fl/logs/{run_name}/server.log"
while True:
    if os.path.exists(server_log):
        with open(server_log) as f:
            content = f.read()
            if "execution finished" in content:
                break
    time.sleep(30)
```

**Option C (Recommended)**: Use `--stream` but validate completion by checking the server log before marking DONE:
```python
def is_run_actually_complete(run_name):
    """Check if the server log indicates the run finished properly."""
    server_log = f"face_rec_fl/logs/{run_name}/server.log"
    if os.path.exists(server_log):
        with open(server_log) as f:
            content = f.read()
            return "execution finished" in content or "Training complete" in content
    return False
```

### Fix 2: Fix the `nbytes` error

In [custom_strategy.py:94](file:///home/pedro.vidal/facerec_flower/face_rec_fl/face_rec_fl/custom_strategy.py#L94), replace:
```python
msg_bytes = sum(a.nbytes for a in arrays.values())
```
with:
```python
msg_bytes = sum(len(a.data) for a in arrays.values())
```
or:
```python
import numpy as np
msg_bytes = sum(np.prod(a.shape) * np.dtype(a.dtype).itemsize for a in arrays.values())
```

## Checkpoint Recovery

> [!TIP]
> The good news is that checkpoints were saved every round. C1 can resume from round 15, C2 from round 37. Re-running the scheduler after fixing the completion detection will automatically resume from the latest checkpoint.

### Current Checkpoint Status
- **C1**: Latest checkpoint at round 15 (best accuracy: 0.8479 at round 13)
- **C2**: Latest checkpoint at round 37 (best accuracy: 0.7912 at round 35)
- **C3**: No checkpoints (just started)
