import os, re, glob
runs = ["C1", "C2", "C3"]
log_dir = "/home/pedro.vidal/facerec_flower/face_rec_fl/face_rec_fl/logs"
for run in runs:
    run_path = os.path.join(log_dir, run)
    
    server_log = os.path.join(run_path, "server.log")
    agedb_count = 0
    if os.path.exists(server_log):
        with open(server_log, "r", errors='ignore') as f:
            for line in f:
                if "agedb_30 Accuracy:" in line:
                    agedb_count += 1
                    
    client_logs = glob.glob(os.path.join(run_path, "client_*.log"))
    max_round = 0
    for clog in client_logs:
        if "eval" in clog: continue
        with open(clog, "r", errors='ignore') as f:
            for line in f:
                m = re.search(r"for round (\d+)\.\.\.", line)
                if m:
                    max_round = max(max_round, int(m.group(1)))
                    
    print(f"{run} - server.log agedb count: {agedb_count}, max round in clients: {max_round}")
