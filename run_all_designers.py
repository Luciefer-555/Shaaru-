import json
import subprocess
import os
import time

def run_all():
    config_path = os.path.join("pipeline", "config", "designers.json")
    with open(config_path, "r") as f:
        designers = json.load(f)
        
    MAX_CONCURRENT = 2
    running_processes = []
    
    for designer in designers:
        source_id = designer["id"]
        print(f"Queueing {source_id}...")
        
        while len(running_processes) >= MAX_CONCURRENT:
            # Check if any have finished
            for p, s_id in running_processes[:]:
                if p.poll() is not None:
                    print(f"Finished {s_id} with return code {p.returncode}")
                    running_processes.remove((p, s_id))
            time.sleep(10)
            
        print(f"Starting {source_id} with 10-product limit...")
        cmd = ["python", "-u", "pipeline/run_pipeline.py", "--source", source_id, "--mode", "product", "--balance-genders"]
        
        # Use full path for log file
        log_path = os.path.join("pipeline", "output", "logs", f"{source_id}_full.log")
        log_file = open(log_path, "w")
        
        # Note: on Windows, python might be in path. We use 'python'
        p = subprocess.Popen(cmd, stdout=log_file, stderr=subprocess.STDOUT)
        running_processes.append((p, source_id))
        time.sleep(15) # Stagger starts to avoid hitting the exact same API second
        
    # Wait for remaining processes
    while len(running_processes) > 0:
        for p, s_id in running_processes[:]:
            if p.poll() is not None:
                print(f"Finished {s_id} with return code {p.returncode}")
                running_processes.remove((p, s_id))
        time.sleep(10)
        
    print("ALL DESIGNERS COMPLETED.")
    print("Running final steps (DB Expansion, Editorial, Neo4j)...")
    subprocess.run(["python", "finish_pipeline.py"])

if __name__ == "__main__":
    os.makedirs(os.path.join("pipeline", "output", "logs"), exist_ok=True)
    run_all()
