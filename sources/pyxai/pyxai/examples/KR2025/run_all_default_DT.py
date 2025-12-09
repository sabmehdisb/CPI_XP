import os
import subprocess
import concurrent.futures
import gc
import sys

# Path to the folder containing the datasets
datasets_dir = os.path.abspath("../../../../../datasets")

# Create all combinations (dataset name, dataset path without .csv)
tasks = []
for filename in os.listdir(datasets_dir):
    if filename.endswith(".csv"):
        dataset_name = filename[:-4]  # Remove .csv
        dataset_path = os.path.join(datasets_dir, dataset_name)  # ← no .csv here
        tasks.append((dataset_name, dataset_path))

# Function to run a task with a timeout
def run_task(dataset_name, dataset_path):
    print(f"Running: {dataset_path} (unlimited)")

    # Disable garbage collector to avoid memory management crashes
    gc.disable()

    try:
        subprocess.run(
            [
                "python3",
                "main.py",
                f"-dataset={dataset_path}"  # ← without .csv
            ],
            # timeout=64800,  # 18 hours
            check=True
        )
    except subprocess.CalledProcessError as e:
        print(f"Error for dataset={dataset_name}: {e}")
    except Exception as e:
        print(f"Unexpected error for dataset={dataset_name}: {e}")
    finally:
        print("--------------------------------------------------")
        gc.enable()

# Launching in parallel with ThreadPool
max_threads = 16  # Adjust to your system
with concurrent.futures.ThreadPoolExecutor(max_workers=max_threads) as executor:
    futures = [executor.submit(run_task, name, path) for name, path in tasks]
    concurrent.futures.wait(futures)

# Clean exit after execution
sys.exit(0)
