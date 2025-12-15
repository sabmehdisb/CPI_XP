import os
import subprocess
import concurrent.futures
import sys
import time
from typing import List, Tuple

# ================= CONFIGURATION =================
# Exact name of the benchmark script to execute
SCRIPT_NAME = "Coverage-Based_Abductive_Explanations_for_DT.py"

# Path to the datasets directory (relative or absolute)
# Adjusted to point to the correct location relative to this script
DATASETS_DIR = os.path.abspath("../../../../../datasets")

# Number of parallel processes
# WARNING: High concurrency may lead to memory saturation (RAM) or CPU thrashing.
# Recommendation: Start with 4 workers and monitor system resources.
MAX_WORKERS = 16
# =================================================

def get_tasks(datasets_dir: str) -> List[Tuple[str, str]]:
    """
    Retrieves the list of valid datasets (.csv) to process.
    Checks for the existence of the corresponding .types file required by PyXAI.

    Args:
        datasets_dir: Path to the directory containing datasets.

    Returns:
        List of tuples (dataset_name, full_path_without_extension).
    """
    tasks = []
    if not os.path.exists(datasets_dir):
        print(f"ERROR: The directory {datasets_dir} does not exist.")
        return []

    print(f"Scanning directory: {datasets_dir}...")

    for filename in os.listdir(datasets_dir):
        if filename.endswith(".csv"):
            # Remove .csv extension because the internal script often appends it automatically
            dataset_name = filename[:-4]
            # Full path without extension
            full_path_no_ext = os.path.join(datasets_dir, dataset_name)
            
            # Verify if the .types file exists (often required for PyXAI/SAT solvers)
            if os.path.exists(full_path_no_ext + ".types"):
                tasks.append((dataset_name, full_path_no_ext))
            else:
                print(f"Warning: Skipped {dataset_name} (.types file missing)")
    
    return tasks


def run_task(dataset_name: str, dataset_path: str) -> bool:
    """
    Executes the benchmark script for a single dataset using a subprocess.

    Args:
        dataset_name: Name of the dataset (for logging).
        dataset_path: Full path to the dataset.

    Returns:
        True if execution was successful, False otherwise.
    """
    print(f"-> Starting: {dataset_name}")
    
    start_time = time.time()
    
    # Command equivalent to: python3 Script.py -dataset=...
    # sys.executable ensures the same Python interpreter/environment is used
    command = [
        sys.executable,  
        SCRIPT_NAME,
        f"-dataset={dataset_path}"
    ]

    try:
        # Run the subprocess
        # capture_output=True prevents parallel outputs from mixing in the console
        completed_process = subprocess.run(
            command,
            capture_output=True, 
            text=True,
            check=True  # Raises CalledProcessError if the script fails
        )
        
        duration = time.time() - start_time
        print(f"✓ Completed: {dataset_name} ({duration:.2f}s)")
        return True

    except subprocess.CalledProcessError as e:
        # This block catches non-zero exit codes (crashes, errors in the child script)
        print(f"X Error on {dataset_name} (Exit Code: {e.returncode})")
        # Print the last few lines of stderr to help debugging
        error_details = e.stderr.strip()[-300:] if e.stderr else "No error message captured."
        print(f"  Error details: ...{error_details}") 
        return False
        
    except Exception as e:
        # Catches system-level errors (e.g., file not found, permission denied)
        print(f"!! Unexpected crash on {dataset_name}: {e}")
        return False


def main():
    """Main entry point for the global benchmark execution."""
    print("=" * 60)
    print(f"BENCHMARK SUITE EXECUTION")
    print("=" * 60)
    print(f"Datasets Directory : {DATASETS_DIR}")
    print(f"Parallel Workers   : {MAX_WORKERS}")
    
    tasks = get_tasks(DATASETS_DIR)
    print(f"Total Datasets     : {len(tasks)}")
    print("-" * 60)

    # Parallel Execution using ThreadPoolExecutor
    # Note: Threads are sufficient here because the heavy lifting is done 
    # by independent subprocesses (which have their own GIL).
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # Submit all tasks
        future_to_dataset = {
            executor.submit(run_task, name, path): name 
            for name, path in tasks
        }
        
        # Wait for completion as they finish
        for future in concurrent.futures.as_completed(future_to_dataset):
            dataset_name = future_to_dataset[future]
            try:
                future.result()
            except Exception as exc:
                print(f"CRITICAL: Thread for {dataset_name} generated an exception: {exc}")

    print("\n" + "=" * 60)
    print("ALL BENCHMARKS COMPLETED")
    print("=" * 60)


if __name__ == "__main__":
    main()