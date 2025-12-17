"""
Experimental Benchmark: Comparison of Explanation Methods for Decision Trees.

This script evaluates the performance of several explanation methods:
1. External CPI (Cooper & Amgoud)
2. PyXAI CPI-XP
3. PyXAI M-CPI-XP
4. PyXAI Sufficient Reason (Baseline)

It records execution time, timeout rates, explanation size, dataset statistics,
and Model Accuracy (Mean +/- Std).
"""

import json
import signal
import time
import os
import statistics
import pandas as pd
from pyxai import Learning, Explainer, Tools

# Import external functions for the specific comparison
from cpi_explaine_DT import findCPIexplanation, cpi_xp_to_sat_format

# Disable PyXAI verbose output for cleaner logs
Tools.set_verbose(0)

class TimeoutException(Exception):
    """Exception raised when a computation exceeds the defined time limit."""
    pass


def timeout_handler(signum, frame):
    """Signal handler to trigger the TimeoutException."""
    raise TimeoutException


# Configure timeout mechanism (Unix/Linux/Mac compatible)
if hasattr(signal, 'SIGALRM'):
    signal.signal(signal.SIGALRM, timeout_handler)


def run_with_timeout(func, args=(), kwargs=None, timeout_duration=60):
    """
    Executes a function with a strict time limit and measures execution time.

    Args:
        func (callable): The function to execute.
        args (tuple): Positional arguments for the function.
        kwargs (dict): Keyword arguments for the function.
        timeout_duration (int): Maximum execution time in seconds.

    Returns:
        tuple: (result, duration, status) where status is 'Success', 'Timeout', or 'Error'.
    """
    if kwargs is None:
        kwargs = {}
    
    start_time = time.perf_counter()
    result = None
    status = "Success"
    
    try:
        if hasattr(signal, 'SIGALRM'):
            signal.alarm(timeout_duration)
        
        result = func(*args, **kwargs)
        
        if hasattr(signal, 'SIGALRM'):
            signal.alarm(0)
            
    except TimeoutException:
        status = "Timeout"
    except Exception as e:
        status = f"Error: {str(e)}"
    finally:
        # Ensure alarm is disabled even if an error occurs
        if hasattr(signal, 'SIGALRM'):
            signal.alarm(0)
    
    end_time = time.perf_counter()
    duration = end_time - start_time
    
    return result, duration, status


def compute_statistics(df_results, method_suffix):
    """
    Calculates aggregate statistics for a specific method, including 
    mean and standard deviation for time and explanation size.
    """
    # Filter only successful executions for valid statistics
    success_df = df_results[df_results[f'status_{method_suffix}'] == 'Success']
    
    # Handle case where no instance was successful
    if len(success_df) == 0:
        return {
            "mean_time_s": None,
            "std_time_s": None,
            "timeout_count": len(df_results),
            "timeout_percentage": 100.0,
            "mean_raw_size": None,
            "mean_feature_size": None,
            "std_feature_size": None,
            "success_count": 0
        }
    
    total_runs = len(df_results)
    nb_timeouts = len(df_results[df_results[f'status_{method_suffix}'] == 'Timeout'])
    
    # Calculate Standard Deviation (handle case with single result where std is NaN)
    std_time = success_df[f'time_{method_suffix}'].std()
    std_size = success_df[f'len_feat_{method_suffix}'].std()
    
    # If only 1 sample, std is NaN; replace with 0.0
    if pd.isna(std_time): std_time = 0.0
    if pd.isna(std_size): std_size = 0.0

    return {
        "mean_time_s": float(success_df[f'time_{method_suffix}'].mean()),
        "std_time_s": float(std_time),
        "timeout_count": int(nb_timeouts),
        "timeout_percentage": float((nb_timeouts / total_runs) * 100),
        "mean_raw_size": float(success_df[f'len_raw_{method_suffix}'].mean()),
        "mean_feature_size": float(success_df[f'len_feat_{method_suffix}'].mean()),
        "std_feature_size": float(std_size),
        "success_count": int(len(success_df))
    }


def display_summary(report):
    """
    Prints a formatted summary table to the console.
    """
    print("\n" + "="*100)
    print("EXPERIMENTAL RESULTS SUMMARY")
    print("="*100)
    
    meta = report['dataset_metadata']
    perf = report.get('model_performance', {})
    
    print(f"\nDataset: {report['dataset']}")
    print(f"Dimensions: {meta['total_rows']} rows x {meta['n_features']} features")
    
    # Display accuracy if available
    if 'accuracy_mean' in perf:
        print(f"Model Accuracy: {perf['accuracy_mean']:.2f}% ± {perf['accuracy_std']:.2f}%")
        
    print(f"Avg. Binary Instance Size: {meta['avg_binary_features_size']:.2f} literals")
    print(f"Total Instances Tested: {report['experiment_config']['total_instances']}")
    
    print("\n" + "-"*100)
    print(f"{'Method':<30} {'Time (s) [mean ± std]':<25} {'Success':<10} {'Timeouts':<12} {'Size [mean ± std]':<20}")
    print("-"*100)
    
    method_names = {
        "external_cooper_amgoud": "External (Cooper/Amgoud)",
        "pyxai_cpi_xp": "PyXAI CPI-XP",
        "pyxai_m_cpi_xp": "PyXAI M-CPI-XP",
        "pyxai_sufficient_reason": "PyXAI Sufficient Reason"
    }
    
    for method_key, method_display in method_names.items():
        stats = report['methods'][method_key]
        
        # Format Time string
        if stats['mean_time_s'] is not None:
            time_str = f"{stats['mean_time_s']:.3f} ± {stats['std_time_s']:.3f}"
        else:
            time_str = "N/A"
            
        # Format Size string
        if stats['mean_feature_size'] is not None:
            size_str = f"{stats['mean_feature_size']:.2f} ± {stats['std_feature_size']:.2f}"
        else:
            size_str = "N/A"

        success_str = f"{stats['success_count']}"
        timeout_str = f"{stats['timeout_count']} ({stats['timeout_percentage']:.1f}%)"
        
        print(f"{method_display:<30} {time_str:<25} {success_str:<10} {timeout_str:<12} {size_str:<20}")
    
    print("="*100 + "\n")


def run_experiment(dataset_name, n_instances=10, n_folds=10, timeout_sec=60):
    """
    Runs the complete experimental benchmark on a dataset.

    Args:
        dataset_name (str): Name of the dataset file (without extension).
        n_instances (int): Number of test instances to process per fold.
        n_folds (int): Number of cross-validation folds.
        timeout_sec (int): Time limit per explanation call in seconds.

    Returns:
        tuple: (results_log, dataset_stats, model_performance)
    """
    # Load dataset to extract metadata
    df = pd.read_csv(f"{dataset_name}.csv")
    
    # Capture dataset dimensions
    dataset_stats = {
        "total_rows": int(df.shape[0]),
        "n_features": int(df.shape[1])-1
    }

    # Initialize Learner
    dt_learner = Learning.Scikitlearn(
        f"{dataset_name}.csv", 
        learner_type=Learning.CLASSIFICATION
    )
    
    # Perform K-fold cross-validation
    # Note: To improve recall on imbalanced datasets, consider adding class_weight='balanced'
    dt_models = dt_learner.evaluate(
        method=Learning.K_FOLDS, 
        output=Learning.DT, 
        seed=0
    )
    
    # --- START ACCURACY EXTRACTION ---
    # Retrieve details from the 10 folds
    details = dt_learner.get_details()
    
    # Extract accuracy for each fold
    accuracies = [d['metrics']['accuracy'] for d in details]
    
    # Calculate mean and standard deviation
    mean_accuracy = statistics.mean(accuracies)
    std_accuracy = statistics.stdev(accuracies) if len(accuracies) > 1 else 0.0
    
    model_performance = {
        "model_name": details[0]['evaluation_output'],
        "accuracy_mean": mean_accuracy,
        "accuracy_std": std_accuracy,
        # Save options for reproducibility
        "parameters": details[0]['extras'].get('learner_options', {})
    }
    # --- END ACCURACY EXTRACTION ---
    
    results_log = []
    
    # Iterate through each fold
    for fold, dt_model in enumerate(dt_models):
        # Retrieve test instances for the current fold
        instances = dt_learner.get_instances(
            dt_model, 
            n=n_instances, 
            indexes=Learning.TEST, 
            details=True
        )
        
        # Process each instance
        for instance_idx, instance_dict in enumerate(instances):
            instance = instance_dict["instance"]
            prediction = instance_dict["label"]
            
            # Initialize the Explainer
            explainer = Explainer.initialize(
                dt_model, 
                instance, 
                features_type=f"{dataset_name}.types"
            )
            
            # Prepare data structures for the explanation algorithms
            theory_clauses = explainer.get_theory()
            binary_representation = explainer.binary_representation
            cnf_encoding = explainer.to_CNF()
            constraints = list(theory_clauses)
            
            # Create a set of literals for the external algorithm
            binary_instance_set = frozenset(
                (abs(lit), 1 if lit > 0 else 0) for lit in binary_representation
            )
            
            # Initialize result dictionary for the current instance
            res_dict = {
                "fold": fold + 1,
                "instance_idx": instance_idx + 1,
                "prediction": int(prediction),
                "len_binary_repr": len(binary_representation),
                "len_theory": len(theory_clauses)
            }
            
            # --- Method 1: External Algorithm (Cooper & Amgoud) ---
            func_ext = lambda: findCPIexplanation(binary_instance_set, cnf_encoding, constraints)
            res_ext, time_ext, status_ext = run_with_timeout(func_ext, timeout_duration=timeout_sec)
            
            res_dict.update({
                "time_ext": time_ext,
                "status_ext": status_ext,
                "len_raw_ext": len(cpi_xp_to_sat_format(res_ext)) if status_ext == "Success" else None,
                "len_feat_ext": len(explainer.to_features(cpi_xp_to_sat_format(res_ext))) if status_ext == "Success" else None
            })
            
            # --- Method 2: PyXAI CPI-XP ---
            feature_ordering = [] # Empty list implies default ordering strategy
            func_cpi = lambda: explainer.cpi_xp(
                n=1, 
                strategy="priority_order", 
                random_seed=None, 
                ordre_features=feature_ordering
            )
            res_cpi, time_cpi, status_cpi = run_with_timeout(func_cpi, timeout_duration=timeout_sec)
            
            res_dict.update({
                "time_cpi": time_cpi,
                "status_cpi": status_cpi,
                "len_raw_cpi": len(res_cpi) if status_cpi == "Success" else None,
                "len_feat_cpi": len(explainer.to_features(res_cpi)) if status_cpi == "Success" else None
            })
            
            # --- Method 3: PyXAI M-CPI-XP ---
            func_mcpi = lambda: explainer.m_cpi_xp(
                n=1, 
                strategy="priority_order", 
                random_seed=None, 
                ordre_features=feature_ordering
            )
            res_mcpi, time_mcpi, status_mcpi = run_with_timeout(func_mcpi, timeout_duration=timeout_sec)
            
            res_dict.update({
                "time_mcpi": time_mcpi,
                "status_mcpi": status_mcpi,
                "len_raw_mcpi": len(res_mcpi) if status_mcpi == "Success" else None,
                "len_feat_mcpi": len(explainer.to_features(res_mcpi)) if status_mcpi == "Success" else None
            })
            
            # --- Method 4: PyXAI Sufficient Reason (Baseline) ---
            func_sr = lambda: explainer.sufficient_reason(n=1)
            res_sr, time_sr, status_sr = run_with_timeout(func_sr, timeout_duration=timeout_sec)
            
            res_dict.update({
                "time_sr": time_sr,
                "status_sr": status_sr,
                "len_raw_sr": len(res_sr) if status_sr == "Success" else None,
                "len_feat_sr": len(explainer.to_features(res_sr)) if status_sr == "Success" else None
            })
            
            results_log.append(res_dict)
    
    # Return model_performance as well
    return results_log, dataset_stats, model_performance


def generate_summary(results_log, dataset_stats, model_performance, dataset_name):
    """
    Generates a statistical summary of the experiment.

    Args:
        results_log (list): List of dictionaries containing raw results.
        dataset_stats (dict): Metadata about the dataset structure.
        model_performance (dict): Summary of model accuracy.
        dataset_name (str): Name of the dataset.

    Returns:
        dict: A structured report containing configuration, stats, and raw data.
    """
    df_results = pd.DataFrame(results_log)
    
    methods = {
        "external_cooper_amgoud": "ext",
        "pyxai_cpi_xp": "cpi",
        "pyxai_m_cpi_xp": "mcpi",
        "pyxai_sufficient_reason": "sr"
    }
    
    statistics_methods = {
        method_name: compute_statistics(df_results, suffix)
        for method_name, suffix in methods.items()
    }
    
    # Calculate average complexity metrics
    avg_binary_len = float(df_results['len_binary_repr'].mean()) if not df_results.empty else 0
    avg_theory_len = float(df_results['len_theory'].mean()) if not df_results.empty else 0
    
    report = {
        "dataset": dataset_name,
        "dataset_metadata": {
            "total_rows": dataset_stats["total_rows"],
            "n_features": dataset_stats["n_features"],
            "avg_binary_features_size": avg_binary_len,
            "avg_theory_clauses": avg_theory_len
        },
        # --- ADDED HERE ---
        "model_performance": model_performance, 
        # ------------------
        "experiment_config": {
            "n_folds": int(df_results['fold'].max()) if not df_results.empty else 0,
            "total_instances": len(results_log),
            "instances_per_fold": len(results_log) // int(df_results['fold'].max()) if not df_results.empty else 0
        },
        "methods": statistics_methods,
        "detailed_results": results_log
    }
    
    return report


def save_results(report, dataset_name):
    """Saves the report to a JSON file."""
    output_file = f"{dataset_name}_DT.json"
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    print(f"\nResults successfully saved to: {output_file}")

def main():
    """Main entry point for the benchmark."""
    dataset_name = Tools.Options.dataset
    dataset_name_clean = os.path.basename(dataset_name)
    
    print(f"Starting experimental benchmark on dataset: {dataset_name_clean}")
    print("Configuration: 10-fold cross-validation, 10 instances per fold\n")
    
    # Run experiment - Now retrieving 3 values
    results_log, dataset_stats, model_performance = run_experiment(
        dataset_name=dataset_name,
        n_instances=10,
        n_folds=10,
        timeout_sec=60
    )
    
    # Generate and save report
    report = generate_summary(results_log, dataset_stats, model_performance, dataset_name_clean)
    save_results(report, dataset_name)
    display_summary(report)


if __name__ == "__main__":
    main()