"""
Experimental Benchmark: Comparison of Explanation Methods for Random Forests.

This script evaluates the performance of several explanation methods:
1. External CPI (Cooper & Amgoud)
2. PyXAI CPI-XP
3. PyXAI M-CPI-XP
4. PyXAI Sufficient Reason (Baseline)

It records execution time, timeout rates, explanation size, and dataset statistics.
"""

import json
import signal
import time
import os
import pandas as pd
from pyxai import Learning, Explainer, Tools
from pyxai.sources.core.structure.type import Encoding 

# Import external functions for the specific comparison (RF version)
from cpi_explaine_RF import findCPIexplanation, cpi_xp_to_sat_format

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
    Calculates aggregate statistics for a specific method from the results DataFrame.

    Args:
        df_results (pd.DataFrame): DataFrame containing experimental results.
        method_suffix (str): The suffix identifying the method (e.g., 'ext', 'cpi').

    Returns:
        dict: A dictionary containing mean time, timeout rates, and average explanation sizes.
    """
    success_df = df_results[df_results[f'status_{method_suffix}'] == 'Success']
    
    if len(success_df) == 0:
        return {
            "mean_time_s": None,
            "timeout_count": len(df_results),
            "timeout_percentage": 100.0,
            "mean_raw_size": None,
            "mean_feature_size": None,
            "success_count": 0
        }
    
    total_runs = len(df_results)
    nb_timeouts = len(df_results[df_results[f'status_{method_suffix}'] == 'Timeout'])
    
    return {
        "mean_time_s": float(success_df[f'time_{method_suffix}'].mean()),
        "timeout_count": int(nb_timeouts),
        "timeout_percentage": float((nb_timeouts / total_runs) * 100),
        "mean_raw_size": float(success_df[f'len_raw_{method_suffix}'].mean()),
        "mean_feature_size": float(success_df[f'len_feat_{method_suffix}'].mean()),
        "success_count": int(len(success_df))
    }


def run_experiment(dataset_name, n_instances=10, n_folds=10, timeout_sec=60):
    """
    Runs the complete experimental benchmark on a dataset.

    Args:
        dataset_name (str): Name of the dataset file (without extension).
        n_instances (int): Number of test instances to process per fold.
        n_folds (int): Number of cross-validation folds.
        timeout_sec (int): Time limit per explanation call in seconds.

    Returns:
        tuple: (results_log, dataset_stats) containing the list of results and dataset metadata.
    """
    # Load dataset to extract metadata
    df = pd.read_csv(f"{dataset_name}.csv")
    
    # Capture dataset dimensions with scientific naming
    dataset_stats = {
        "n_instances": int(df.shape[0]),
        "n_features": int(df.shape[1]) - 1 
    }

    # Initialize Learner (Random Forest)
    rf_learner = Learning.Scikitlearn(
        f"{dataset_name}.csv", 
        learner_type=Learning.CLASSIFICATION
    )
    
    # Perform K-fold cross-validation
    rf_models = rf_learner.evaluate(
        method=Learning.K_FOLDS, 
        output=Learning.RF, 
        seed=0
    )
    
    results_log = []
    
    # Iterate through each fold
    for fold, rf_model in enumerate(rf_models):
        # Retrieve test instances for the current fold
        instances = rf_learner.get_instances(
            rf_model, 
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
                rf_model, 
                instance, 
                features_type=f"{dataset_name}.types"
            )
            
            # Prepare data structures for the explanation algorithms
            theory_clauses = explainer.get_theory()
            binary_representation = explainer.binary_representation
            constraints = list(theory_clauses)
            target_prediction = explainer.target_prediction
            
            # RF Specific: CNF Generation using MUS Encoding
            cnf_encoding = explainer._random_forest.to_CNF(
                explainer._instance, 
                explainer._binary_representation,
                target_prediction, 
                tree_encoding=Encoding.MUS
            )
            
            # Create a set of literals for the external algorithm
            binary_instance_set = frozenset(
                (abs(lit), 1 if lit > 0 else 0) for lit in binary_representation
            )
            
            # Initialize result dictionary for the current instance
            res_dict = {
                "fold": fold + 1,
                "instance_idx": instance_idx + 1,
                "prediction": int(prediction),
                "len_binary_repr": len(binary_representation), # Size of the binarized instance
                "len_theory": len(theory_clauses)              # Number of clauses in the model theory
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
    
    return results_log, dataset_stats


def generate_summary(results_log, dataset_stats, dataset_name):
    """
    Generates a statistical summary of the experiment.

    Args:
        results_log (list): List of dictionaries containing raw results.
        dataset_stats (dict): Metadata about the dataset structure.
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
    
    statistics = {
        method_name: compute_statistics(df_results, suffix)
        for method_name, suffix in methods.items()
    }
    
    # Calculate average complexity metrics
    avg_binary_len = float(df_results['len_binary_repr'].mean())
    avg_theory_len = float(df_results['len_theory'].mean())
    
    report = {
        "dataset": dataset_name,
        "dataset_metadata": {
            "n_instances": dataset_stats["n_instances"],
            "n_features": dataset_stats["n_features"],
            "avg_binary_instance_size": avg_binary_len,
            "avg_theory_clauses": avg_theory_len
        },
        "experiment_config": {
            "n_folds": int(df_results['fold'].max()),
            "total_instances": len(results_log),
            "instances_per_fold": len(results_log) // int(df_results['fold'].max())
        },
        "methods": statistics,
        "detailed_results": results_log
    }
    
    return report


def save_results(report, dataset_name):
    """Saves the report to a JSON file."""
    output_file = f"{dataset_name}_RF.json"
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    print(f"\nResults successfully saved to: {output_file}")


def display_summary(report):
    """Prints a formatted summary table to the console."""
    print("\n" + "="*80)
    print("EXPERIMENTAL RESULTS SUMMARY")
    print("="*80)
    
    meta = report['dataset_metadata']
    print(f"\nDataset: {report['dataset']}")
    print(f"Dimensions: {meta['n_instances']} instances x {meta['n_features']} features")
    print(f"Avg. Binary Instance Size: {meta['avg_binary_instance_size']:.2f} literals")
    print(f"Avg. Theory Size: {meta['avg_theory_clauses']:.2f} clauses")
    print(f"Total Instances Tested: {report['experiment_config']['total_instances']}")
    
    print("\n" + "-"*80)
    print(f"{'Method':<30} {'Mean Time (s)':<15} {'Success':<10} {'Timeouts':<12} {'Mean Size':<12}")
    print("-"*80)
    
    method_names = {
        "external_cooper_amgoud": "External (Cooper/Amgoud)",
        "pyxai_cpi_xp": "PyXAI CPI-XP",
        "pyxai_m_cpi_xp": "PyXAI M-CPI-XP",
        "pyxai_sufficient_reason": "PyXAI Sufficient Reason"
    }
    
    for method_key, method_display in method_names.items():
        stats = report['methods'][method_key]
        
        time_str = f"{stats['mean_time_s']:.4f}" if stats['mean_time_s'] is not None else "N/A"
        success_str = f"{stats['success_count']}"
        timeout_str = f"{stats['timeout_count']} ({stats['timeout_percentage']:.1f}%)"
        size_str = f"{stats['mean_feature_size']:.2f}" if stats['mean_feature_size'] is not None else "N/A"
        
        print(f"{method_display:<30} {time_str:<15} {success_str:<10} {timeout_str:<12} {size_str:<12}")
    
    print("="*80 + "\n")


def main():
    """Main entry point for the benchmark."""
    dataset_name = Tools.Options.dataset
    dataset_name_clean = os.path.basename(dataset_name)
    
    print(f"Starting experimental benchmark on dataset: {dataset_name_clean}")
    print("Configuration: 10-fold cross-validation, 10 instances per fold\n")
    
    # Run experiment
    results_log, dataset_stats = run_experiment(
        dataset_name=dataset_name,
        n_instances=10,
        n_folds=10,
        timeout_sec=1
    )
    
    # Generate and save report
    report = generate_summary(results_log, dataset_stats, dataset_name_clean)
    save_results(report, dataset_name)
    display_summary(report)


if __name__ == "__main__":
    main()