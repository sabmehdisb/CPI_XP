"""
Experimental Benchmark: Comparison of Explanation Methods for Decision Trees
Evaluates External CPI, PyXAI CPI-XP, M-CPI-XP, and Sufficient Reason methods
"""

import json
import signal
import time
from pathlib import Path

import pandas as pd
from pyxai import Learning, Explainer, Tools

# Import external functions
from cpi_explaine_DT import findCPIexplanation, cpi_xp_to_sat_format

Tools.set_verbose(0)
class TimeoutException(Exception):
    """Exception raised when computation exceeds time limit"""
    pass


def timeout_handler(signum, frame):
    """Signal handler for timeout"""
    raise TimeoutException


# Configure timeout mechanism (Unix/Linux/Mac)
if hasattr(signal, 'SIGALRM'):
    signal.signal(signal.SIGALRM, timeout_handler)


def run_with_timeout(func, args=(), kwargs=None, timeout_duration=60):
    """
    Execute a function with time limit and measure execution time
    
    Args:
        func: Function to execute
        args: Positional arguments
        kwargs: Keyword arguments
        timeout_duration: Maximum execution time in seconds
        
    Returns:
        tuple: (result, duration, status)
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
        if hasattr(signal, 'SIGALRM'):
            signal.alarm(0)
    
    end_time = time.perf_counter()
    duration = end_time - start_time
    
    return result, duration, status


def compute_statistics(df_results, method_suffix):
    """
    Calculate statistics for a given method
    
    Args:
        df_results: DataFrame containing experimental results
        method_suffix: Suffix identifying the method (e.g., 'ext', 'cpi')
        
    Returns:
        dict: Statistics including mean time, timeouts, and explanation sizes
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
    Run complete experimental benchmark
    
    Args:
        dataset_name: Name of the dataset (without .csv extension)
        n_instances: Number of instances to test per fold
        n_folds: Number of cross-validation folds
        timeout_sec: Timeout duration in seconds
        
    Returns:
        dict: Complete experimental results
    """
    # Load dataset and train models
    df = pd.read_csv(f"{dataset_name}.csv")
    dt_learner = Learning.Scikitlearn(
        f"{dataset_name}.csv", 
        learner_type=Learning.CLASSIFICATION
    )
    
    # K-fold cross-validation
    dt_models = dt_learner.evaluate(
        method=Learning.K_FOLDS, 
        output=Learning.DT, 
        seed=0
    )
    
    results_log = []
    
    # Process each fold
    for fold, dt_model in enumerate(dt_models):
        # Get test instances for current fold
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
            
            # Initialize explainer
            explainer = Explainer.initialize(
                dt_model, 
                instance, 
                features_type=f"{dataset_name}.types"
            )
            
            # Prepare common data structures
            th = explainer.get_theory()
            v_list = explainer.binary_representation
            forme_cnf = explainer.to_CNF()
            constraints = list(th)
            x_frozenset = frozenset(
                (abs(lit), 1 if lit > 0 else 0) for lit in v_list
            )
            
            # Initialize result dictionary
            res_dict = {
                "fold": fold + 1,
                "instance_idx": instance_idx + 1,
                "prediction": int(prediction)
            }
            
            # Method 1: External Algorithm (Cooper & Amgoud)
            func_ext = lambda: findCPIexplanation(x_frozenset, forme_cnf, constraints)
            res_ext, time_ext, status_ext = run_with_timeout(
                func_ext, timeout_duration=timeout_sec
            )
            
            res_dict.update({
                "time_ext": time_ext,
                "status_ext": status_ext,
                "len_raw_ext": len(cpi_xp_to_sat_format(res_ext)) if status_ext == "Success" else None,
                "len_feat_ext": len(explainer.to_features(cpi_xp_to_sat_format(res_ext))) if status_ext == "Success" else None
            })
            
            # Method 2: PyXAI CPI-XP
            ordre_features = []
            func_cpi = lambda: explainer.cpi_xp(
                n=1, 
                strategy="priority_order", 
                random_seed=None, 
                ordre_features=ordre_features
            )
            res_cpi, time_cpi, status_cpi = run_with_timeout(
                func_cpi, timeout_duration=timeout_sec
            )
            
            res_dict.update({
                "time_cpi": time_cpi,
                "status_cpi": status_cpi,
                "len_raw_cpi": len(res_cpi) if status_cpi == "Success" else None,
                "len_feat_cpi": len(explainer.to_features(res_cpi)) if status_cpi == "Success" else None
            })
            
            # Method 3: PyXAI M-CPI-XP
            func_mcpi = lambda: explainer.m_cpi_xp(
                n=1, 
                strategy="priority_order", 
                random_seed=None, 
                ordre_features=ordre_features
            )
            res_mcpi, time_mcpi, status_mcpi = run_with_timeout(
                func_mcpi, timeout_duration=timeout_sec
            )
            
            res_dict.update({
                "time_mcpi": time_mcpi,
                "status_mcpi": status_mcpi,
                "len_raw_mcpi": len(res_mcpi) if status_mcpi == "Success" else None,
                "len_feat_mcpi": len(explainer.to_features(res_mcpi)) if status_mcpi == "Success" else None
            })
            
            # Method 4: PyXAI Sufficient Reason (Baseline)
            func_sr = lambda: explainer.sufficient_reason(n=1)
            res_sr, time_sr, status_sr = run_with_timeout(
                func_sr, timeout_duration=timeout_sec
            )
            
            res_dict.update({
                "time_sr": time_sr,
                "status_sr": status_sr,
                "len_raw_sr": len(res_sr) if status_sr == "Success" else None,
                "len_feat_sr": len(explainer.to_features(res_sr)) if status_sr == "Success" else None
            })
            
            results_log.append(res_dict)
    
    return results_log


def generate_summary(results_log, dataset_name):
    """
    Generate summary statistics and save results
    
    Args:
        results_log: List of experimental results
        dataset_name: Name of the dataset
        
    Returns:
        dict: Complete experimental report
    """
    df_results = pd.DataFrame(results_log)
    
    # Calculate statistics for each method
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
    
    # Prepare complete report
    report = {
        "dataset": dataset_name,
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
    """
    Save experimental results to JSON file
    
    Args:
        report: Complete experimental report
        dataset_name: Name of the dataset
    """
    output_file = f"{dataset_name}_DT.json"
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    print(f"\nResults saved to: {output_file}")


def display_summary(report):
    """
    Display summary statistics in formatted table
    
    Args:
        report: Complete experimental report
    """
    print("\n" + "="*80)
    print("EXPERIMENTAL RESULTS SUMMARY")
    print("="*80)
    print(f"\nDataset: {report['dataset']}")
    print(f"Total Instances: {report['experiment_config']['total_instances']}")
    print(f"Cross-Validation Folds: {report['experiment_config']['n_folds']}")
    
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
    """Main execution function"""
    # Get dataset name from command line options
    dataset_name = Tools.Options.dataset
    
    print(f"Starting experimental benchmark on dataset: {dataset_name}")
    print("Configuration: 10-fold cross-validation, 10 instances per fold\n")
    
    # Run experiment
    results_log = run_experiment(
        dataset_name=dataset_name,
        n_instances=10,
        n_folds=10,
        timeout_sec=1
    )
    
    # Generate summary and save results
    report = generate_summary(results_log, dataset_name)
    save_results(report, dataset_name)
    display_summary(report)


if __name__ == "__main__":
    main()