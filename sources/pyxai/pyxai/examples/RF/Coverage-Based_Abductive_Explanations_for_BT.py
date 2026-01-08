"""
Experimental Benchmark: Comparison of Explanation Methods for Boosted Trees (BT).

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

# Import external functions (Assurez-vous d'avoir une version compatible BT ou générique)
# Ici on adapte l'import pour refléter le changement de modèle
from cpi_explaine_BT import findCPIexplanation_BT, cpi_xp_to_sat_format

# Disable PyXAI verbose output
Tools.set_verbose(0)

class TimeoutException(Exception):
    pass

def timeout_handler(signum, frame):
    raise TimeoutException

if hasattr(signal, 'SIGALRM'):
    signal.signal(signal.SIGALRM, timeout_handler)

from multiprocessing import Process, Queue

def run_with_timeout(func, args=(), kwargs=None, timeout_duration=300):
    if kwargs is None: kwargs = {}
    
    # On utilise une Queue pour récupérer le résultat du processus enfant
    q = Queue()
    
    def wrapper(queue, *f_args, **f_kwargs):
        try:
            res = func(*f_args, **f_kwargs)
            queue.put(("Success", res))
        except Exception as e:
            queue.put(("Error", str(e)))

    start_time = time.perf_counter()
    p = Process(target=wrapper, args=(q, *args), kwargs=kwargs)
    p.start()
    
    # On attend le processus pendant la durée du timeout
    p.join(timeout_duration)
    
    if p.is_alive():
        # ICI : On force l'arrêt du processus car il a dépassé le temps
        p.terminate() 
        p.join() # On nettoie le processus
        status = "Timeout"
        result = None
    else:
        # Si le processus a fini, on récupère le résultat dans la queue
        if not q.empty():
            status, result = q.get()
        else:
            status = "Error: Process died unexpectedly"
            result = None
            
    duration = time.perf_counter() - start_time
    return result, duration, status

def compute_statistics(df_results, method_suffix):
    success_df = df_results[df_results[f'status_{method_suffix}'] == 'Success']
    if len(success_df) == 0:
        return {k: None for k in ["mean_time_s", "std_time_s", "mean_raw_size", "mean_feature_size", "std_feature_size"]} | {"timeout_count": len(df_results), "success_count": 0}
    
    std_time = success_df[f'time_{method_suffix}'].std()
    std_size = success_df[f'len_feat_{method_suffix}'].std()
    
    return {
        "mean_time_s": float(success_df[f'time_{method_suffix}'].mean()),
        "std_time_s": float(std_time) if not pd.isna(std_time) else 0.0,
        "timeout_count": int(len(df_results[df_results[f'status_{method_suffix}'] == 'Timeout'])),
        "timeout_percentage": float((len(df_results[df_results[f'status_{method_suffix}'] == 'Timeout']) / len(df_results)) * 100),
        "mean_raw_size": float(success_df[f'len_raw_{method_suffix}'].mean()),
        "mean_feature_size": float(success_df[f'len_feat_{method_suffix}'].mean()),
        "std_feature_size": float(std_size) if not pd.isna(std_size) else 0.0,
        "success_count": int(len(success_df))
    }

def display_summary(report):
    print("\n" + "="*100)
    print("EXPERIMENTAL RESULTS SUMMARY (BOOSTED TREES)")
    print("="*100)
    meta = report['dataset_metadata']
    perf = report.get('model_performance', {})
    print(f"\nDataset: {report['dataset']} | Dimensions: {meta['n_instances']} x {meta['n_features']}")
    if 'accuracy_mean' in perf:
        print(f"Model Accuracy (BT): {perf['accuracy_mean']:.2f}% ± {perf['accuracy_std']:.2f}%")
    print(f"Total Instances Tested: {report['experiment_config']['total_instances']}")
    print("\n" + "-"*100)
    print(f"{'Method':<30} {'Time (s) [mean ± std]':<25} {'Success':<10} {'Timeouts':<12} {'Size [feat. ± std]':<20}")
    print("-"*100)
    
    methods = {"external_cooper_amgoud": "External (Cooper/Amgoud)", "pyxai_cpi_xp": "PyXAI CPI-XP", "pyxai_m_cpi_xp": "PyXAI M-CPI-XP", "pyxai_tree_specific_reason": "PyXAI Sufficient Reason"}
    for k, name in methods.items():
        s = report['methods'][k]
        t_str = f"{s['mean_time_s']:.3f} ± {s['std_time_s']:.3f}" if s['mean_time_s'] is not None else "N/A"
        sz_str = f"{s['mean_feature_size']:.2f} ± {s['std_feature_size']:.2f}" if s['mean_feature_size'] is not None else "N/A"
        print(f"{name:<30} {t_str:<25} {s['success_count']:<10} {s['timeout_count']:<12} {sz_str:<20}")

def run_experiment(dataset_name, n_instances=10, n_folds=10, timeout_sec=300):
    df = pd.read_csv(f"{dataset_name}.csv")
    dataset_stats = {"n_instances": int(df.shape[0]), "n_features": int(df.shape[1]) - 1}

    # Initialize Learner (XGBOOST pour BT)
    bt_learner = Learning.Xgboost(
        f"{dataset_name}.csv", 
        learner_type=Learning.CLASSIFICATION
    )
    
    bt_models = bt_learner.evaluate(method=Learning.K_FOLDS, output=Learning.BT, seed=0)
    details = bt_learner.get_details()
    accuracies = [d['metrics']['accuracy'] for d in details]
    
    model_performance = {
        "model_name": "XGBoost",
        "accuracy_mean": statistics.mean(accuracies),
        "accuracy_std": statistics.stdev(accuracies) if len(accuracies) > 1 else 0.0,
        "parameters": details[0]['extras'].get('learner_options', {})
    }
    
    results_log = []
    for fold, bt_model in enumerate(bt_models):
        instances = bt_learner.get_instances(bt_model, n=n_instances, indexes=Learning.TEST, details=True)
        
        for instance_idx, instance_dict in enumerate(instances):
            instance, prediction = instance_dict["instance"], instance_dict["label"]
            explainer = Explainer.initialize(bt_model, instance, features_type=f"{dataset_name}.types")
            
            theory_clauses = explainer._boosted_trees.get_theory(explainer._binary_representation)
            binary_representation = explainer.binary_representation
            target_prediction = explainer.target_prediction
            
            binary_instance_set = frozenset((abs(lit), 1 if lit > 0 else 0) for lit in binary_representation)
            res_dict = {"fold": fold + 1, "instance_idx": instance_idx + 1, "len_binary_repr": len(binary_representation), "len_theory": len(theory_clauses)}
            
            # Methods definitions
            methods = {
                "ext": lambda: findCPIexplanation_BT(binary_instance_set, list(theory_clauses)),
                "cpi": lambda: explainer.cpi_xp(n=1, strategy="priority_order"),
                "mcpi": lambda: explainer.m_cpi_xp(n=1, strategy="priority_order"),
                "sr": lambda: explainer.tree_specific_reason()
            }
            
            for key, func in methods.items():
                print('time out')
                res, duration, status = run_with_timeout(func, timeout_duration=timeout_sec)
                # Conversion spécifique pour l'algo externe si succès
                if key == "ext" and status == "Success":
                    res = cpi_xp_to_sat_format(res)
                
                res_dict.update({
                    f"time_{key}": duration, f"status_{key}": status,
                    f"len_raw_{key}": len(res) if status == "Success" else None,
                    f"len_feat_{key}": len(explainer.to_features(res)) if status == "Success" else None
                })
            results_log.append(res_dict)
    
    return results_log, dataset_stats, model_performance

def generate_summary(results_log, dataset_stats, model_performance, dataset_name):
    df_results = pd.DataFrame(results_log)
    methods_map = {"external_cooper_amgoud": "ext", "pyxai_cpi_xp": "cpi", "pyxai_m_cpi_xp": "mcpi", "pyxai_tree_specific_reason": "sr"}
    
    return {
        "dataset": dataset_name,
        "dataset_metadata": {**dataset_stats, "avg_binary_features_size": float(df_results['len_binary_repr'].mean()), "avg_theory_clauses": float(df_results['len_theory'].mean())},
        "model_performance": model_performance,
        "experiment_config": {"n_folds": int(df_results['fold'].max()), "total_instances": len(results_log)},
        "methods": {name: compute_statistics(df_results, suffix) for name, suffix in methods_map.items()},
        "detailed_results": results_log
    }

def main():
    dataset_name = Tools.Options.dataset
    dataset_name_clean = os.path.basename(dataset_name)
    
    print(f"Starting BT experimental benchmark: {dataset_name_clean}")
    
    results_log, dataset_stats, model_performance = run_experiment(dataset_name, n_instances=2, n_folds=10, timeout_sec=300)
    report = generate_summary(results_log, dataset_stats, model_performance, dataset_name_clean)
    
    output_file = f"{dataset_name_clean}_BT.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    display_summary(report)

if __name__ == "__main__":
    main()