from pyxai import Learning, Explainer, Tools
import pandas as pd
import time
from cpi_explaine_BT import *

dataset_name = Tools.Options.dataset
Tools.set_verbose(0)

# 1. Chargement et Entraînement avec K-Folds (par défaut 10 folds dans PyXAI)
learner = Learning.Xgboost(
    dataset_name + '.csv',
    learner_type=Learning.CLASSIFICATION
)

# K_FOLDS génère une liste de modèles (un par fold)
models = learner.evaluate(method=Learning.K_FOLDS, output=Learning.BT)

print(f"--- Début des tests sur {len(models)} folds ---")

# 2. Boucle sur chaque modèle de la validation croisée
for i, model in enumerate(models):
    print(f"\n>>> FOLD {i+1}/{len(models)}")
    
    # Pour chaque fold, on récupère 10 instances que le modèle classifie correctement
    # On utilise learner.get_instances en passant le modèle spécifique du fold
    instances, predictions = learner.get_instances(model, n=1, correct=True)
            # Initialisation de l'explainer pour ce modèle et cette instance
    explainer = Explainer.initialize(
    model,
    instances,
    features_type=dataset_name + '.types'
)
    
    # Préparation des données (tuples de littéraux binaires)
    v_binary = explainer.binary_representation
    x_frozenset = frozenset((abs(lit), 1 if lit > 0 else 0) for lit in v_binary)
    
    # --- Calcul de la CPI-Xp ---
    start_time = time.time()
    cpi_xp = findCPIexplanation_BT(x_frozenset, explainer)
    elapsed = time.time() - start_time
    
    if cpi_xp:
        reason_sat = cpi_xp_to_sat_format(cpi_xp)
        
        # Validation des résultats
        is_valid = explainer.is_implicant_BT(reason_sat)
        is_sufficient = explainer.is_sufficient_reason(reason_sat, n_samples=100)
        print("raison",reason_sat)
        print(f"    [OK] Time: {elapsed:.2f}s | Size: {len(reason_sat)} | Valid: {is_valid}")
        # print(f"    Reason SAT: {reason_sat}")
    else:
        print("    [FAILED] Aucune explication trouvée.")
        
print("-" * 30)

print("\n--- Expérience terminée ---")