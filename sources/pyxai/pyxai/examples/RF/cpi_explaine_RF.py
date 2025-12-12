from pysat.solvers import Glucose3
import json
from pyxai import Learning, Explainer, Tools
from pyxai.sources.core.structure.type import Encoding # Important pour Encoding.MUS
import pandas as pd
from sklearn.model_selection import train_test_split
import time
import numpy as np

# ===================== Utilitaires =====================
def literal_to_sat(literal):
    """Convertit (var, value) en SAT: (1,1)->1, (1,0)->-1"""
    return literal[0] if literal[1] == 1 else -literal[0]

def cpi_xp_to_sat_format(cpi_xp):
    """Convertit frozenset({(1, 0), (5, 1)}) en tuple (-1, 5)"""
    sat_list = []
    for var, value in sorted(cpi_xp, key=lambda x: x[0]):
        sat_list.append(var if value == 1 else -var)
    return tuple(sat_list)

# =================== Fonction 1: weakAXpc (Modifiée pour RF) ===================
def weakAXpc(A, cnf_negative, constraints):
    """
    Vérifie si A est une explication suffisante (Weak AXp).
    
    Pour les RF avec Encoding.MUS :
    - cnf_negative représente "Le modèle prédit l'INVERSE de la cible".
    - On cherche à prouver que (A ∧ Contraintes ∧ cnf_negative) est IMPOSSIBLE (UNSAT).
    - Si c'est UNSAT, cela signifie que le modèle est OBLIGÉ de prédire la cible.
    """
    solver = Glucose3()
    
    # 1. Ajouter les contraintes du domaine (théorie)
    for cl in constraints:
        solver.add_clause(list(cl))
        
    # 2. Ajouter l'explication candidate A
    for lit in A:
        solver.add_clause([literal_to_sat(lit)])
        
    # 3. Ajouter la CNF négative (Le monde "interdit")
    for cl in cnf_negative:
        solver.add_clause(list(cl))

    # Résolution :
    # Si solve() renvoie True (SAT) -> On a trouvé un contre-exemple (Mauvaise explication)
    # Si solve() renvoie False (UNSAT) -> Pas de contre-exemple possible (Bonne explication)
    is_sat = solver.solve()
    solver.delete()
    
    return not is_sat  # On retourne True si c'est une explication valide (UNSAT)

# =================== Fonction 2: implies ===================
def implies(A, B, constraints):
    """Teste si A → B sous contraintes C (Subsumption)."""
    solver = Glucose3()
    for clause in constraints:
        solver.add_clause(list(clause))
    for lit in A:
        solver.add_clause([literal_to_sat(lit)])
    
    # Ajouter ¬B
    if B:
        neg_B = [-literal_to_sat(lit) for lit in B]
        solver.add_clause(neg_B)
    
    result = not solver.solve()
    solver.delete()
    return result

# =================== Fonction 3: CPIexplanation ===================
def CPIexplanation(E, infL, supL, cnf_negative, constraints):
    # Test 1: Validité (Est-ce que supL force toujours la prédiction ?)
    if not weakAXpc(supL, cnf_negative, constraints):
        return (True, None)
    
    # Test 2: Minimalité relative
    if not implies(supL, E, constraints):
        return (False, supL)
    
    # Exploration
    supL_minus_infL = supL - infL
    for lit in supL_minus_infL:
        res, counterE = CPIexplanation(E, infL, supL - {lit}, cnf_negative, constraints)
        if not res:
            return (res, counterE)
        infL = infL | {lit}
    
    return (True, None)

# =================== Fonction 4: findCPIexplanation ===================
def findCPIexplanation(v, cnf_negative, constraints):
    E = frozenset(v)
    iteration = 1
    
    while True:
        E_sorted = sorted(E, key=lambda x: (x[0], -x[1]))
        print(f"\nItération {iteration}:")
        print(f"  Taille E: {len(E)}")
        
        # Calcul de L (Littéraux impliqués par E)
        L = set(E)
        for lit in v:
            if lit not in E:
                if implies(E, frozenset({lit}), constraints):
                    L.add(lit)
        L = frozenset(L)
        # print(f"  |L| = {len(L)}")
        
        # Recherche CPI
        res, counterE = CPIexplanation(E, frozenset(), L, cnf_negative, constraints)
        
        if res:
            # Succès
            E_sorted = sorted(E, key=lambda x: (x[0], -x[1]))
            print(f"\n{'='*70}")
            print(f"✓ CPI-Xp trouvée!")
            print(f"  Format Features: {', '.join(f'x{x[0]}={x[1]}' for x in E_sorted)}")
            print(f"  Taille: {len(E)}")
            print(f"{'='*70}")
            return E
        else:
            # Contre-exemple trouvé, on réduit E
            counterE_sorted = sorted(counterE, key=lambda x: (x[0], -x[1]))
            print(f"  -> Réduction trouvée (taille {len(counterE)})")
            E = frozenset(counterE)
            iteration += 1

# ===================== Main =====================
if __name__ == "__main__":
    print("="*70)
    print("Recherche CPI-Xp pour RANDOM FOREST avec Encoding.MUS")
    print("="*70)
    
    dataset_name = Tools.Options.dataset
    df = pd.read_csv(dataset_name + '.csv')
    
    # 1. Modèle Random Forest
    # n_estimators=5 pour que l'exemple tourne vite (la CNF grossit vite avec les RF)
    learner = Learning.Scikitlearn(df, learner_type=Learning.CLASSIFICATION)
    model = learner.evaluate(method=Learning.HOLD_OUT, output=Learning.RF, n_estimators=4, seed=42)
    
    instance, prediction = learner.get_instances(model, n=1, correct=True)
    explainer = Explainer.initialize(model, instance, features_type=dataset_name + '.types')
    reason =explainer.sufficient_reason()

    # 2. Récupération des données
    th = explainer.get_theory()
    v_list = explainer.binary_representation
    constraints = list(th)
    
    # 3. GÉNÉRATION DE LA CNF TYPE 'MUS' (LA PARTIE CLÉ)
    # On utilise l'objet interne _random_forest pour accéder à la méthode to_CNF spécifique
    # Note : Encoding.MUS s'occupe tout seul d'inverser la prédiction pour créer le "monde négatif".
    print(f"Génération de la CNF (Encoding.MUS)... Cela peut prendre un peu de temps.")
    
    target_prediction = explainer.target_prediction
    
    # Appel spécifique demandé
    cnf_negative = explainer._random_forest.to_CNF(
        explainer._instance, 
        explainer._binary_representation,
        target_prediction, 
        tree_encoding=Encoding.MUS
    )
    
    print(f"CNF générée. Nombre de clauses : {len(cnf_negative)}")
    # Conversion instance
    x = frozenset((abs(lit), 1 if lit > 0 else 0) for lit in v_list)
    
    # 4. Lancement de la recherche
    try:
        print("\n" + "="*70)
        print("PHASE 1: Recherche d'une CPI-Xp")
        print("="*70)
        
        cpi_xp = findCPIexplanation(x, cnf_negative, constraints)
        cpi_xp_sat = cpi_xp_to_sat_format(cpi_xp)
        print("\n" + "="*70)
        print("Raison pyxai")
        print("="*70)
        print(f"Raison originale: {reason}")
        print(f"Est une raison valide: {explainer.is_reason(reason)}")
        print(f"Features: {explainer.to_features(reason)}")

        print("\n" + "="*70)
        print("VÉRIFICATION FINALE")
        print("="*70)
        print("cpi_xp (SAT):", cpi_xp_sat)
        print(f"Features cpi_xp: {explainer.to_features(cpi_xp_sat)}")
        
        # Vérification avec l'outil officiel PyXAI
        is_valid = explainer.is_reason(cpi_xp_sat)
        print(f"Est une raison valide (PyXAI) ? {is_valid}")
        print("instance binarisé", explainer.binary_representation)
        print("th",th)
    except Exception as e:
        print(f"\nErreur lors de la recherche: {e}")
        import traceback
        traceback.print_exc()