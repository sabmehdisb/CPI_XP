from pysat.solvers import Glucose3
import json
from pyxai import Learning, Explainer, Tools
import pandas as pd
from sklearn.model_selection import train_test_split
import time
import numpy as np

# ===================== Utilitaires =====================
def literal_to_sat(literal):
    """Convertit (var, value) en SAT: (1,1)->1, (1,0)->-1"""
    return literal[0] if literal[1] == 1 else -literal[0]

def sat_to_literal(sat_lit):
    """Convertit SAT en tuple: 1->(1,1), -1->(1,0)"""
    return (abs(sat_lit), 1 if sat_lit > 0 else 0)

def cpi_xp_to_sat_format(cpi_xp):
    """Convertit frozenset({(1, 0), (5, 1), (6, 0)}) en tuple (-1, 5, -6)"""
    sat_list = []
    for var, value in sorted(cpi_xp, key=lambda x: x[0]):
        sat_list.append(var if value == 1 else -var)
    return tuple(sat_list)

# =================== Fonction 1: weakAXpc ===================
def weakAXpc(A,forme_cnf,constraints):
    
    for clause in forme_cnf:
        s = Glucose3()
        # contraintes
        for cl in constraints:
            s.add_clause(list(cl))
        # A
        for lit in A:
            s.add_clause([literal_to_sat(lit)])
        # ¬clause
        for lit in clause:
            s.add_clause([-lit])

        if s.solve():   # SAT → cette clause n’est pas impliquée
            s.delete()
            return False
        s.delete()

    return True

# def weakAXpc(A, forme_cnf, constraints):
#     """
#     Teste si A est un weak AXpc.
#     Retourne True si ∄x: A(x) ∧ C(x) ∧ (κ(x) ≠ c)
    
#     CORRECTION: On doit tester si pour TOUTES les clauses de la CNF (qui encode κ(x)=c),
#     il n'existe pas d'instance satisfaisant A, C et violant au moins une clause.
#     """
#     solver = Glucose3()
    
#     # Ajouter les contraintes C
#     for clause in constraints:
#         solver.add_clause(list(clause))
    
#     # Ajouter A (l'explication candidate)
#     for lit in A:
#         solver.add_clause([literal_to_sat(lit)])
    
#     # Ajouter la négation de la CNF (κ(x) ≠ c)
#     # La CNF est κ(x) = c, donc ¬CNF signifie qu'au moins une clause est fausse
#     # On doit tester si A ∧ C → CNF
#     # C'est équivalent à tester si A ∧ C ∧ ¬CNF est UNSAT
    
#     # Pour chaque clause, on teste si on peut la violer
#     for clause in forme_cnf:
#         s = Glucose3()
        
#         # Ajouter contraintes
#         for cl in constraints:
#             s.add_clause(list(cl))
        
#         # Ajouter A
#         for lit in A:
#             s.add_clause([literal_to_sat(lit)])
        
#         # Tester si on peut violer cette clause (tous les littéraux sont faux)
#         for lit in clause:
#             s.add_clause([-lit])
        
#         if s.solve():  # Si SAT, alors on peut violer une clause → pas un weak AXpc
#             s.delete()
#             solver.delete()
#             return False
#         s.delete()
    
#     solver.delete()
#     return True

# =================== Fonction 2: implies ===================
def implies(A, B, constraints):
    """
    Teste si A → B sous contraintes C.
    Retourne True si ∄y: A(y) ∧ C(y) ∧ ¬B(y)
    
    CORRECTION: La logique était correcte, juste quelques optimisations.
    """
    solver = Glucose3()
    
    # Ajouter les contraintes
    for clause in constraints:
        solver.add_clause(list(clause))
    
    # Ajouter A
    for lit in A:
        solver.add_clause([literal_to_sat(lit)])
    
    # Ajouter ¬B (au moins un littéral de B est faux)
    if B:
        neg_B = [-literal_to_sat(lit) for lit in B]
        solver.add_clause(neg_B)
    
    result = not solver.solve()  # UNSAT → A implique B
    solver.delete()
    return result

# =================== Fonction 3: CPIexplanation ===================
def CPIexplanation(E, infL, supL, forme_cnf, constraints):
    """
    Cherche un contre-exemple à E dans l'intervalle [infL, supL].
    
    CORRECTION: L'algorithme était globalement correct, mais il faut bien
    gérer les ensembles et s'assurer que supL \ infL est bien calculé.
    """
    # Test 1: Si supL n'est pas un weak AXpc, aucun sous-ensemble ne peut être contre-exemple
    if not weakAXpc(supL, forme_cnf, constraints):
        return (True, None)
    
    # Test 2: Si supL ne subsume pas E, alors supL est un contre-exemple
    if not implies(supL, E, constraints):
        return (False, supL)
    
    # Test 3: Exploration récursive
    # Pour chaque littéral dans supL mais pas dans infL
    supL_minus_infL = supL - infL
    
    for lit in supL_minus_infL:
        # Tester s'il existe un contre-exemple sans utiliser ce littéral
        res, counterE = CPIexplanation(E, infL, supL - {lit}, forme_cnf, constraints)
        
        if not res:  # Contre-exemple trouvé
            return (res, counterE)
        
        # Sinon, ce littéral doit être dans tout contre-exemple
        infL = infL | {lit}
    
    return (True, None)

# =================== Fonction 4: findCPIexplanation ===================
def findCPIexplanation(v, forme_cnf, constraints):
    """
    Trouve une CPI-Xp pour κ(v) = c.
    
    CORRECTIONS:
    1. Initialisation correcte de E comme frozenset
    2. Calcul correct de L (ensemble des littéraux impliqués)
    3. Gestion correcte de l'itération
    """
    E = frozenset(v)  # CORRECTION: S'assurer que E est un frozenset
    iteration = 1
    res=False
    while res == False :
        E_sorted = sorted(E, key=lambda x: (x[0], -x[1]))
        print(f"\nItération {iteration}:")
        print(f"  E = {{{', '.join(f'{x[0]}={x[1]}' for x in E_sorted)}}}")
        print(f"  |E| = {(E)}")
        
        # CORRECTION: Calculer L_{v,E,C} correctement
        # L contient tous les littéraux de v qui sont impliqués par E sous C
        L = set(E)
        
        for lit in v:
            if lit not in E:
                # Tester si E ∧ C → lit
                if implies(E, frozenset({lit}), constraints):
                    L.add(lit)
        
        L = frozenset(L)
        print(f"  |L| = {(L)}")
        
        # Tester si E est une CPI-Xp
        res, counterE = CPIexplanation(E, frozenset(), L, forme_cnf, constraints)
        
        if res:
            # E est une CPI-Xp!
            E_sorted = sorted(E, key=lambda x: (x[0], -x[1]))
            print(f"\n{'='*70}")
            print(f"✓ CPI-Xp trouvée!")
            print(f"  {{{', '.join(f'{x[0]}={x[1]}' for x in E_sorted)}}}")
            print(f"  Taille: {len(E)} littéraux")
            print(f"{'='*70}")
            return E
        else:
            # On a trouvé un contre-exemple qui subsume E
            counterE_sorted = sorted(counterE, key=lambda x: (x[0], -x[1]))
            print(f"  → Contre-exemple: {{{', '.join(f'{x[0]}={x[1]}' for x in counterE_sorted)}}}")
            E = frozenset(counterE)  # CORRECTION: Convertir en frozenset
            iteration += 1

# ===================== Test =====================
if __name__ == "__main__":
    print("="*70)
    print("Recherche de CPI-Xp avec solveur SAT - Dataset Balance")
    print("="*70)
    
    # Charger les données
    dataset_name=Tools.Options.dataset
    df = pd.read_csv(dataset_name + '.csv')
    
    # Apprentissage du modèle
    dt_learner = Learning.Scikitlearn(df, learner_type=Learning.CLASSIFICATION)
    dt_model = dt_learner.evaluate(method=Learning.HOLD_OUT, output=Learning.DT, seed=0)
    
    # Obtenir une instance
    instance, prediction = dt_learner.get_instances(dt_model, n=1,correct=True)
    
    # Initialiser l'explainer
    dt_explainer = Explainer.initialize(dt_model, instance, features_type=dataset_name + '.types')
    
    # Récupérer les données nécessaires
    th = dt_explainer.get_theory()
    v_list = dt_explainer.binary_representation
    c = prediction
    forme_cnf = dt_explainer.to_CNF()
    constraints = list(th)
    
    # CORRECTION: Créer x comme frozenset de tuples (var, value)
    x = frozenset((abs(lit), 1 if lit > 0 else 0) for lit in v_list)
    
    print(f"\nInstance binaire: {sorted(v_list, key=abs)}")
    print(f"Prédiction: {c}")
    print(f"Nombre de clauses CNF: {len(forme_cnf)}")
    print(f"Nombre de contraintes: {len(constraints)}")
    
    # Trouver la CPI-Xp
    try:
        print("\n" + "="*70)
        print("PHASE 1: Recherche d'une CPI-Xp")
        print("="*70)
        
        cpi_xp = findCPIexplanation(x, forme_cnf, constraints)
        cpi_xp_sat = cpi_xp_to_sat_format(cpi_xp)
        
        print("\n" + "="*70)
        print("COMPARAISON AVEC SUFFICIENT REASON")
        print("="*70)
        print('th',th)
        print("cnf",forme_cnf)
        sufficient_reason = dt_explainer.sufficient_reason()
        print("\nsufficient_reason:", sufficient_reason)
        print(f"sufficient_reason to Features: {dt_explainer.to_features(sufficient_reason)}")
        print("is sufficient_reason:", dt_explainer.is_sufficient_reason(sufficient_reason))
        print(f"Est une raison valide: {dt_explainer.is_reason(sufficient_reason)}")
        print("\n" + "#"*70)
        print("cpi_xp:", cpi_xp_sat)
        print(f"cpi_xp to Features: {dt_explainer.to_features(cpi_xp_sat)}")
        print("is sufficient_reason cpi_xp:", dt_explainer.is_sufficient_reason(cpi_xp_sat))
        print(f"Est une raison valide: {dt_explainer.is_reason(cpi_xp_sat)}")
        
    except Exception as e:
        print(f"\nErreur lors de la recherche: {e}")
        import traceback
        traceback.print_exc()