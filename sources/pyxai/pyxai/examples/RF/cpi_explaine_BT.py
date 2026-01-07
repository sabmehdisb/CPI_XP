from pysat.solvers import Glucose3
from pyxai.sources.solvers.ORTOOLS.IsImplicantBT import IsImplicantBT
import pandas as pd

# =================== 1. Test de Subsomption (implies_BT) ===================
from pysat.solvers import Glucose3

# ===================== Utilities =====================
def literal_to_sat(literal):
    """Convertit (var, value) en littéral SAT : (1,1)->1, (1,0)->-1"""
    return literal[0] if literal[1] == 1 else -literal[0]
def cpi_xp_to_sat_format(cpi_xp):
    """Convert frozenset({(1, 0), (5, 1)}) into tuple (-1, 5)"""
    sat_list = []
    for var, value in sorted(cpi_xp, key=lambda x: x[0]):
        sat_list.append(var if value == 1 else -var)
    return tuple(sat_list)
# =================== 1. Test de Subsomption (SAT) ===================
def implies_BT(A, B, solver_subsumption):
    """
    Vérifie si A implique B sous les contraintes de la théorie Sigma.
    Note : solver_subsumption doit être une instance de Glucose3.
    """
    if not B:
        return True

    # Hypothèse de base : tout ce qui est dans A (format SAT)
    base_assumptions = [literal_to_sat(lit) for lit in A]

    for lit in B:
        # Test : A et Sigma => lit  <=> (A et Sigma et non-lit) est UNSAT
        lit_sat = literal_to_sat(lit)
        
        # Si le solveur trouve une solution (SAT) avec non-lit, 
        # alors A n'implique pas b_lit.
        if solver_subsumption.solve(assumptions=base_assumptions + [-lit_sat]):
            return False

    return True

# =================== 2. Recherche de Contre-Exemple (Récursif) ===================
def CPIexplanation_BT(E, infL, supL, explainer, solver_subsumption):
    """
    Recherche récursive. 
    L'argument 'explainer' sert pour le modèle BT.
    L'argument 'solver_subsumption' sert pour la théorie SAT.
    """
    
    # 1. Validité du Modèle : Appel à votre méthode is_implicant_BT
    # On convertit supL en liste de littéraux signés pour votre méthode
    supL_list = [literal_to_sat(lit) for lit in supL]
    if not explainer.is_implicant_BT(supL_list):
        return (True, None)

    # 2. Subsomption (Généralité) : Via le solveur SAT
    # ATTENTION : On passe bien 'solver_subsumption' ici
    if not implies_BT(supL, E, solver_subsumption):
        return (False, supL)

    # Exploration
    for lit in (supL - infL):
        # ATTENTION : On passe bien les 5 arguments dans l'appel récursif
        res, counterE = CPIexplanation_BT(E, infL, supL - {lit}, explainer, solver_subsumption)
        if not res:
            return (res, counterE)
        infL = infL | {lit}

    return (True, None)

# =================== 3. Algorithme Principal ===================
def findCPIexplanation_BT(v, explainer):
    """
    Calcule la CPI-Xp pour les Boosted Trees.
    v : instance complète (frozenset de tuples)
    """
    E = frozenset(v)
    
    # Récupération des clauses de la théorie du domaine
    theory_clauses = list(explainer._boosted_trees.get_theory(explainer._binary_representation))
    
    # Initialisation du solveur SAT uniquement pour la théorie
    with Glucose3() as solver_subsumption:
        for cl in theory_clauses:
            solver_subsumption.add_clause(list(cl))

        while True:
            # Calcul de L (Clôture de E)
            L = set(E)
            for lit in (v - E):
                # On passe bien le solveur SAT
                if implies_BT(E, frozenset({lit}), solver_subsumption):
                    L.add(lit)
            L = frozenset(L)

            # Recherche de contre-exemple
            # /!\ BIEN PASSER LES 5 ARGUMENTS ICI /!\
            res, counterE = CPIexplanation_BT(E, frozenset(), L, explainer, solver_subsumption)

            if res:
                return E # E est une CPI-Xp
            else:
                E = frozenset(counterE)
                if len(E) == 0: break
    return None