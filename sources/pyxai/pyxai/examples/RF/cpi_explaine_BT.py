from pysat.solvers import Glucose3
from pyxai.sources.solvers.ORTOOLS.IsImplicantBT import IsImplicantBT
import pandas as pd

# ===================== Utilities =====================

def literal_to_sat(literal):
    """
    Converts a (var, value) tuple to a SAT literal:
    Example: (1, 1) -> 1, (1, 0) -> -1
    """
    return literal[0] if literal[1] == 1 else -literal[0]

def cpi_xp_to_sat_format(cpi_xp):
    """
    Converts a frozenset({(1, 0), (5, 1)}) into a tuple (-1, 5)
    """
    sat_list = []
    for var, value in sorted(cpi_xp, key=lambda x: x[0]):
        sat_list.append(var if value == 1 else -var)
    return tuple(sat_list)

# =================== 1. Subsumption Test (implies_BT) ===================

def implies_BT(A, B, solver_subsumption):
    """
    Checks if set A implies set B under the constraints of domain theory Sigma.
    Note: solver_subsumption must be an instance of Glucose3.
    """
    if not B:
        return True

    # Base assumptions: all literals in A (converted to SAT format)
    base_assumptions = [literal_to_sat(lit) for lit in A]

    for lit in B:
        # Logical test: A and Sigma => lit <=> (A and Sigma and not-lit) is UNSAT
        lit_sat = literal_to_sat(lit)
        
        # If the solver finds a solution (SAT) with not-lit, 
        # then A does not imply b_lit.
        if solver_subsumption.solve(assumptions=base_assumptions + [-lit_sat]):
            return False

    return True

# =================== 2. Counter-Example Search (Recursive) ===================

def CPIexplanation_BT(E, infL, supL, explainer, solver_subsumption):
    """
    Recursive search for a counter-example.
    'explainer' is used for the BT model (MIP-based).
    'solver_subsumption' is used for the SAT theory (logical consistency).
    """
    
    # 1. Model Validity: Call to the custom BT implicant method
    # supL is converted to a list of signed literals for the explainer
    supL_list = [literal_to_sat(lit) for lit in supL]
    if not explainer.is_implicant_BT(supL_list):
        return (True, None)

    # 2. Subsumption (Generality): Using the SAT solver
    if not implies_BT(supL, E, solver_subsumption):
        return (False, supL)

    # Search Exploration
    for lit in (supL - infL):
        # Recursive call passing all 5 required arguments
        res, counterE = CPIexplanation_BT(E, infL, supL - {lit}, explainer, solver_subsumption)
        if not res:
            return (res, counterE)
        infL = infL | {lit}

    return (True, None)

# =================== 3. Main Algorithm ===================

def findCPIexplanation_BT(v, explainer):
    """
    Computes the CPI-Xp for Boosted Trees.
    v: complete instance (frozenset of tuples)
    """
    E = frozenset(v)
    
    # Retrieve domain theory clauses (Sigma)
    theory_clauses = list(explainer._boosted_trees.get_theory(explainer._binary_representation))
    
    # Initialize SAT solver for the domain theory only
    with Glucose3() as solver_subsumption:
        for cl in theory_clauses:
            solver_subsumption.add_clause(list(cl))

        while True:
            # Step 1: Compute L (Closure of E)
            L = set(E)
            for lit in (v - E):
                # Pass the SAT solver to verify implications
                if implies_BT(E, frozenset({lit}), solver_subsumption):
                    L.add(lit)
            L = frozenset(L)

            # Step 2: Search for a counter-example
            # Ensure all 5 arguments are passed correctly
            res, counterE = CPIexplanation_BT(E, frozenset(), L, explainer, solver_subsumption)

            if res:
                return E # E is a valid CPI-Xp
            else:
                E = frozenset(counterE)
                if len(E) == 0: 
                    break
    return None