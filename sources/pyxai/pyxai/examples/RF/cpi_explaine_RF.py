from pysat.solvers import Glucose3
import json
from pyxai import Learning, Explainer, Tools
from pyxai.sources.core.structure.type import Encoding  # Important for Encoding.MUS
import pandas as pd
from sklearn.model_selection import train_test_split
import time
import numpy as np

# ===================== Utilities =====================
def literal_to_sat(literal):
    """Convert (var, value) to SAT literal: (1,1)->1, (1,0)->-1"""
    return literal[0] if literal[1] == 1 else -literal[0]

def cpi_xp_to_sat_format(cpi_xp):
    """Convert frozenset({(1, 0), (5, 1)}) into tuple (-1, 5)"""
    sat_list = []
    for var, value in sorted(cpi_xp, key=lambda x: x[0]):
        sat_list.append(var if value == 1 else -var)
    return tuple(sat_list)

# =================== Function 1: weakAXpc (Optimized) ===================
def weakAXpc(A, solver_validity):
    """
    Checks whether A is a sufficient explanation.
    solver_validity already contains: Constraints + CNF_Negative.

    Logic: If (A + Constraints + Neg) is UNSAT, then A enforces the prediction.
    """
    # Pass A as assumptions
    assumptions = [literal_to_sat(lit) for lit in A]

    # If solve() returns False (UNSAT), then A is a valid explanation
    return not solver_validity.solve(assumptions=assumptions)

# =================== Function 2: implies (Optimized) ===================
def implies(A, B, solver_subsumption):
    """
    Tests whether A → B under constraints (Subsumption).
    solver_subsumption contains only the Constraints (no CNF Negative).

    We use an iterative logic to avoid modifying the solver.
    """
    base_assumptions = [literal_to_sat(lit) for lit in A]

    if not B:
        return True

    for lit in B:
        # Test: A => lit  <==> (A and not-lit) is UNSAT
        lit_sat = literal_to_sat(lit)
        current_assumptions = base_assumptions + [-lit_sat]

        # If a solution is found (SAT), then A does NOT imply this literal
        if solver_subsumption.solve(assumptions=current_assumptions):
            return False

    return True

# =================== Function 3: CPIexplanation (Recursive) ===================
def CPIexplanation(E, infL, supL, solver_validity, solver_subsumption):
    # Test 1: Validity (Does supL always enforce the prediction?)
    # We use the solver loaded with cnf_negative
    if not weakAXpc(supL, solver_validity):
        return (True, None)

    # Test 2: Relative minimality
    # We use the solver loaded only with the constraints
    if not implies(supL, E, solver_subsumption):
        return (False, supL)

    # Exploration
    supL_minus_infL = supL - infL
    for lit in supL_minus_infL:
        res, counterE = CPIexplanation(
            E, infL, supL - {lit}, solver_validity, solver_subsumption
        )
        if not res:
            return (res, counterE)
        infL = infL | {lit}

    return (True, None)

# =================== Function 4: findCPIexplanation (Main) ===================
def findCPIexplanation(v, cnf_negative, constraints):
    E = frozenset(v)
    iteration = 1

    # Initialization of BOTH solvers (to avoid recreating them each time)
    # solver_validity: to check weakAXpc (Constraints + CNF Negative)
    # solver_subsumption: to check implies (Constraints only)
    with Glucose3() as solver_validity, Glucose3() as solver_subsumption:

        # 1. Load clauses into solver_validity
        for cl in constraints:
            solver_validity.add_clause(list(cl))
        for cl in cnf_negative:
            solver_validity.add_clause(list(cl))

        # 2. Load clauses into solver_subsumption
        for cl in constraints:
            solver_subsumption.add_clause(list(cl))

        while True:
            E_sorted = sorted(E, key=lambda x: (x[0], -x[1]))

            # Compute L (literals implied by E)
            L = set(E)
            for lit in v:
                if lit not in E:
                    # Use the subsumption solver here
                    if implies(E, frozenset({lit}), solver_subsumption):
                        L.add(lit)
            L = frozenset(L)

            # Search for CPI using both solvers
            res, counterE = CPIexplanation(
                E, frozenset(), L, solver_validity, solver_subsumption
            )

            if res:
                # Success
                return E
            else:
                # Counter-example found, reduce E
                E = frozenset(counterE)
                iteration += 1

                # Safety to avoid infinite loops (optional)
                if len(E) == 0:
                    break

    # Return None as requested if no explanation is found
    return None
