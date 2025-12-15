from pysat.solvers import Glucose3
import json
import time
import numpy as np
from pyxai.sources.core.structure.type import Encoding # Important pour Encoding.MUS

# ===================== Utilities =====================

def literal_to_sat(literal):
    """
    Converts a feature tuple (var, value) to SAT format.
    (1, 1) -> 1 (Positive literal)
    (1, 0) -> -1 (Negative literal)
    """
    return literal[0] if literal[1] == 1 else -literal[0]


def sat_to_literal(sat_lit):
    """
    Converts a SAT integer back to a feature tuple.
    1 -> (1, 1)
    -1 -> (1, 0)
    """
    return (abs(sat_lit), 1 if sat_lit > 0 else 0)


def cpi_xp_to_sat_format(cpi_xp):
    """
    Converts a frozenset of features to a tuple of SAT integers.
    Example: frozenset({(1, 0), (5, 1)}) -> (-1, 5)
    """
    sat_list = []
    for var, value in sorted(cpi_xp, key=lambda x: x[0]):
        sat_list.append(var if value == 1 else -var)
    return tuple(sat_list)


# =================== Function 1: weakAXpc (Optimized) ===================

def weakAXpc(A, cnf_formula, solver):
    """
    Checks if the set of literals A implies every clause in the CNF formula.
    
    Logic:
        For each clause C = (l1 v l2), we test if (A ^ -C) is UNSAT.
        Since -C = -(l1 v l2) = (-l1 ^ -l2) (De Morgan's laws),
        we can pass [-l1, -l2] as assumptions to the solver.
    
    Args:
        A: The antecedent set of literals.
        cnf_formula: The formula in CNF format (list of clauses).
        solver: The shared Glucose3 solver instance.
    
    Returns:
        True if A implies all clauses in cnf_formula, False otherwise.
    """
    # Convert A to SAT assumptions once
    base_assumptions = [literal_to_sat(lit) for lit in A]

    for clause in cnf_formula:
        # Negate the clause to test implication by contradiction
        # negation of (l1 v l2) is (-l1 ^ -l2)
        negated_clause_assumptions = [-lit for lit in clause]
        
        full_assumptions = base_assumptions + negated_clause_assumptions

        # If SAT, a model exists where A is True and Clause is False
        # Therefore, A does NOT imply this clause.
        if solver.solve(assumptions=full_assumptions):
            return False

    return True


# =================== Function 2: implies (Corrected) ===================

def implies(A, B, solver):
    """
    Tests if A => B (where B is a conjunction of literals) under the solver's constraints.
    
    CRITICAL LOGIC:
        We cannot test the entire set B at once using standard assumptions.
        A => (b1 ^ b2) is equivalent to (A => b1) AND (A => b2).
        Therefore, we must iterate through each literal in B.

    Args:
        A: Antecedent set (frozenset of tuples).
        B: Consequent set (frozenset of tuples).
        solver: The shared Glucose3 solver instance loaded with domain constraints.

    Returns:
        True if A implies every literal in B, False otherwise.
    """
    # Prepare base assumptions (A is True)
    base_assumptions = [literal_to_sat(lit) for lit in A]
    
    # If B is empty, A trivially implies True
    if not B:
        return True

    for lit in B:
        # We test: A => lit
        # By contradiction: Is (A ^ -lit) UNSAT?
        lit_sat = literal_to_sat(lit)
        
        # Add -lit to assumptions
        current_assumptions = base_assumptions + [-lit_sat]
        
        # If the solver finds a solution (SAT), then A does NOT imply this literal.
        if solver.solve(assumptions=current_assumptions):
            return False 

    # If no counter-example was found for any literal in B
    return True


# =================== Function 3: CPIexplanation (Recursive) ===================

def CPIexplanation(E, infL, supL, cnf_formula, solver):
    """
    Recursively searches for a counter-example to E within the lattice interval [infL, supL].
    
    Args:
        E: The candidate explanation.
        infL: Lower bound of the search lattice.
        supL: Upper bound of the search lattice.
        cnf_formula: The original formula.
        solver: The shared solver instance.
        
    Returns:
        (bool, set): (True, None) if E is a CPI-Xp.
                     (False, C) if a counter-example C is found.
    """
    # Step 1: If supL is not a weak abductive explanation, no subset can be a counter-example.
    if not weakAXpc(supL, cnf_formula, solver):
        return (True, None)
    
    # Step 2: If supL does not imply E (i.e., supL does not subsume E logic-wise),
    # then supL itself is a counter-example.
    if not implies(supL, E, solver):
        return (False, supL)
    
    # Step 3: Recursive exploration
    # Iterate over literals present in supL but not in infL
    supL_minus_infL = supL - infL
    
    for lit in supL_minus_infL:
        # Try to find a counter-example without this literal
        res, counterE = CPIexplanation(E, infL, supL - {lit}, cnf_formula, solver)
        
        if not res:
            # Counter-example found
            return (res, counterE)
        
        # Otherwise, this literal is necessary; add it to the lower bound
        infL = infL | {lit}
    
    return (True, None)


# =================== Function 4: findCPIexplanation (Main) ===================

def findCPIexplanation(v, cnf_formula, constraints):
    """
    Computes a CPI-Xp (Coverage-based Prime Implicant Explanation) for instance v.
    
    Architecture:
        Uses a SINGLE solver instance with 'assumptions' to prevent memory leaks 
        and segmentation faults associated with repeated solver instantiation.
    
    Args:
        v: The instance to explain (full set of features).
        cnf_formula: The decision tree/model converted to CNF.
        constraints: Background knowledge/constraints.
        
    Returns:
        frozenset: The computed CPI-Xp explanation.
    """
    E = frozenset(v)
    
    # Initialize the solver once using a context manager
    with Glucose3() as solver:
        # 1. Load domain constraints (background knowledge)
        for clause in constraints:
            solver.add_clause(list(clause))
            
        iteration = 1
        found = False
        
        while not found:
            # Calculate L: The set of literals in v implied by E under constraints.
            # L = { l in v | E ^ Constraints |= l }
            L = set(E)
            for lit in v:
                if lit not in E:
                    # Check implication literal by literal
                    if implies(E, frozenset({lit}), solver):
                        L.add(lit)
            
            L = frozenset(L)
            
            # Search for a counter-example that subsumes E
            is_cpi, counter_example = CPIexplanation(E, frozenset(), L, cnf_formula, solver)
            
            if is_cpi:
                # E is a CPI-Xp
                return E
            else:
                # E is not sufficient; update E with the found counter-example and iterate
                E = frozenset(counter_example)
                iteration += 1
                
    return None