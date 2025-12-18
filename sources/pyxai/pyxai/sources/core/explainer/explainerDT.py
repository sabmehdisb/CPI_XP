import time

from pyxai.sources.core.explainer.Explainer import Explainer
from pyxai.sources.core.structure.decisionTree import DecisionTree
from pyxai.sources.core.structure.type import PreferredReasonMethod, TypeTheory, ReasonExpressivity
from pyxai.sources.core.tools.encoding import CNFencoding
from pyxai.sources.core.tools.utils import compute_weight
from pyxai.sources.solvers.COMPILER.D4Solver import D4Solver
from pyxai.sources.solvers.MAXSAT.OPENWBOSolver import OPENWBOSolver
from pyxai.sources.solvers.SAT.glucoseSolver import GlucoseSolver
from pyxai import Tools
from pysat.solvers import Glucose3
import random

import c_explainer


class ExplainerDT(Explainer):

    def __init__(self, tree, instance=None):
        """Create object dedicated to finding explanations from a decision tree ``tree`` and an instance ``instance``.

        Args:
            tree (DecisionTree): The model in the form of a DecisionTree object.
            instance (:obj:`list` of :obj:`int`, optional): The instance (an observation) on which explanations must be calculated. Defaults to None.
        """
        super().__init__()
        self._tree = tree  # The decision _tree.
        if instance is not None:
            self.set_instance(instance)
        self.c_rectifier = None
        self._additional_theory = []
        self.c_RF = None

    @property
    def tree(self):
        """Return the model, the associated tree"""
        return self._tree

    def set_instance(self, instance):
        super().set_instance(instance)
        self._n_sufficient_reasons = None

    def _to_binary_representation(self, instance):
        return self._tree.instance_to_binaries(instance)

    def is_implicant(self, binary_representation, *, prediction=None):
        if prediction is None:
            prediction = self.target_prediction
        binary_representation = self.extend_reason_with_theory(binary_representation)
        return self._tree.is_implicant(binary_representation, prediction)

    def predict(self, instance):
        return self._tree.predict_instance(instance)

    def simplify_reason(self, binary_representation):
        glucose = GlucoseSolver()
        glucose.add_clauses(self.get_theory())

        present = [True for _ in binary_representation]
        position = []
        for i, lit in enumerate(binary_representation):
            if present[i] is False:
                continue
            status, propagated = glucose.propagate([lit])
            assert (status is not False)
            print(propagated)
            for p in propagated:
                if p != lit and p in binary_representation:
                    present[binary_representation.index(p)] = False
        # print(present)
        return [lit for i, lit in enumerate(binary_representation) if present[i]]

    def to_features(self, binary_representation, *, eliminate_redundant_features=True, details=False, contrastive=False,
                    without_intervals=False):
        """_summary_

        Args:
            binary_representation (_type_): _description_

        Returns:
            _type_: _description_
        """
        return self._tree.to_features(binary_representation, details=details,
                                      eliminate_redundant_features=eliminate_redundant_features,
                                      contrastive=contrastive, without_intervals=without_intervals,
                                      feature_names=self.get_feature_names())

    def add_clause_to_theory(self, clause):
        self._additional_theory.append(tuple(clause))
        self._theory = True
        self.c_rectifier = None
        self.c_RF = None
        self._glucose = None

    def direct_reason(self):
        """
        Returns:
            _type_: _description_
        """
        if self._instance is None:
            raise ValueError("Instance is not set")

        self._elapsed_time = 0
        direct_reason = self._tree.direct_reason(self._instance)
        if any(not self._is_specific(lit) for lit in direct_reason):
            direct_reason = None  # The reason contains excluded features
        else:
            direct_reason = Explainer.format(direct_reason)

        self._visualisation.add_history(self._instance, self.__class__.__name__, self.direct_reason.__name__,
                                        direct_reason)
        return direct_reason

    def contrastive_reason(self, *, n=1):
        if self._instance is None:
            raise ValueError("Instance is not set")
        self._elapsed_time = 0
        cnf = self._tree.to_CNF(self._instance)
        core = CNFencoding.extract_core(cnf, self._binary_representation)
        core = [c for c in core if all(self._is_specific(lit) for lit in c)]  # remove excluded
        tmp = sorted(core, key=lambda clause: len(clause))
        if self._theory:  # Remove bad contrastive wrt theory
            contrastives = []
            for c in tmp:
                extended = self.extend_reason_with_theory([-lit for lit in c])
                if (len(extended) > 0):  # otherwise unsat => not valid with theory
                    contrastives.append(c)
        else:
            contrastives = tmp

        contrastives = Explainer.format(contrastives, n) if type(n) != int else Explainer.format(contrastives[:n], n)
        self._visualisation.add_history(self._instance, self.__class__.__name__, self.contrastive_reason.__name__,
                                        contrastives)
        return contrastives

    def necessary_literals(self):
        if self._instance is None:
            raise ValueError("Instance is not set")
        self._elapsed_time = 0
        cnf = self._tree.to_CNF(self._instance, target_prediction=self.target_prediction)
        core = CNFencoding.extract_core(cnf, self._binary_representation)
        # DO NOT remove excluded features. If they appear, they explain why there is no sufficient

        literals = sorted({lit for _, clause in enumerate(core) if len(clause) == 1 for lit in clause})
        # self.add_history(self._instance, self.__class__.__name__, self.necessary_literals.__name__, literals)
        return literals

    def relevant_literals(self):
        if self._instance is None:
            raise ValueError("Instance is not set")
        self._elapsed_time = 0
        cnf = self._tree.to_CNF(self._instance, target_prediction=self.target_prediction)
        core = CNFencoding.extract_core(cnf, self._binary_representation)

        literals = [lit for _, clause in enumerate(core) if len(clause) > 1 for lit in clause if
                    self._is_specific(lit)]  # remove excluded features
        # self.add_history(self._instance, self.__class__.__name__, self.relevant_literals.__name__, literals)
        return list(dict.fromkeys(literals))

    def _excluded_features_are_necesssary(self, prime_cnf):
        return any(not self._is_specific(lit) for lit in prime_cnf.necessary)
    def get_suppression_order2(
        self, instance, th, strategy="priority_order", seed=None, ordre_features=None
    ):
        """
        Return the literal removal order according to the chosen strategy.

        Args:
            instance: List of literals
            th: Theory (clauses)
            strategy: Removal strategy to use
            seed: Seed for random strategy (optional)

        Returns:
            Ordered list of literals according to the strategy
        """
        if strategy == "priority_order":
            chains_by_feature = self.get_feature_chain_lists_with_positive_first(th, instance)
            priority_features = self.merge_chains_and_instance2(
                chains_by_feature, instance, ordre_features=ordre_features
            )
            return priority_features

        elif strategy == "beginning_to_end":
            # Remove from beginning to end (original order)
            return list(instance)

        elif strategy == "end_to_beginning":
            # Remove from end to beginning (reverse order)
            return list(reversed(instance))

        elif strategy == "random":
            # Random removal
            random_order = list(instance)
            if seed is not None:
                random.seed(seed)
            random.shuffle(random_order)
            return random_order

        else:
            raise ValueError(f"Unknown strategy: {strategy}")


    def get_feature_chain_lists_with_positive_first(self, th, instance):
        """
        For each feature, build the standard chain,
        then reorder it to put reversed positives first.
        """
        # 1) Group clauses by feature
        feature_groups = {}
        for a, b in th:
            feature = self.to_features((a,))[0].split()[0]
            feature_groups.setdefault(feature, []).append((a, b))

        result = {}
        for feature, clauses in feature_groups.items():
            chain = []

            # Traverse clauses in reverse order
            for a, b in reversed(clauses):
                a_in = a in instance
                b_in = b in instance

                # XOR condition: exactly one literal is in the instance
                if a_in and not b_in:
                    if -b not in chain:
                        chain.append(-b)
                    if a not in chain:
                        chain.append(a)

                elif b_in and not a_in:
                    if b not in chain:
                        chain.append(b)
                    if -a not in chain:
                        chain.append(-a)

            # Apply final ordering if chain is non-empty
            if chain:
                result[feature] = self.reorder_chain(chain)

        return result


    def reorder_chain(self, chain):
        """
        Return a reordered chain where:
        - positive literals (reversed) come first
        - negative literals follow in original order
        """
        positives = [x for x in chain if x > 0]
        negatives = [x for x in chain if x < 0]
        positives.reverse()
        return positives + negatives


    def merge_chains_and_instance2(self, chains_by_feature, instance, ordre_features=None):
        """
        Merge priority chains and remaining instance literals.
        If a feature order is provided, it is respected first.
        """
        merged = []

        # 1. Determine full feature traversal order
        priority_features = list(ordre_features) if ordre_features else []
        all_features = list(chains_by_feature.keys())
        other_features = [f for f in all_features if f not in priority_features]
        features_to_process = priority_features + other_features

        # 2. Preprocess: group instance literals by feature
        instance_by_feature = {}
        for lit in instance:
            feature_name = self.to_features((lit,))[0].split()[0]
            instance_by_feature.setdefault(feature_name, []).append(lit)

            if feature_name not in features_to_process:
                features_to_process.append(feature_name)

        # 3. Main merge loop
        for feature in features_to_process:

            # A. Priority chains
            if feature in chains_by_feature:
                for lit in chains_by_feature[feature]:
                    if lit not in merged:
                        merged.append(lit)

            # B. Remaining instance literals
            if feature in instance_by_feature:
                for lit in instance_by_feature[feature]:
                    if lit not in merged:
                        merged.append(lit)

        # 4. Safety net
        for lit in instance:
            if lit not in merged:
                merged.append(lit)

        return merged


    def m_cpi_xp2(
        self, *, n=1, strategy="priority_order", random_seed=42, ordre_features=None
    ):
        """
        Extract 'n' explanations (AXp) for a Decision Tree (DT).
        Cleaned version: no prints, only logic.
        """
        # 1. Preparation
        th = tuple(self.get_theory())
        if self._instance is None:
            raise ValueError("Instance is not defined.")

        cnf_target = self._tree.to_CNF(
            self._instance, target_prediction=self.target_prediction
        )

        all_explanations = []
        k = 0
        attempts = 0
        max_retries = n * 5

        while k < n and attempts < max_retries:
            attempts += 1

            current_instance = list(self._binary_representation)
            current_seed = (random_seed + attempts) if random_seed is not None else None

            suppression_order = self.get_suppression_order2(
                current_instance,
                th,
                strategy,
                seed=current_seed,
                ordre_features=ordre_features if strategy == "priority_order" else None,
            )

            # === Greedy minimization ===
            for literal_to_test in suppression_order:
                if literal_to_test not in current_instance:
                    continue

                temp_instance = [
                    lit for lit in current_instance if lit != literal_to_test
                ]
                can_remove = True

                solver = Glucose3()
                for clause in th:
                    solver.add_clause(list(clause))
                for lit in temp_instance:
                    solver.add_clause([lit])

                for clause_target in cnf_target:
                    assumptions = [-lit for lit in clause_target]
                    if solver.solve(assumptions=assumptions):
                        can_remove = False
                        break

                solver.delete()

                if can_remove:
                    current_instance.remove(literal_to_test)

            final_explanation = tuple(sorted(current_instance))
            if final_explanation not in all_explanations:
                all_explanations.append(final_explanation)
                k += 1

        return all_explanations

    
    def get_suppression_order(
        self, instance, th, strategy="priority_order", seed=None, ordre_features=None
    ):
        """
        Return the literal removal order according to the chosen strategy.

        Args:
            instance: List of literals
            th: Theory (clauses)
            strategy: Removal strategy to use
            seed: Seed for random strategy (optional)

        Returns:
            Ordered list of literals according to the strategy
        """
        if strategy == "priority_order":
            chains_by_feature = self.get_feature_chain_lists_with_positive_first(th, instance)
            priority_features = self.merge_chains_and_instance(
                chains_by_feature, instance, ordre_features=ordre_features
            )
            return priority_features

        elif strategy == "beginning_to_end":
            # Remove from beginning to end (original order)
            return list(instance)

        elif strategy == "end_to_beginning":
            # Remove from end to beginning (reverse order)
            return list(reversed(instance))

        elif strategy == "random":
            # Random removal
            random_order = list(instance)
            if seed is not None:
                random.seed(seed)
            random.shuffle(random_order)
            return random_order

        else:
            raise ValueError(f"Unknown strategy: {strategy}")


    def merge_chains_and_instance(self, chains_by_feature, instance, ordre_features=None):
        """
        Return a list of lists, where each sublist contains the literals
        associated with a single feature.
        Example: [[-3, -19, ...], [-12, -18, ...]]
        """
        grouped_features = []

        # 1. Determine the complete list of features to process
        priority_features = list(ordre_features) if ordre_features else []
        all_features = list(chains_by_feature.keys())
        other_features = [f for f in all_features if f not in priority_features]

        features_to_process = priority_features + other_features

        # 2. Preprocessing: group instance literals by feature
        instance_by_feature = {}
        for lit in instance:
            feature_name = self.to_features((lit,))[0].split()[0]
            instance_by_feature.setdefault(feature_name, []).append(lit)

            if feature_name not in features_to_process:
                features_to_process.append(feature_name)

        # 3. Main loop: build feature groups
        processed_literals = set()

        for feature in features_to_process:
            current_group = []

            # A. Priority chains
            if feature in chains_by_feature:
                for lit in chains_by_feature[feature]:
                    if lit not in processed_literals and lit in instance:
                        current_group.append(lit)
                        processed_literals.add(lit)

            # B. Remaining instance literals for this feature
            if feature in instance_by_feature:
                for lit in instance_by_feature[feature]:
                    if lit not in processed_literals:
                        current_group.append(lit)
                        processed_literals.add(lit)

            if current_group:
                grouped_features.append(current_group)

        # 4. Safety net for literals without identified features
        leftovers = []
        for lit in instance:
            if lit not in processed_literals:
                leftovers.append(lit)

        if leftovers:
            grouped_features.append(leftovers)

        return grouped_features


    def cpi_xp(
            self, *, n=1, strategy="priority_order", random_seed=42, ordre_features=None
        ):
            """
            Implementation of Algorithm 1 (CPI-Xp) adapted for Decision Trees (DT).
            Includes the "Symmetric Safe Jump" optimization for numerical features.
            
            TECHNICAL FIX:
            The solver is initialized ONLY ONCE here and passed to the imp_dt function.
            """
            # 1. DT-specific initialization
            theory_clauses = list(self.get_theory())

            # CNF representing the target prediction
            cnf_target = self._tree.to_CNF(
                self._instance, target_prediction=self.target_prediction
            )

            current_instance = list(self._binary_representation)
            features_groups = self.get_suppression_order(
                current_instance,
                theory_clauses,
                strategy="priority_order",
                seed=None,
                ordre_features=ordre_features,
            )

            cx = []
            remaining_features_map = {i: group for i, group in enumerate(features_groups)}

            # === PERSISTENT SOLVER INITIALIZATION ===
            # Created here to avoid recreating it thousands of times inside imp_dt()
            with Glucose3() as solver:
                # Load the theory clauses once and for all
                for clause in theory_clauses:
                    solver.add_clause(list(clause))

                # Main loop over feature groups
                for i, t_prime in enumerate(features_groups):

                    # Feature type detection
                    feature_type = "categorical"
                    if t_prime:
                        first_lit = t_prime[0]
                        # Note: Ensure self.to_features is efficient or cached
                        feature_str = self.to_features((first_lit,))[0]
                        if (
                            ("in ]" in feature_str or "in [" in feature_str)
                            or any(op in feature_str for op in ["<", ">", "<=", ">="])
                        ):
                            feature_type = "numeric"

                    while True:
                        if not t_prime:
                            break

                        found = False
                        gs = self.msg(t_prime, feature_type)

                        for g in gs:
                            hypothesis = list(cx) + list(g)
                            for j in range(i + 1, len(features_groups)):
                                hypothesis.extend(remaining_features_map[j])

                            # Pass the persistent solver
                            if self.imp_dt(hypothesis, solver, cnf_target):
                                t_prime = g
                                found = True
                                break

                        if not found:
                            # Symmetric Safe Jump optimization
                            if feature_type == "numeric":
                                failed_literal = t_prime[0]

                                if failed_literal > 0:
                                    pending_opposite = [x for x in t_prime if x < 0]
                                    if pending_opposite:
                                        remaining_same_sign = [x for x in t_prime if x > 0]
                                        cx.extend(remaining_same_sign)
                                        t_prime = pending_opposite
                                        continue

                                elif failed_literal < 0:
                                    pending_opposite = [x for x in t_prime if x > 0]
                                    if pending_opposite:
                                        remaining_same_sign = [x for x in t_prime if x < 0]
                                        cx.extend(remaining_same_sign)
                                        t_prime = pending_opposite
                                        continue

                            cx.extend(t_prime)
                            break

            return sorted(tuple(cx))


    def msg(self, t_part, feature_type):
        """Implementation of msg (Most Specific Generalization)."""
        gs = []
        if feature_type == "numeric":
            if t_part:
                gs.append(t_part[1:])
        else:
            for k in range(len(t_part)):
                gs.append(t_part[:k] + t_part[k + 1 :])
        return gs


    def imp_dt(self, hypothesis, solver, cnf_target):
        """
        DT-specific implicant test using a PERSISTENT solver.
        
        We check if (Hypothesis AND Theory) => cnf_target.
        Equivalent to: For each clause C in cnf_target, (Hypothesis AND Theory AND NOT C) is UNSAT.
        
        The 'solver' already contains 'Theory'.
        We add 'Hypothesis' and 'NOT C' as assumptions (transient clauses).
        """
        
        # Iterate over each clause of the target CNF
        for clause_target in cnf_target:
            # Negate the clause (De Morgan laws: NOT(A or B) = NOT A and NOT B)
            # This becomes a set of literals to add to assumptions
            negated_target_clause = [-lit for lit in clause_target]
            
            # Assumptions = Hypothesis + Negated Target Clause
            # This tests the hypothesis against this specific part of the target
            assumptions = hypothesis + negated_target_clause
            
            # If solve returns True (SAT), it means the implication is FALSE (we found a counter-example)
            if solver.solve(assumptions=assumptions):
                return False

        # If UNSAT for all negated clauses, the implication holds
        return True


    def m_cpi_xp(
        self, *, n=1, strategy="priority_order", random_seed=42, ordre_features=None
    ):
        """
        Compute a minimal CPI-Xp (mCPI-Xp) for a Decision Tree (DT).
        Optimized version with persistent solver.
        """
        # 1. Compute a valid (potentially non-minimal) CPI-Xp explanation
        cpi_explanation = list(
            self.cpi_xp(
                n=n,
                strategy=strategy,
                random_seed=random_seed,
                ordre_features=ordre_features,
            )
        )

        theory_clauses = list(self.get_theory())
        cnf_target = self._tree.to_CNF(
            self._instance, target_prediction=self.target_prediction
        )

        # 2. Minimization loop with a PERSISTENT SOLVER
        with Glucose3() as solver:
            # Load theory once
            for clause in theory_clauses:
                solver.add_clause(list(clause))
                
            for literal in list(cpi_explanation):
                # Try to remove the current literal
                candidate_explanation = [
                    l for l in cpi_explanation if l != literal
                ]

                # Reuse the persistent solver for the check
                if self.imp_dt(candidate_explanation, solver, cnf_target):
                    cpi_explanation.remove(literal)

        return sorted(tuple(cpi_explanation))

    def _minimize_prime_implicant_v2(self, candidate_model, theory, cnf):
        """
        Version améliorée de la minimisation
        """
        print(f"\n--- MINIMISATION V2 ---")
        current_model = list(candidate_model)
        
        # Trier par ordre de priorité si la méthode existe
        if hasattr(self, 'get_priority_order'):
            try:
                suppression_order = self.get_priority_order(current_model, theory)
            except:
                suppression_order = sorted(current_model, reverse=True)  # Fallback
        else:
            suppression_order = sorted(current_model, reverse=True)
        
        print(f"Ordre de suppression: {suppression_order}")
        
        for lit_to_remove in suppression_order:
            if lit_to_remove not in current_model:
                continue
            
            print(f"\nTest suppression de {lit_to_remove}")
            temp_model = [lit for lit in current_model if lit != lit_to_remove]
            
            # Test: est-ce que (theory ∧ temp_model) → cnf ?
            if self._implies_cnf(temp_model, theory, cnf):
                current_model.remove(lit_to_remove)
                print(f"✓ {lit_to_remove} supprimé")
            else:
                print(f"✗ {lit_to_remove} nécessaire")
        
        result = tuple(current_model)
        print(f"Prime implicant final: {result}")
        return result

    def _implies_cnf(self, model, theory, cnf):
        """
        Test si (theory ∧ model) → cnf
        """
        for clause in cnf:
            if not clause:
                continue
            
            try:
                from pysat.solvers import Glucose3
                glucose = Glucose3()
                
                # theory ∧ model ∧ ¬clause
                for th_clause in theory:
                    if th_clause:
                        glucose.add_clause(list(th_clause))
                
                for lit in model:
                    glucose.add_clause([lit])
                
                for lit in clause:
                    glucose.add_clause([-lit])
                
                result = glucose.solve()
                glucose.delete()
                
                if result:
                    return False  # Il existe un contre-exemple
                    
            except Exception as e:
                print(f"Erreur test implication: {e}")
                return False
        
        return True  # Toutes les clauses sont impliquées
    def sufficient_reason(self, *, n=1, time_limit=None):
        if self._instance is None:
            raise ValueError("Instance is not set")
        time_used = 0
        n = n if type(n) == int else float('inf')
        cnf = self._tree.to_CNF(self._instance, target_prediction=self.target_prediction)
        prime_implicant_cnf = CNFencoding.to_prime_implicant_CNF(cnf, self._binary_representation)

        if self._excluded_features_are_necesssary(prime_implicant_cnf):
            self._elapsed_time = 0
            return []

        SATsolver = GlucoseSolver()
        SATsolver.add_clauses(prime_implicant_cnf.cnf)

        # Remove excluded features
        SATsolver.add_clauses([[-prime_implicant_cnf.from_original_to_new(lit)]
                               for lit in self._excluded_literals
                               if prime_implicant_cnf.from_original_to_new(lit) is not None])

        sufficient_reasons = []
        while True:
            if (time_limit is not None and time_used > time_limit) or len(sufficient_reasons) == n:
                break
            result, _time = SATsolver.solve(None if time_limit is None else time_limit - time_used)
            time_used += _time
            if result is None:
                break
            sufficient_reasons.append(prime_implicant_cnf.get_reason_from_model(result))
            SATsolver.add_clauses([prime_implicant_cnf.get_blocking_clause(result)])
        self._elapsed_time = time_used if (time_limit is None or time_used < time_limit) else Explainer.TIMEOUT

        reasons = Explainer.format(sufficient_reasons, n)
        self._visualisation.add_history(self._instance, self.__class__.__name__, self.sufficient_reason.__name__,
                                        reasons)
        return reasons

    def sufficient_theory_reason(self, *, n_iterations=50, time_limit=None, seed=0):
        if self._instance is None:
            raise ValueError("Instance is not set")
        print(self.get_theory())
        if seed is None: seed = -1
        if self.c_RF is None:
            # Preprocessing to give all trees in the c++ library
            self.c_RF = c_explainer.new_classifier_RF(len(self._tree.target_class))

            try:
                c_explainer.add_tree(self.c_RF, self._tree.raw_data_for_CPP())
            except Exception as e:
                print("Erreur", str(e))
                exit(1)

        if time_limit is None:
            time_limit = 0
        implicant_id_features = ()  # FEATURES : TODO
        c_explainer.set_excluded(self.c_RF, tuple(self._excluded_literals))
        if self._theory:
            c_explainer.set_theory(self.c_RF, tuple(self.get_theory()))
        current_time = time.process_time()
        reason = c_explainer.compute_reason(self.c_RF, self._binary_representation, implicant_id_features,
                                            self.target_prediction, n_iterations,
                                            time_limit, int(ReasonExpressivity.Conditions), seed, 0)
        total_time = time.process_time() - current_time
        self._elapsed_time = total_time if time_limit == 0 or total_time < time_limit else Explainer.TIMEOUT

        reason = Explainer.format(reason)

        return reason

    def is_reason(self, reason, *, n_samples=-1):
        extended = self.extend_reason_with_theory(reason)
        return self._tree.is_implicant(extended, self.target_prediction)

    def get_theory(self):
        return self.tree.get_theory(self._binary_representation) + self._additional_theory

    def preferred_sufficient_reason(self, *, method, n=1, time_limit=None, weights=None, features_partition=None):
        if self._instance is None:
            raise ValueError("Instance is not set")
        n = n if type(n) == int else float('inf')
        cnf = self._tree.to_CNF(self._instance)
        self._elapsed_time = 0

        prime_implicant_cnf = CNFencoding.to_prime_implicant_CNF(cnf, self._binary_representation)

        # excluded are necessary => no reason
        if self._excluded_features_are_necesssary(prime_implicant_cnf):
            return None

        cnf = prime_implicant_cnf.cnf
        if len(cnf) == 0:
            reasons = Explainer.format([[lit for lit in prime_implicant_cnf.necessary]], n=n)
            if method == PreferredReasonMethod.Minimal:
                self._visualisation.add_history(self._instance, self.__class__.__name__,
                                                self.minimal_sufficient_reason.__name__, reasons)
            else:
                self._visualisation.add_history(self._instance, self.__class__.__name__,
                                                self.preferred_sufficient_reason.__name__, reasons)
            return reasons

        weights = compute_weight(method, self._instance, weights, self._tree.learner_information,
                                 features_partition=features_partition)
        weights_per_feature = {i + 1: weight for i, weight in enumerate(weights)}

        soft = [lit for lit in prime_implicant_cnf.mapping_original_to_new if lit != 0]
        weights_soft = []
        for lit in soft:  # soft clause
            for i in range(len(self._instance)):
                # if self.to_features([lit], eliminate_redundant_features=False, details=True)[0]["id"] == i + 1:

                if self._tree.get_id_features([lit])[0] == i + 1:
                    weights_soft.append(weights[i])

        solver = OPENWBOSolver()

        # Hard clauses
        solver.add_hard_clauses(cnf)

        # Soft clauses
        for i in range(len(soft)):
            solver.add_soft_clause([-soft[i]], weights_soft[i])

        # Remove excluded features
        for lit in self._excluded_literals:
            if prime_implicant_cnf.from_original_to_new(lit) is not None:
                solver.add_hard_clause([-prime_implicant_cnf.from_original_to_new(lit)])

        # Solving
        time_used = 0
        best_score = -1
        reasons = []
        first_call = True

        while True:
            status, model, _time = solver.solve(time_limit=0 if time_limit is None else time_limit - time_used)
            time_used += _time
            if model is None:
                break

            preferred = prime_implicant_cnf.get_reason_from_model(model)
            solver.add_hard_clause(prime_implicant_cnf.get_blocking_clause(model))
            # Compute the score
            # score = sum([weights_per_feature[feature["id"]] for feature in
            #             self.to_features(preferred, eliminate_redundant_features=False, details=True)])

            score = sum([weights_per_feature[id_feature] for id_feature in self._tree.get_id_features(preferred)])
            if first_call:
                best_score = score
            elif score != best_score:
                break
            first_call = False
            reasons.append(preferred)
            if (time_limit is not None and time_used > time_limit) or len(reasons) == n:
                break
        self._elapsed_time = time_used if time_limit is None or time_used < time_limit else Explainer.TIMEOUT
        reasons = Explainer.format(reasons, n)
        if method == PreferredReasonMethod.Minimal:
            self._visualisation.add_history(self._instance, self.__class__.__name__,
                                            self.minimal_sufficient_reason.__name__, reasons)
        else:
            self._visualisation.add_history(self._instance, self.__class__.__name__,
                                            self.preferred_sufficient_reason.__name__, reasons)
        return reasons

    def minimal_sufficient_reason(self, *, n=1, time_limit=None):
        return self.preferred_sufficient_reason(method=PreferredReasonMethod.Minimal, n=n, time_limit=time_limit)

    def n_sufficient_reasons(self, time_limit=None):
        self.n_sufficient_reasons_per_attribute(time_limit=time_limit)
        return self._n_sufficient_reasons

    def n_sufficient_reasons_per_attribute(self, *, time_limit=None):
        if self._instance is None:
            raise ValueError("Instance is not set")
        cnf = self._tree.to_CNF(self._instance)
        prime_implicant_cnf = CNFencoding.to_prime_implicant_CNF(cnf, self._binary_representation)

        if self._excluded_features_are_necesssary(prime_implicant_cnf):
            self._elapsed_time = 0
            self._n_sufficient_reasons = 0
            return None

        if len(prime_implicant_cnf.cnf) == 0:  # Special case where all in necessary
            return {lit: 1 for lit in prime_implicant_cnf.necessary}

        compiler = D4Solver()
        # Remove excluded features
        cnf = list(prime_implicant_cnf.cnf)
        for lit in self._excluded_literals:
            if prime_implicant_cnf.from_original_to_new(lit) is not None:
                cnf.append([-prime_implicant_cnf.from_original_to_new(lit)])

        compiler.add_cnf(cnf, prime_implicant_cnf.n_literals - 1)
        compiler.add_count_model_query(cnf, prime_implicant_cnf.n_literals - 1, prime_implicant_cnf.n_literals_mapping)

        time_used = -time.time()
        n_models = compiler.solve(time_limit)
        self._n_sufficient_reasons = n_models[0]
        time_used += time.time()

        self._elapsed_time = Explainer.TIMEOUT if n_models[1] == -1 else time_used
        if self._elapsed_time == Explainer.TIMEOUT:
            self._n_sufficient_reasons = None
            return {}

        n_necessary = n_models[0] if len(n_models) > 0 else 1

        n_sufficients_per_attribute = {n: n_necessary for n in prime_implicant_cnf.necessary}
        for lit in range(1, prime_implicant_cnf.n_literals_mapping):
            n_sufficients_per_attribute[prime_implicant_cnf.mapping_new_to_original[lit]] = n_models[lit]

        return n_sufficients_per_attribute

    def condi(self, *, conditions):
        conditions, change = self._tree.parse_conditions_for_rectify(conditions)
        return conditions

    def rectify_cxx(self, *, conditions, label, tests=False, theory_cnf=None):
        """
        C++ version
        Rectify the Decision Tree (self._tree) of the explainer according to a `conditions` and a `label`.
        Simplify the model (the theory can help to eliminate some nodes).
        """

        # check conditions and return a list of literals

        conditions, change = self._tree.parse_conditions_for_rectify(conditions)
        if change is True and self._last_features_types is not None:
            self.set_features_type(self._last_features_types)

        current_time = time.process_time()
        if self.c_rectifier is None:
            self.c_rectifier = c_explainer.new_rectifier()

        if tests is True:
            is_implicant = self.is_implicant(conditions, prediction=label)
            print("is_implicant ?", is_implicant)

        c_explainer.rectifier_add_tree(self.c_rectifier, self._tree.raw_data_for_CPP())
        n_nodes_cxx = c_explainer.rectifier_n_nodes(self.c_rectifier)
        Tools.verbose("Rectify - Number of nodes - Initial (c++):", n_nodes_cxx)

        # Rectification part
        c_explainer.rectifier_improved_rectification(self.c_rectifier, conditions, label)
        n_nodes_ccx = c_explainer.rectifier_n_nodes(self.c_rectifier)
        Tools.verbose("Rectify - Number of nodes - After rectification (c++):", n_nodes_ccx)
        if tests is True:

            # for i in range(len(self._random_forest.forest)):
            tree_tuples = c_explainer.rectifier_get_tree(self.c_rectifier, 0)
            self._tree.delete(self._tree.root)
            self._tree.root = self._tree.from_tuples(tree_tuples)
            is_implicant = self.is_implicant(conditions, prediction=label)
            print("is_implicant after rectification ?", is_implicant)
            if is_implicant is False:
                raise ValueError("Problem 2")

        # Simplify Theory part
        if theory_cnf is None:
            theory_cnf = self.get_model().get_theory(None)
        else:
            print("my theorie")
        c_explainer.rectifier_set_theory(self.c_rectifier, tuple(theory_cnf))
        c_explainer.rectifier_simplify_theory(self.c_rectifier)

        n_nodes_cxx = c_explainer.rectifier_n_nodes(self.c_rectifier)
        Tools.verbose("Rectify - Number of nodes - After simplification with the theory (c++):", n_nodes_cxx)

        if tests is True:
            tree_tuples = c_explainer.rectifier_get_tree(self.c_rectifier, 0)
            self._tree.delete(self._tree.root)
            self._tree.root = self._tree.from_tuples(tree_tuples)
            is_implicant = self.is_implicant(conditions, prediction=label)
            print("is_implicant after simplify theory ?", is_implicant)
            if is_implicant is False:
                raise ValueError("Problem 3")

        # Simplify part
        c_explainer.rectifier_simplify_redundant(self.c_rectifier)
        n_nodes_cxx = c_explainer.rectifier_n_nodes(self.c_rectifier)
        Tools.verbose("Rectify - Number of nodes - After elimination of redundant nodes (c++):", n_nodes_cxx)

        # Get the C++ trees and convert it :)
        tree_tuples = c_explainer.rectifier_get_tree(self.c_rectifier, 0)
        self._tree.delete(self._tree.root)
        self._tree.root = self._tree.from_tuples(tree_tuples)

        c_explainer.rectifier_free(self.c_rectifier)
        Tools.verbose("Rectify - Number of nodes - Final (c++):", self._tree.n_nodes())
        if tests is True:
            is_implicant = self.is_implicant(conditions, prediction=label)
            print("is_implicant after simplify ?", is_implicant)
            if is_implicant is False:
                raise ValueError("Problem 4")

        if self._instance is not None:
            self.set_instance(self._instance)

        self._elapsed_time = time.process_time() - current_time

        Tools.verbose("Rectification time:", self._elapsed_time)

        Tools.verbose("--------------")
        return self._tree

    def rectify(self, *, conditions, label, cxx=True, tests=False, theory_cnf=None):
        """
        Rectify the Decision Tree (self._tree) of the explainer according to a `conditions` and a `label`.
        Simplify the model (the theory can help to eliminate some nodes).

        Args:
            decision_rule (list or tuple): A decision rule in the form of list of literals (binary variables representing the conditions of the tree).
            label (int): The label of the decision rule.
        Returns:
            DecisionTree: The rectified tree.
        """
        if cxx is True:
            return self.rectify_cxx(conditions=conditions, label=label, tests=tests, theory_cnf=theory_cnf)

        Tools.verbose("")
        Tools.verbose("-------------- Rectification information:")

        is_implicant = self._tree.is_implicant(conditions, label)
        print("is_implicant before rectification ?", is_implicant)

        tree_decision_rule = self._tree.decision_rule_to_tree(conditions, label)
        Tools.verbose("Classification Rule - Number of nodes:", tree_decision_rule.n_nodes())
        Tools.verbose("Model - Number of nodes:", self._tree.n_nodes())
        if label == 1:
            # When label is 1, we have to inverse the decision rule and disjoint the two trees.
            tree_decision_rule = tree_decision_rule.negating_tree()
            tree_rectified = self._tree.disjoint_tree(tree_decision_rule)
        elif label == 0:
            # When label is 0, we have to concatenate the two trees.
            tree_rectified = self._tree.concatenate_tree(tree_decision_rule)
        else:
            raise NotImplementedError("Multiclasses is in progress.")

        print("tree_rectified:", tree_rectified.raw_data_for_CPP())
        print("label:", label)

        is_implicant = tree_rectified.is_implicant(conditions, label)
        print("is_implicant after rectification ?", is_implicant)
        if is_implicant is False:
            raise ValueError("Problem 2")

        Tools.verbose("Model - Number of nodes (after rectification):", tree_rectified.n_nodes())
        tree_rectified = self.simplify_theory(tree_rectified)

        is_implicant = tree_rectified.is_implicant(conditions, label)
        print("is_implicant after rectification ?", is_implicant)
        if is_implicant is False:
            raise ValueError("Problem 3")

        Tools.verbose("Model - Number of nodes (after simplification using the theory):", tree_rectified.n_nodes())
        tree_rectified.simplify()
        Tools.verbose("Model - Number of nodes (after elimination of redundant nodes):", tree_rectified.n_nodes())

        self._tree = tree_rectified
        if self._instance is not None:
            self.set_instance(self._instance)
        Tools.verbose("--------------")
        return self._tree

    @staticmethod
    def _rectify_tree(_tree, positive_rectifying__tree, negative_rectifying__tree):
        not_positive_rectifying__tree = positive_rectifying__tree.negating_tree()
        not_negative_rectifying__tree = negative_rectifying__tree.negating_tree()

        _tree_1 = positive_rectifying__tree.concatenate_tree(not_negative_rectifying__tree)
        _tree_2 = negative_rectifying__tree.concatenate_tree(not_positive_rectifying__tree)

        not__tree_2 = _tree_2.negating_tree()

        _tree_and_not__tree_2 = _tree.concatenate_tree(not__tree_2)
        _tree_and_not__tree_2.simplify()

        _tree_and_not__tree_2_or__tree_1 = _tree_and_not__tree_2.disjoint_tree(_tree_1)

        _tree_and_not__tree_2_or__tree_1.simplify()

        return _tree_and_not__tree_2_or__tree_1

    def anchored_reason(self, *, n_anchors=2, reference_instances, time_limit=None, check=False):
        cnf = self._tree.to_CNF(self._instance, target_prediction=self.target_prediction, inverse_coding=True)
        n_variables = CNFencoding.compute_n_variables(cnf)
        return self._anchored_reason(n_variables=n_variables, cnf=cnf, n_anchors=n_anchors,
                                     reference_instances=reference_instances, time_limit=time_limit, check=check)

    def to_CNF(self):
        return self._tree.to_CNF(self._instance)
