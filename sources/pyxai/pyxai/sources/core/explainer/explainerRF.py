import random
import time

import c_explainer
import numpy

from pyxai.sources.core.explainer.Explainer import Explainer
from pyxai.sources.core.structure.type import Encoding, PreferredReasonMethod, TypeTheory, ReasonExpressivity
from pyxai.sources.core.tools.encoding import CNFencoding
from pyxai.sources.core.tools.utils import compute_weight
from pyxai.sources.solvers.MAXSAT.OPENWBOSolver import OPENWBOSolver
from pyxai.sources.solvers.MUS.MUSERSolver import MUSERSolver
from pyxai.sources.solvers.MUS.OPTUXSolver import OPTUXSolver
from pyxai.sources.solvers.SAT.glucoseSolver import GlucoseSolver
from pyxai import Tools
from pysat.solvers import Glucose3


class ExplainerRF(Explainer):

    def __init__(self, random_forest, instance=None):
        """Create object dedicated to finding explanations from a random forest ``random_forest`` and an instance ``instance``.

        Args:
            random_forest (RandomForest): The model in the form of a RandomForest object.
            instance (:obj:`list` of :obj:`int`, optional): The instance (an observation) on which explanations must be calculated. Defaults to None.

        Raises:
            NotImplementedError: Currently, the explanations from a random forest are not available in the multi-class scenario (work in progress).
        """
        super().__init__()

        self._random_forest = random_forest
        self.c_RF = None
        self.c_rectifier = None
        self._additional_theory = []
        if instance is not None:
            self.set_instance(instance)

    @property
    def random_forest(self):
        return self._random_forest

    def get_theory(self):
        return self._random_forest.get_theory(self._binary_representation) + self._additional_theory

    def add_clause_to_theory(self, clause):
        self._additional_theory.append(tuple(clause))
        self._theory = True
        self.c_RF = None
        self._glucose = None

    def to_features(self, binary_representation, *, eliminate_redundant_features=True, details=False, contrastive=False,
                    without_intervals=False):
        """
        Convert each literal of the implicant (representing a condition) to a tuple (``id_feature``, ``threshold``, ``sign``, ``weight``).
          - ``id_feature``: the feature identifier.
          - ``threshold``: threshold in the condition id_feature < threshold ?
          - ``sign``: indicate if the condition (id_feature < threshold) is True or False in the reason.
          - ``weight``: possible weight useful for some methods (can be None).

        Remark: Eliminate the redundant features (example: feature5 < 0.5 and feature5 < 0.4 => feature5 < 0.4).

        Args:
            binary_representation (obj:`list` of :obj:`int`): The reason or an implicant.

        Returns:
            obj:`tuple` of :obj:`tuple` of size 4: Represent the reason in the form of features (with their respective thresholds, signs and possible
            weights)
        """
        return self._random_forest.to_features(binary_representation,
                                               eliminate_redundant_features=eliminate_redundant_features,
                                               details=details,
                                               contrastive=contrastive, without_intervals=without_intervals,
                                               feature_names=self.get_feature_names())

    def _to_binary_representation(self, instance):
        return self._random_forest.instance_to_binaries(instance)

    def is_implicant(self, binary_representation, *, prediction=None):
        if prediction is None:
            prediction = self.target_prediction
        binary_representation = self.extend_reason_with_theory(binary_representation)
        return self._random_forest.is_implicant(binary_representation, prediction)

    def predict_votes(self, instance):
        return self._random_forest.predict_votes(instance)

    def predict(self, instance):
        return self._random_forest.predict_instance(instance)

    def direct_reason(self):
        """The direct reason of an instance x is the term t of the implicant (binary form of the instance) corresponding to the unique root-to-leaf
        path of the tree that covers x. (see the Trading Complexity for Sparsity
        in Random Forest Explanations paper (Gilles Audemard, Steve Bellart, Louenas Bounia, Frederic Koriche,
        Jean-Marie Lagniez and Pierre Marquis1) for more information)

        Returns:
            (obj:`list` of :obj:`int`): Reason in the form of literals (binary form). The to_features() method allows to obtain the features
            of this reason.
        """
        if self._instance is None:
            raise ValueError("Instance is not set")

        self._elapsed_time = 0
        tmp = [False for _ in range(len(self.binary_representation) + 1)]
        for tree in self._random_forest.forest:
            local_target_prediction = tree.predict_instance(self._instance)
            if local_target_prediction == self.target_prediction or self._random_forest.n_classes > 2:
                local_direct = tree.direct_reason(self._instance)
                for l in local_direct:
                    tmp[abs(l)] = True
        direct_reason = []
        for l in self.binary_representation:
            if tmp[abs(l)]:
                direct_reason.append(l)

        # remove excluded features
        if any(not self._is_specific(lit) for lit in direct_reason):
            reason = None
        else:
            reason = Explainer.format(list(direct_reason))
        self._visualisation.add_history(self._instance, self.__class__.__name__, self.direct_reason.__name__, reason)
        return reason

    def minimal_contrastive_reason(self, *, n=1, time_limit=None):
        """Formally, a contrastive explanation for an instance x given f is a subset t of the characteristics of x that is minimal w.r.t. set
        inclusion among those such that at least one other instance x' that coincides with x except on the characteristics from t is not classified
        by f as x is. See the On the Explanatory Power of Decision Trees paper (Gilles Audemard, Steve Bellart, Louenas Bounia, Frédéric Koriche,
        Jean-Marie Lagniez, and Pierre Marquis) for more details. This method compute one or several minimal (in size) contrastive reasons.

        Args:
            n (int, optional): The desired number of reasons. Defaults to 1.

        Returns:
            (:obj:`tuple` of :obj:`tuple` of `int`): Reasons in the form of literals (binary form). The to_features() method allows to obtain
            the features of reasons. When only one reason is requested through the 'n' parameter, return just one :obj:`tuple` of `int`
            (and not a :obj:`tuple` of :obj:`tuple`).
        """
        if self._instance is None:
            raise ValueError("Instance is not set")
        if self._random_forest.n_classes > 2:
            raise NotImplementedError("Minimal contrastive reason is not implemented for the multi class case")
        n = n if type(n) == int else float('inf')
        first_call = True
        time_limit = 0 if time_limit is None else time_limit
        best_score = 0
        tree_cnf = self._random_forest.to_CNF(self._instance, self._binary_representation,
                                              target_prediction=1 if self.target_prediction == 0 else 0,
                                              tree_encoding=Encoding.SIMPLE)

        # structure to help to do this method faster
        map_in_binary_representation = dict()
        for id in self._binary_representation:
            map_in_binary_representation[id] = True
            map_in_binary_representation[-id] = False

        # print("tree_cnf:", tree_cnf)
        max_id_binary_representation = CNFencoding.compute_max_id_variable(self._binary_representation)
        # print("max_id_binary_representation:", max_id_binary_representation)

        max_id_binary_cnf = CNFencoding.compute_max_id_variable(tree_cnf)
        # print("max_id_variable:", max_id_binary_cnf)

        MAXSATsolver = OPENWBOSolver()
        # print("Length of the binary representation:", len(self._binary_representation))
        # print("Number of hard clauses in the CNF encoding the random forest:", len(tree_cnf))
        MAXSATsolver.add_hard_clauses(tree_cnf)
        if self._theory is False:
            for lit in self._binary_representation:
                MAXSATsolver.add_soft_clause([lit], weight=1)
        else:
            # Hard clauses
            raise NotImplementedError("TODO get theory")
            theory_cnf, theory_new_variables = self._random_forest.get_theory(
                self._binary_representation,
                theory_type=TypeTheory.NEW_VARIABLES,
                id_new_var=max_id_binary_cnf)
            theory_new_variables, map_is_represented_by_new_variables = theory_new_variables
            # print("Number of hard clauses in the theory:", len(theory_cnf))
            MAXSATsolver.add_hard_clauses(theory_cnf)
            count = 0
            for lit in self._binary_representation:
                if map_is_represented_by_new_variables[abs(lit)] is False:
                    MAXSATsolver.add_soft_clause([lit], weight=1)
                    count += 1
            for new_variable in theory_new_variables:
                MAXSATsolver.add_soft_clause([new_variable], weight=1)

        # Remove excluded features
        for lit in self._excluded_literals:
            MAXSATsolver.add_hard_clause([lit])

        time_used = 0
        results = []
        while True:
            status, reason, _time = MAXSATsolver.solve(time_limit=time_limit)
            time_used += _time
            if time_limit != 0 and time_used > time_limit:
                break
            if reason is None:
                break
            # We have to invert the reason :)
            true_reason = [-lit for lit in reason if
                           abs(lit) <= len(self._binary_representation) and map_in_binary_representation[-lit] == True]

            # Add a blocking clause to avoid this reason in the next steps
            MAXSATsolver.add_hard_clause([-lit for lit in reason if abs(lit) <= max_id_binary_representation])

            # Compute the score
            score = len(true_reason)
            # Stop or not due to score :)
            if first_call:
                best_score = score
            elif score != best_score:
                break
            first_call = False

            # Add this contrastive
            results.append(true_reason)

            # Stop or not due to time or n :)
            if (time_limit != 0 and time_used > time_limit) or len(results) == n:
                # print("End by time_limit or 'n' reached.")
                break

        self._elapsed_time = time_used if time_limit == 0 or time_used < time_limit else Explainer.TIMEOUT

        reasons = Explainer.format(results, n)
        self._visualisation.add_history(self._instance, self.__class__.__name__,
                                        self.minimal_contrastive_reason.__name__, reasons)
        return reasons

    def sufficient_reason(self, *, time_limit=None):
        """A sufficient reason (also known as prime implicant explanation) for an instance x given a class described by a Boolean function f is a
        subset t of the characteristics of x that is minimal w.r.t. set inclusion such that any instance x' sharing this set t of characteristics
        is classified by f as x is.

        Returns:
            (obj:`list` of :obj:`int`): Reason in the form of literals (binary form). The to_features() method allows to obtain the features of this
             reason.
        """
        if self._instance is None:
            raise ValueError("Instance is not set")

        if self._random_forest.n_classes == 2:
            hard_clauses = self._random_forest.to_CNF(self._instance, self._binary_representation,
                                                      self.target_prediction, tree_encoding=Encoding.MUS)
        else:
            hard_clauses = self._random_forest.to_CNF_sufficient_reason_multi_classes(self._instance,
                                                                                      self.binary_representation,
                                                                                      self.target_prediction)

        if self._theory:
            hard_clauses = hard_clauses + tuple(self.get_theory())

        # Check if excluded features produce a SAT problem => No sufficient reason
        if len(self._excluded_literals) > 0:
            SATSolver = GlucoseSolver()
            SATSolver.add_clauses(hard_clauses)
            SATSolver.add_clauses([[lit] for lit in self._binary_representation if self._is_specific(lit)])
            result, time = SATSolver.solve(time_limit=time_limit)
            if result is not None:
                return None

        hard_clauses = list(hard_clauses)
        soft_clauses = tuple((lit,) for lit in self._binary_representation if self._is_specific(lit))
        mapping = [0 for _ in range(len(self._binary_representation) + 2)]
        i = 2  # first good group for literals is 2 (1 is for the CNF representing the RF.
        for lit in self._binary_representation:
            if self._is_specific(lit):
                mapping[i] = lit
                i += 1
        n_variables = CNFencoding.compute_n_variables(hard_clauses)
        muser_solver = MUSERSolver()
        muser_solver.write_gcnf(n_variables, hard_clauses, soft_clauses)
        model, status, self._elapsed_time = muser_solver.solve(time_limit)
        reason = [mapping[i] for i in model if i > 1]

        reason = Explainer.format(reason, 1)
        self._visualisation.add_history(self._instance, self.__class__.__name__, self.sufficient_reason.__name__,
                                        reason)
        return reason

    def minimal_sufficient_reason(self, time_limit=None):
        """A sufficient reason (also known as prime implicant explanation) for an instance x given a class described by a Boolean function f is a
         subset t of the characteristics of x that is minimal w.r.t. set inclusion such that any instance x' sharing this set t of characteristics
         is classified by f as x is.

        This method compute minimal sufficient reason. A minimal sufficient reason for x given f is a sufficient reason for x given f that contains
        a minimal number of literals.

        Returns:
            (obj:`list` of :obj:`int`): Reason in the form of literals (binary form). The to_features() method allows to obtain the features of
            this reason.
        """

        if self._instance is None:
            raise ValueError("Instance is not set")

        if self._random_forest.n_classes == 2:
            hard_clauses = self._random_forest.to_CNF(self._instance, self._binary_representation,
                                                      self.target_prediction, tree_encoding=Encoding.MUS)
        else:
            hard_clauses = self._random_forest.to_CNF_sufficient_reason_multi_classes(self._instance,
                                                                                      self.binary_representation,
                                                                                      self.target_prediction)

        if self._theory:
            clauses_theory = self.get_theory()
            hard_clauses = hard_clauses + tuple(clauses_theory)

        if len(self._excluded_literals) > 0:
            SATSolver = GlucoseSolver()
            SATSolver.add_clauses(hard_clauses)
            SATSolver.add_clauses([[lit] for lit in self._binary_representation if self._is_specific(lit)])
            result = SATSolver.solve(time_limit=time_limit)
            if result is not None:
                return None

        soft_clauses = tuple((lit,) for lit in self._binary_representation if self._is_specific(lit))
        optux_solver = OPTUXSolver()
        optux_solver.add_hard_clauses(hard_clauses)
        optux_solver.add_soft_clauses(soft_clauses, weight=1)
        reason = optux_solver.solve(self._binary_representation)

        reason = Explainer.format(reason)
        self._visualisation.add_history(self._instance, self.__class__.__name__, self.sufficient_reason.__name__,
                                        reason)
        return reason

    def majoritary_reason(self, *, n=1, n_iterations=50, time_limit=None, seed=0):
        """Informally, a majoritary reason for classifying a instance x as positive by some random forest f
        is a prime implicant t of a majority of decision trees in f that covers x. (see the Trading Complexity for Sparsity
        in Random Forest Explanations paper (Gilles Audemard, Steve Bellart, Louenas Bounia, Frederic Koriche,
        Jean-Marie Lagniez and Pierre Marquis1) for more information)

        Args:
            n (int|ALL, optional): The desired number of reasons. Defaults to 1, currently needs to be 1 or Exmplainer.ALL.
            time_limit (int, optional): The maximum time to compute the reasons. None to have a infinite time. Defaults to None.
        """
        if self._instance is None:
            raise ValueError("Instance is not set")

        reason_expressivity = ReasonExpressivity.Conditions
        if seed is None: seed = -1
        if isinstance(n, int) and n == 1:
            if self.c_RF is None:
                # Preprocessing to give all trees in the c++ library
                self.c_RF = c_explainer.new_classifier_RF(self._random_forest.n_classes)
                for tree in self._random_forest.forest:
                    try:
                        c_explainer.add_tree(self.c_RF, tree.raw_data_for_CPP())
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
                                                time_limit, int(reason_expressivity), seed, 0)
            total_time = time.process_time() - current_time
            self._elapsed_time = total_time if time_limit == 0 or total_time < time_limit else Explainer.TIMEOUT
            if reason_expressivity == ReasonExpressivity.Features:
                reason = self.to_features_indexes(reason)  # TODO

            reason = Explainer.format(reason)
            self._visualisation.add_history(self._instance, self.__class__.__name__, self.majoritary_reason.__name__,
                                            reason)
            return reason

        if self._theory:
            raise NotImplementedError("Theory and all majoritary is not yet implanted")
        n = n if type(n) == int else float('inf')

        clauses = self._random_forest.to_CNF(self._instance, self._binary_representation, self.target_prediction,
                                             tree_encoding=Encoding.SIMPLE)
        max_id_variable = CNFencoding.compute_max_id_variable(self._binary_representation)
        solver = GlucoseSolver()

        solver.add_clauses(
            [[lit for lit in clause if lit in self._binary_representation or abs(lit) > max_id_variable] for clause in
             clauses])

        if len(self._excluded_literals) > 0:
            solver.add_clauses([-lit] for lit in self._excluded_literals)

        majoritaries = []
        time_used = 0
        while True:
            result, _time = solver.solve()
            time_used += _time
            if result is None or (time_limit is not None and time_used > time_limit) or (
                    time_limit is None and len(majoritaries) >= n):
                break

            majoritary = [lit for lit in result if lit in self._binary_representation]
            if majoritary not in majoritaries:
                majoritaries.append(majoritary)
            solver.add_clauses([[-lit for lit in result if abs(lit) <= max_id_variable]])  # block this implicant
        self._elapsed_time = time_used if (time_limit is None or time_used < time_limit) else Explainer.TIMEOUT
        reasons = Explainer.format(CNFencoding.remove_subsumed(majoritaries), n)
        self._visualisation.add_history(self._instance, self.__class__.__name__, self.majoritary_reason.__name__,
                                        reasons)
        return reasons

    def preferred_majoritary_reason(self, *, method, n=1, time_limit=None, weights=None, features_partition=None):
        """This approach consists in exploiting a model, making precise her / his
        preferences about reasons, to derive only preferred reasons. See the On Preferred Abductive Explanations for Decision Trees and Random Forests
         paper (Gilles Audemard, Steve Bellart, Louenas Bounia, Frederic Koriche, Jean-Marie Lagniez and Pierre Marquis) for more details.

        Several methods are available thought the method parameter:
          `PreferredReasonMethod.MINIMAL`: Equivalent to minimal_majoritary_reason().
          `PreferredReasonMethod.WEIGHTS`: Weight given bu the user thought the weights parameters.
          `PreferredReasonMethod.SHAPELY`: Shapely values for the model trained on data. Only available with a model from scikitlearn or a save from
          scikitlearn (not compatible with a generic save). For each random forest F, the opposite of the SHAP score
           [Lundberg and Lee, 2017; Lundberg et al., 2020] of each feature of the instance x at hand given F computed using SHAP
           (shap.readthedocs.io/en/latest/api.html)
          `PreferredReasonMethod.FEATURE_IMPORTANCE`: The opposite of the f-importance of each feature in F as computed by Scikit-Learn
          [Pedregosa et al., 2011]. Only available with a model from scikitlearn or a save from scikitlearn (not compatible with a generic save).
          `PreferredReasonMethod.WORD_FREQUENCY`: The opposite of the Zipf frequency of each feature viewed as a word in the wordfreq library.
          `PreferredReasonMethod.WORD_FREQUENCY_LAYERS`: WORD_FREQUENCY with layers.

        Args:
            n (int|ALL, optional): The desired number of reasons. Defaults to Explainer.ALL.
            time_limit (int, optional): The maximum time to compute the reasons. None to have a infinite time. Defaults to None.
            method (Explainer.MINIMAL|Explainer.WEIGHTS|, optional): _description_. Defaults to None.
            weights (:obj:`list` of `int`|:obj:`list`, optional): Can be a list representing the weight of each feature or a dict where a key is
            a feature and the value its weight. Defaults to None.

        Returns:
            _type_: _description_
        """

        if self._instance is None:
            raise ValueError("Instance is not set")

        n = n if type(n) == int else float('inf')

        if self._random_forest.n_classes == 2:
            clauses = self._random_forest.to_CNF(self._instance, self._binary_representation, self.target_prediction,
                                                 tree_encoding=Encoding.SIMPLE)
        else:
            clauses = self._random_forest.to_CNF_majoritary_reason_multi_classes(self._instance,
                                                                                 self._binary_representation,
                                                                                 self.target_prediction)

        n_variables = CNFencoding.compute_n_variables(clauses)
        id_features = self._random_forest.get_id_features(self._binary_representation)

        weights = compute_weight(method, self._instance, weights, self._random_forest.forest[0].learner_information,
                                 features_partition=features_partition)
        solver = OPENWBOSolver()
        max_id_variable = CNFencoding.compute_max_id_variable(self._binary_representation)
        map_abs_implicant = [0 for _ in range(0, n_variables + 1)]
        for lit in self._binary_representation:
            map_abs_implicant[abs(lit)] = lit
        # Hard clauses
        for c in clauses:
            solver.add_hard_clause(
                [lit for lit in c if abs(lit) > max_id_variable or map_abs_implicant[abs(lit)] == lit])

        if self._theory:
            clauses_theory = self.get_theory()
            for c in clauses_theory:
                solver.add_hard_clause(c)

        # excluded features
        for lit in self._binary_representation:
            if not self._is_specific(lit):
                solver.add_hard_clause([-lit])

        # Soft clauses
        for i in range(len(self._binary_representation)):
            solver.add_soft_clause([-self._binary_representation[i]], weights[id_features[i] - 1])

        # Solving
        time_used = 0
        best_score = -1
        reasons = []
        first_call = True

        while True:
            status, model, _time = solver.solve(time_limit=None if time_limit is None else time_limit - time_used)
            time_used += _time
            if model is None:
                if first_call:
                    return ()
                reasons = Explainer.format(reasons, n)
                if method == PreferredReasonMethod.Minimal:
                    self._visualisation.add_history(self._instance, self.__class__.__name__,
                                                    self.minimal_majoritary_reason.__name__, reasons)
                else:
                    self._visualisation.add_history(self._instance, self.__class__.__name__,
                                                    self.preferred_majoritary_reason.__name__, reasons)
                return reasons

            prefered_reason = [lit for lit in model if lit in self._binary_representation]
            solver.add_hard_clause([-lit for lit in model if abs(lit) <= max_id_variable])

            # Compute the score
            score = numpy.sum([weights[id_features[abs(lit) - 1] - 1] for lit in prefered_reason])
            if first_call:
                best_score = score
            elif score != best_score:
                reasons = Explainer.format(reasons, n)
                if method == PreferredReasonMethod.Minimal:
                    self._visualisation.add_history(self._instance, self.__class__.__name__,
                                                    self.minimal_majoritary_reason.__name__, reasons)
                else:
                    self._visualisation.add_history(self._instance, self.__class__.__name__,
                                                    self.preferred_majoritary_reason.__name__, reasons)
                return reasons
            first_call = False

            reasons.append(prefered_reason)
            if (time_limit is not None and time_used > time_limit) or len(reasons) == n:
                break
        self._elapsed_time = time_used if time_limit is None or time_used < time_limit else Explainer.TIMEOUT
        reasons = Explainer.format(reasons, n)
        if method == PreferredReasonMethod.Minimal:
            self._visualisation.add_history(self._instance, self.__class__.__name__,
                                            self.minimal_majoritary_reason.__name__, reasons)
        else:
            self._visualisation.add_history(self._instance, self.__class__.__name__,
                                            self.preferred_majoritary_reason.__name__, reasons)
        return reasons

    def minimal_majoritary_reason(self, *, n=1, time_limit=None):
        return self.preferred_majoritary_reason(method=PreferredReasonMethod.Minimal, n=n, time_limit=time_limit)

    def is_majoritary_reason(self, reason, n_samples=50):
        extended_reason = self.extend_reason_with_theory(reason)
        if not self.is_implicant(extended_reason):
            return False
        tmp = list(reason)
        random.shuffle(tmp)
        nb = 0
        for lit in tmp:
            copy_reason = list(reason).copy()
            copy_reason.remove(lit)
            if self.is_implicant(tuple(copy_reason)):
                return False
            nb += 1
            if nb > n_samples:
                break
        return True

    def most_anchored_reason(self, *, time_limit=None, check=False, type_references="normal"):
        if self._random_forest.n_classes == 2:
            cnf = self._random_forest.to_CNF(self._instance, self._binary_representation, self.target_prediction,
                                             tree_encoding=Encoding.MUS)
            if self._theory:
                size = len(cnf)
                clauses_theory = self.get_theory()
                cnf = cnf + tuple(clauses_theory)
                print("Theory enabled: clauses: " + str(size) + " to " + str(len(cnf)))

                # for c in clauses_theory:
                #    cnf.append(c)

            n_variables = CNFencoding.compute_max_id_variable(cnf)
            return self._most_anchored_reason(n_variables=n_variables, cnf=cnf, time_limit=time_limit, check=check,
                                              type_references=type_references)

        raise NotImplementedError("The anchored_reason() method for RF works only with binary-class datasets.")

    def condi(self, *, conditions):
        conditions, change = self._random_forest.parse_conditions_for_rectify(conditions)
        return conditions

    def rectify_cxx(self, *, conditions, label, tests=False, theory_cnf=None):
        """
        C++ version
        Rectify the Decision Tree (self._tree) of the explainer according to a `conditions` and a `label`.
        Simplify the model (the theory can help to eliminate some nodes).

        Args:
            conditions (list or tuple): A decision rule in the form of list of literals (binary variables representing the conditions of the tree). 
            label (int): The label of the decision rule.   
        Returns:
            RandomForest: The rectified random forest.  
        """

        # check conditions and return a list of literals
        # print("conditions:", conditions)

        conditions, change = self._random_forest.parse_conditions_for_rectify(conditions)
        if change is True:
            self.set_features_type(self._last_features_types)

        current_time = time.process_time()
        if self.c_rectifier is None:
            self.c_rectifier = c_explainer.new_rectifier()

        if tests is True:
            is_implicant = self.is_implicant(conditions, prediction=label)
            print("is_implicant ?", is_implicant)

        for i, tree in enumerate(self._random_forest.forest):
            c_explainer.rectifier_add_tree(self.c_rectifier, tree.raw_data_for_CPP())

        n_nodes_cxx = c_explainer.rectifier_n_nodes(self.c_rectifier)
        Tools.verbose("Rectify - Number of nodes - Initial (c++):", n_nodes_cxx)

        # Rectification part
        c_explainer.rectifier_improved_rectification(self.c_rectifier, conditions, label)
        n_nodes_ccx = c_explainer.rectifier_n_nodes(self.c_rectifier)
        Tools.verbose("Rectify - Number of nodes - After rectification (c++):", n_nodes_ccx)
        if tests is True:
            for i in range(len(self._random_forest.forest)):
                tree_tuples = c_explainer.rectifier_get_tree(self.c_rectifier, i)
                self._random_forest.forest[i].delete(self._random_forest.forest[i].root)
                self._random_forest.forest[i].root = self._random_forest.forest[i].from_tuples(tree_tuples)
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
            for i in range(len(self._random_forest.forest)):
                tree_tuples = c_explainer.rectifier_get_tree(self.c_rectifier, i)
                self._random_forest.forest[i].delete(self._random_forest.forest[i].root)
                self._random_forest.forest[i].root = self._random_forest.forest[i].from_tuples(tree_tuples)
            is_implicant = self.is_implicant(conditions, prediction=label)
            print("is_implicant after simplify theory ?", is_implicant)
            if is_implicant is False:
                raise ValueError("Problem 3")

        # Simplify part
        c_explainer.rectifier_simplify_redundant(self.c_rectifier)
        n_nodes_cxx = c_explainer.rectifier_n_nodes(self.c_rectifier)
        Tools.verbose("Rectify - Number of nodes - After elimination of redundant nodes (c++):", n_nodes_cxx)

        # Get the C++ trees and convert it :) 
        for i in range(len(self._random_forest.forest)):
            tree_tuples = c_explainer.rectifier_get_tree(self.c_rectifier, i)
            self._random_forest.forest[i].delete(self._random_forest.forest[i].root)
            self._random_forest.forest[i].root = self._random_forest.forest[i].from_tuples(tree_tuples)

        c_explainer.rectifier_free(self.c_rectifier)
        Tools.verbose("Rectify - Number of nodes - Final (c++):", self._random_forest.n_nodes())
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
        return self._random_forest

    def rectify(self, *, conditions, label, cxx=True, tests=False, theory_cnf=None):
        """
        Rectify the Decision Tree (self._tree) of the explainer according to a `conditions` and a `label`.
        Simplify the model (the theory can help to eliminate some nodes).

        Args:
            conditions (list or tuple): A decision rule in the form of list of literals (binary variables representing the conditions of the tree). 
            label (int): The label of the decision rule.   
        Returns:
            RandomForest: The rectified random forest.  
        """

        if cxx is True:
            return self.rectify_cxx(conditions=conditions, label=label, tests=tests, theory_cnf=theory_cnf)

        current_time = time.process_time()
        # print("conditions:", conditions)
        # print("conditions to features:", self.to_features(conditions, eliminate_redundant_features=False))

        Tools.verbose("")
        Tools.verbose("-------------- C++ Rectification information:")
        n_nodes_python = self._random_forest.n_nodes()
        Tools.verbose("Model - Number of nodes (initial):", n_nodes_python)

        is_implicant = self.is_implicant(conditions, prediction=label)
        print("is_implicant ?", is_implicant)
        Tools.verbose("Label:", label)

        # Rectification part
        for i, tree in enumerate(self._random_forest.forest):

            tree_decision_rule = self._random_forest.forest[i].decision_rule_to_tree(conditions, label)
            if label == 1:
                # When label is 1, we have to inverse the decision rule and disjoint the two trees. 
                tree_decision_rule = tree_decision_rule.negating_tree()
                self._random_forest.forest[i] = tree.disjoint_tree(tree_decision_rule)
            elif label == 0:
                # When label is 0, we have to concatenate the two trees.  
                self._random_forest.forest[i] = tree.concatenate_tree(tree_decision_rule)
            else:
                raise NotImplementedError("Multiclasses is in progress.")

        n_nodes_python = self._random_forest.n_nodes()
        Tools.verbose("Model - Number of nodes (after rectification):", n_nodes_python)

        if tests is True:
            is_implicant = self.is_implicant(conditions, prediction=label)
            print("is_implicant after rectification ?", is_implicant)
            if is_implicant is False:
                raise ValueError("Problem 2")

        # Simplify Theory part
        for i, tree in enumerate(self._random_forest.forest):
            self._random_forest.forest[i] = self.simplify_theory(tree)

        n_nodes_python = self._random_forest.n_nodes()
        Tools.verbose("Model - Number of nodes (after simplification using the theory):", n_nodes_python)

        if tests is True:
            is_implicant = self.is_implicant(conditions, prediction=label)
            print("is_implicant after simplify theory ?", is_implicant)
            if is_implicant is False:
                raise ValueError("Problem 3")

        # Simplify part
        for i, tree in enumerate(self._random_forest.forest):
            tree.simplify()

        n_nodes_python = self._random_forest.n_nodes()
        Tools.verbose("Model - Number of nodes (after elimination of redundant nodes):", n_nodes_python)

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

        return self._random_forest
    def get_suppression_order2(self, instance, th, strategy="priority_order", seed=None,ordre_features=None):
        """
        Retourne l'ordre de suppression selon la stratégie choisie
        
        Args:
            instance: Liste des littéraux
            th: Théorie (clauses)
            strategy: Stratégie de suppression à utiliser
            seed: Graine pour la stratégie aléatoire (optionnel)
        
        Returns:
            Liste ordonnée des littéraux selon la stratégie
        """
        print("ordre_features",ordre_features)
        # if strategy == "priority_order":
        #     return self.get_priority_order(instance, th)
        if strategy == "priority_order":
                chains_by_feature=self.get_feature_chain_lists_with_positive_first(th,instance)
                print("chains_by_feature",chains_by_feature)
                priorityfeatures=self.merge_chains_and_instance2(chains_by_feature,instance,ordre_features=ordre_features)
                print("priorityfeatures",priorityfeatures)
                #exit(0)
#                 print("to_feature",self.to_features((-10,)
# ))
#                 exit(0)
                return priorityfeatures

        elif strategy == "beginning_to_end":
            # Suppression du début à la fin (ordre original)
            return list(instance)
        
        else:
            raise ValueError(f"Stratégie inconnue: {strategy}")
    def merge_chains_and_instance2(self, chains_by_feature, instance, ordre_features=None):
            """
            Fusionne les chaînes prioritaires et les littéraux restants de l'instance.
            Si un ordre est donné, il est respecté en premier, puis les autres features suivent.
            """
            merged = []
            
            # 1. Déterminer la liste COMPLÈTE des features à parcourir
            # On commence par celles imposées par l'utilisateur
            features_prioritaires = list(ordre_features) if ordre_features else []
            
            # On identifie toutes les features disponibles (clés du dictionnaire)
            toutes_les_features = list(chains_by_feature.keys())
            
            # On ajoute celles qui manquent (qui ne sont pas dans l'ordre imposé)
            autres_features = [f for f in toutes_les_features if f not in features_prioritaires]
            
            # Liste finale : D'abord les imposées, puis le reste
            features_a_parcourir = features_prioritaires + autres_features

            # 2. PRÉ-TRAITEMENT : Classer les littéraux de l'instance par Feature
            instance_by_feature = {}
            for lit in instance:
                feature_full_name = self.to_features((lit,))[0]
                feature_name = feature_full_name.split()[0]
                
                if feature_name not in instance_by_feature:
                    instance_by_feature[feature_name] = []
                instance_by_feature[feature_name].append(lit)
                
                # Si une feature est dans l'instance mais n'était pas dans chains_by_feature, on l'ajoute à la fin
                if feature_name not in features_a_parcourir:
                    features_a_parcourir.append(feature_name)

            print(f"Ordre final de parcours : {features_a_parcourir}")

            # 3. BOUCLE PRINCIPALE : Fusion feature par feature
            for feature in features_a_parcourir:
                
                # A. Chaînes prioritaires (complexes/négatifs)
                if feature in chains_by_feature:
                    chain = chains_by_feature[feature]
                    for lit in chain:
                        if lit not in merged:
                            merged.append(lit)
                
                # B. Orphelins de l'instance (positifs/restants)
                if feature in instance_by_feature:
                    orphans = instance_by_feature[feature]
                    for lit in orphans:
                        if lit not in merged:
                            merged.append(lit)

            # 4. FILET DE SÉCURITÉ
            for lit in instance:
                if lit not in merged:
                    merged.append(lit)
                    
            return merged
    def m_cpi_xp2(self, *, n=1, strategy="priority_order", random_seed=42, ordre_features=None):
        """
        Méthode manuelle optimisée pour Random Forest (RF) et Decision Tree (DT).
        Utilise la preuve par l'absurde (UNSAT) pour valider l'explication.
        """
        # 1. Préparation de la structure logique (Théorie + Structure du Modèle)
        # On récupère la théorie (ex: One-Hot encoding des features)
        theory_clauses = list(self.get_theory())
        
        # 2. On génère la CNF qui représente "Le modèle prédit l'INVERSE de la cible"
        # C'est crucial : Encoding.MUS crée les contraintes pour (Prédiction != Cible)
        if self._random_forest.n_classes == 2:
            inverse_model_cnf = self._random_forest.to_CNF(
                self._instance, 
                self._binary_representation,
                self.target_prediction, 
                tree_encoding=Encoding.MUS
            )
        else:
            # Gestion multi-classes
            inverse_model_cnf = self._random_forest.to_CNF_sufficient_reason_multi_classes(
                self._instance,
                self._binary_representation,
                self.target_prediction
            )
        
        # On combine tout : Théorie + (Modèle != Cible)
        # Ce bloc représente "Le monde impossible" qu'on veut contredire
        hard_clauses = theory_clauses + list(inverse_model_cnf)

        # 3. Initialisation des variables de boucle
        if strategy == "priority_order" and not ordre_features:
            ordre_features = None

        all_explanations = []
        k = 0
        attempts = 0
        max_retries = n * 5

        while k < n and attempts < max_retries:
            attempts += 1
            print(f"\n{'='*50}")
            print(f"RECHERCHE EXPLICATION {k+1}/{n} (Tentative {attempts})")
            
            # Instance de départ (Toute l'instance)
            current_instance = list(self._binary_representation)
            
            # Gestion de l'aléatoire pour varier les explications
            current_seed = (random_seed + attempts) if random_seed is not None else None
            
            # Ordre de suppression
            suppression_order = self.get_suppression_order2(
                current_instance, theory_clauses, strategy, 
                seed=current_seed,
                ordre_features=ordre_features
            )

            # === BOUCLE GLOUTONNE (MANUELLE ET OPTIMISÉE) ===
            for literal_to_test in suppression_order:
                if literal_to_test not in current_instance:
                    continue
                
                # On tente d'enlever le littéral
                temp_instance = [lit for lit in current_instance if lit != literal_to_test]
                
                # --- VÉRIFICATION SAT ---
                # Question : Est-ce que (hard_clauses ET temp_instance) est UNSAT ?
                # Si UNSAT -> Impossible de prédire autre chose -> Explication Valide.
                
                solver = Glucose3()
                
                # A. Ajouter le monde "Impossible" (Théorie + Inverse Modèle)
                for clause in hard_clauses:
                    solver.add_clause(list(clause))
                
                # B. Ajouter notre candidat explication (ce qu'on a gardé)
                for lit in temp_instance:
                    solver.add_clause([lit])
                
                # C. Résolution
                is_sat = solver.solve()
                solver.delete()

                if is_sat == False: 
                    # UNSAT ! Victoire !
                    # Cela signifie qu'avec 'temp_instance', le modèle ne PEUT PAS se tromper.
                    # Le littéral testé était donc inutile.
                    current_instance.remove(literal_to_test)
                    print(f"✓ Supprimé : {literal_to_test}")
                else:
                    # SAT. Échec.
                    # Le solveur a trouvé un cas où l'instance est respectée MAIS le modèle prédit l'inverse.
                    # Donc ce littéral était nécessaire pour bloquer ce contre-exemple.
                    # print(f"✗ Nécessaire : {literal_to_test}")
                    pass

            # === FIN DE L'ITÉRATION ===
            final_explanation = tuple(sorted(current_instance))
            
            if final_explanation not in all_explanations:
                all_explanations.append(final_explanation)
                print(f"\n>>> SUCCÈS : Explication trouvée : {final_explanation}")
                print(f">>> Taille : {len(final_explanation)}")
                k += 1
            else:
                print("⚠ Doublon trouvé, nouvel essai...")

        return all_explanations

    def get_suppression_order(self, instance, th, strategy="priority_order", seed=None,ordre_features=None):
        """
        Retourne l'ordre de suppression selon la stratégie choisie
        
        Args:
            instance: Liste des littéraux
            th: Théorie (clauses)
            strategy: Stratégie de suppression à utiliser
            seed: Graine pour la stratégie aléatoire (optionnel)
        
        Returns:
            Liste ordonnée des littéraux selon la stratégie
        """
        print("ordre_features",ordre_features)
        # if strategy == "priority_order":
        #     return self.get_priority_order(instance, th)
        if strategy == "priority_order":
                chains_by_feature=self.get_feature_chain_lists_with_positive_first(th,instance)
                print("chains_by_feature",chains_by_feature)
                priorityfeatures=self.merge_chains_and_instance(chains_by_feature,instance,ordre_features=ordre_features)
                print("priorityfeatures",priorityfeatures)
                #exit(0)
#                 print("to_feature",self.to_features((-10,)
# ))
#                 exit(0)
                return priorityfeatures

        elif strategy == "beginning_to_end":
            # Suppression du début à la fin (ordre original)
            return list(instance)
        
        elif strategy == "end_to_beginning":
            # Suppression de la fin au début (ordre inverse)
            return list(reversed(instance))
        
        elif strategy == "random":
            # Suppression aléatoire
            random_order = list(instance)
            if seed is not None:
                random.seed(seed)
            random.shuffle(random_order)
            return random_order
        
        else:
            raise ValueError(f"Stratégie inconnue: {strategy}")



    def get_feature_chain_lists_with_positive_first(self, th, instance):
        """
        Pour chaque feature, construit d'abord la chaîne standard,
        puis réordonne pour mettre les positifs inversés en tête.
        """
        # 1) Regroupement exact comme avant
        feature_groups = {}
        print("th", th)
        for a, b in th:
            cond = self.to_features((a,))[0].split()[0]
            feature_groups.setdefault(cond, []).append((a, b))
        print("feature_groups", feature_groups)
        
        result = {}
        for feature, clauses in feature_groups.items():
            chain = []
            # Parcourir toutes les clauses
            for a, b in reversed(clauses):
                # Vérifier la condition XOR : exactement UN des deux littéraux dans l'instance
                a_in = a in instance
                b_in = b in instance
                #print("a_in",a)
                #print("b_in",b)
                # XOR: (a dans instance ET b pas dans instance) OU (b dans instance ET a pas dans instance)
                if (a_in and not b_in):
                    # a est dans l'instance, on ajoute a et -b
                    if -b not in chain:
                        chain.append(-b)
                    if a not in chain:
                        chain.append(a)
                    #print("chain",chain)
                elif (b_in and not a_in):
                    # b est dans l'instance, on ajoute b et -a
                    if b not in chain:
                        chain.append(b)
                    if -a not in chain:
                        chain.append(-a)
                    #print("chain",chain)
                # Sinon, on passe au suivant (continue implicite)
            
            # 2) on réapplique l'ordre désiré
            if chain:  # Seulement si la chaîne n'est pas vide
                result[feature] = self.reorder_chain(chain)
                #print("result[feature]", result[feature])
        
        return result
    def reorder_chain(self,chain):
        """
        Donne une nouvelle chaîne où :
        - on prend d'abord les positifs, inversés
        - puis on ajoute les négatifs, dans l'ordre original
        """
        positives = [x for x in chain if x > 0]
        negatives = [x for x in chain if x < 0]
        positives.reverse()
        return positives + negatives
    def merge_chains_and_instance(self, chains_by_feature, instance, ordre_features=None):
        """
        Retourne une liste de listes. Chaque sous-liste contient les littéraux d'une feature.
        Exemple : [[-3, -19...], [-12, -18...]]
        """
        grouped_features = []
        
        # 1. Déterminer la liste COMPLÈTE des features à parcourir
        features_prioritaires = list(ordre_features) if ordre_features else []
        toutes_les_features = list(chains_by_feature.keys())
        autres_features = [f for f in toutes_les_features if f not in features_prioritaires]
        
        features_a_parcourir = features_prioritaires + autres_features

        # 2. PRÉ-TRAITEMENT : Classer les littéraux de l'instance par Feature
        instance_by_feature = {}
        for lit in instance:
            feature_full_name = self.to_features((lit,))[0]
            feature_name = feature_full_name.split()[0]
            
            if feature_name not in instance_by_feature:
                instance_by_feature[feature_name] = []
            instance_by_feature[feature_name].append(lit)
            
            if feature_name not in features_a_parcourir:
                features_a_parcourir.append(feature_name)

        print(f"Ordre final de parcours : {features_a_parcourir}")

        # 3. BOUCLE PRINCIPALE : Construction des groupes
        processed_literals = set()

        for feature in features_a_parcourir:
            current_group = []
            
            # A. Chaînes prioritaires
            if feature in chains_by_feature:
                chain = chains_by_feature[feature]
                for lit in chain:
                    if lit not in processed_literals and lit in instance: # Vérifier qu'il est dans l'instance
                        current_group.append(lit)
                        processed_literals.add(lit)
            
            # B. Orphelins de l'instance pour cette feature
            if feature in instance_by_feature:
                orphans = instance_by_feature[feature]
                for lit in orphans:
                    if lit not in processed_literals:
                        current_group.append(lit)
                        processed_literals.add(lit)
            
            # On ajoute le groupe s'il n'est pas vide
            if current_group:
                grouped_features.append(current_group)

        # 4. FILET DE SÉCURITÉ (Pour les littéraux qui n'auraient pas de feature identifiée)
        leftovers = []
        for lit in instance:
            if lit not in processed_literals:
                leftovers.append(lit)
        
        if leftovers:
            grouped_features.append(leftovers)
                
        return grouped_features
    def to_CNF(self):
        return self._random_forest.to_CNF(self._instance, self._binary_representation,
                                                      self.target_prediction, tree_encoding=Encoding.SIMPLE)
    def cpi_xp(self, *, n=1, strategy="priority_order", random_seed=42, ordre_features=None):
        """
        [cite_start]Implémentation fidèle de l'Algorithme 1 (CPI-Xp)[cite: 173].
        Intègre une optimisation "Symmetric Safe Jump" pour les numériques :
        Permet de sauter des Positifs aux Négatifs ET inversement en cas de blocage.
        """
        # --- Initialisation (Correspond aux lignes 1-5 de l'Algo 1) ---
        theory_clauses = list(self.get_theory())
        
        if self._random_forest.n_classes == 2:
            inverse_model_cnf = self._random_forest.to_CNF(
                self._instance, self._binary_representation, self.target_prediction, tree_encoding=Encoding.MUS
            )
        else:
            inverse_model_cnf = self._random_forest.to_CNF_sufficient_reason_multi_classes(
                self._instance, self._binary_representation, self.target_prediction
            )
        
        hard_clauses = theory_clauses + list(inverse_model_cnf)

        # Ligne 6: t <- tx
        current_instance = list(self._binary_representation)
        features_groups = self.get_suppression_order(
            current_instance, theory_clauses, strategy="priority_order", 
            seed=None, ordre_features=ordre_features
        )
        
        # Ligne 7: cx <- ensemble vide
        cx = [] 
        
        remaining_features_map = {i: group for i, group in enumerate(features_groups)}

        # === Ligne 8: for each Ai in A do ===
        for i, t_prime in enumerate(features_groups): 
            
            # --- Détection du Type ---
            feature_type = "categorical"
            if t_prime:
                first_lit = t_prime[0]
                feature_str = self.to_features((first_lit,))[0]
                if ("in ]" in feature_str or "in [" in feature_str) or \
                   any(op in feature_str for op in ['<', '>', '<=', '>=']):
                    feature_type = "numeric"

            # === Ligne 9: while true do ===
            while True:
                # Ligne 10-11: if t' is empty then exit-while
                if not t_prime:
                    break

                found = False # Ligne 13
                
                # Ligne 14: gs <- msg(t', Sigma)
                gs = self.msg(t_prime, feature_type)

                # Ligne 16: for each g in gs do
                for g in gs:
                    hypothesis = list(cx) + list(g)
                    for j in range(i + 1, len(features_groups)):
                        hypothesis.extend(remaining_features_map[j])

                    # Ligne 18: if imp(...) then
                    if self.imp(hypothesis, hard_clauses):
                        t_prime = g     # Ligne 19
                        found = True    # Ligne 20
                        break           # Ligne 21
                
                # Ligne 23: if not found then
                if not found:
                    
                    # --- OPTIMISATION : SYMMETRIC SAFE JUMP ---
                    if feature_type == "numeric":
                        failed_literal = t_prime[0]
                        
                        # CAS 1 : Bloqué sur un POSITIF -> Chercher des Négatifs
                        if failed_literal > 0:
                            # Y a-t-il des négatifs dans ce qui reste ?
                            pending_opposite = [x for x in t_prime if x < 0]
                            
                            if pending_opposite:
                                # On sauvegarde TOUS les positifs restants (Safe Keep)
                                remaining_same_sign = [x for x in t_prime if x > 0]
                                cx.extend(remaining_same_sign)
                                
                                # On bascule sur les négatifs
                                t_prime = pending_opposite
                                # print(f"-> Jump: Positifs gardés, passage aux Négatifs.")
                                continue 

                        # CAS 2 : Bloqué sur un NÉGATIF -> Chercher des Positifs
                        elif failed_literal < 0:
                            # Y a-t-il des positifs dans ce qui reste ?
                            # (Cela arrive si votre tri initial met les négatifs avant les positifs)
                            pending_opposite = [x for x in t_prime if x > 0]
                            
                            if pending_opposite:
                                # On sauvegarde TOUS les négatifs restants (Safe Keep)
                                remaining_same_sign = [x for x in t_prime if x < 0]
                                cx.extend(remaining_same_sign)
                                
                                # On bascule sur les positifs
                                t_prime = pending_opposite
                                # print(f"-> Jump: Négatifs gardés, passage aux Positifs.")
                                continue

                    # Comportement standard (Fin de feature ou Catégorique)
                    cx.extend(t_prime) # Ligne 24
                    break              # Ligne 25

        # Ligne 31: return cx
        return sorted(tuple(cx))

    def msg(self, t_part, feature_type):
        """
        Implémentation de msg (Most Specific Generalisation).
        """
        gs = []
        
        # Pour le numérique, on suppose toujours qu'on veut enlever l'élément 
        # le plus "externe" (le premier de la liste triée).
        if feature_type == "numeric":
            if t_part:
                gs.append(t_part[1:])
        else:
            # Pour le catégorique, on teste toutes les suppressions possibles
            for k in range(len(t_part)):
                gs.append(t_part[:k] + t_part[k+1:])
                
        return gs

    def imp(self, hypothesis, hard_clauses):
        """ Test d'implicant via SAT solver. """
        solver = Glucose3()
        for clause in hard_clauses:
            solver.add_clause(list(clause))
        for lit in hypothesis:
            solver.add_clause([lit])
        is_sat = solver.solve()
        solver.delete()
        return not is_sat
    def m_cpi_xp(self, *, n=1, strategy="priority_order", random_seed=42, ordre_features=None):
        """
        Calcule une explication minimale (mCPI-Xp)[cite: 99].
        
        Processus :
        1. Appelle cpi_xp pour obtenir une explication valide (mais potentiellement non minimale)[cite: 108].
        2. Applique une réduction gloutonne : supprime chaque littéral s'il est redondant[cite: 196].
        """
        # 1. Obtenir l'explication de base (CPI-Xp)
        # Note : cpi_xp retourne un tuple trié, on le convertit en liste pour modification
        cpi_explanation = list(self.cpi_xp(n=n, strategy=strategy, random_seed=random_seed, ordre_features=ordre_features))
        
        # 2. Préparation des contraintes (Hard Clauses = Sigma + NOT(f*))
        # (Nécessaire pour le test d'implicant)
        theory_clauses = list(self.get_theory())
        
        if self._random_forest.n_classes == 2:
            inverse_model_cnf = self._random_forest.to_CNF(
                self._instance, self._binary_representation, self.target_prediction, tree_encoding=Encoding.MUS
            )
        else:
            inverse_model_cnf = self._random_forest.to_CNF_sufficient_reason_multi_classes(
                self._instance, self._binary_representation, self.target_prediction
            )
            
        hard_clauses = theory_clauses + list(inverse_model_cnf)

        # 3. Boucle de Minimisation (Greedy Deletion)
        # On itère sur une COPIE de la liste pour pouvoir modifier l'originale
        for literal in list(cpi_explanation):
            
            # On tente de supprimer le littéral courant
            candidate_explanation = [l for l in cpi_explanation if l != literal]
            
            # On teste si l'explication réduite est toujours valide
            # self.imp renvoie True si UNSAT (Valide), False si SAT (Invalide)
            is_valid_implicant = self.imp(candidate_explanation, hard_clauses)
            
            if is_valid_implicant:
                # UNSAT : Le modèle ne peut toujours pas prédire l'inverse.
                # Le littéral était donc inutile (redondant). On valide la suppression.
                cpi_explanation.remove(literal)
                # print(f"✓ Redondant supprimé : {literal}")
            else:
                # SAT : Le solveur a trouvé un contre-exemple.
                # Le littéral était nécessaire pour bloquer ce contre-exemple. On le garde.
                # print(f"✗ Nécessaire conservé : {literal}")
                pass

        # Retourne l'explication minimale triée
        return sorted(tuple(cpi_explanation))