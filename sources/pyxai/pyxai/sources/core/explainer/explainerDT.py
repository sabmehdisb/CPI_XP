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
    def get_priority_order(self,instance, th):
        """
        Retourne l'ordre de suppression basé sur les implications transitives dans th
        Priorité aux littéraux les plus en amont de la chaîne (les plus généraux)
        """
        implications = {}  # Dict: impliquant -> [liste des impliqués]
        all_literals = set(instance)
        
        # Construire le graphe d'implications
        for clause in th:
            a, b = clause
            # ¬a → b
            if -a in all_literals and b in all_literals:
                if -a not in implications:
                    implications[-a] = []
                implications[-a].append(b)
            
            # ¬b → a  
            if -b in all_literals and a in all_literals:
                if -b not in implications:
                    implications[-b] = []
                implications[-b].append(a)
        
        # Calculer la profondeur de chaque littéral dans la chaîne
        def get_chain_depth(literal, visited=None):
            if visited is None:
                visited = set()
            if literal in visited:  # Éviter les cycles
                return 0
            visited.add(literal)
            
            if literal not in implications:
                return 0
            
            max_depth = 0
            for implied in implications[literal]:
                depth = 1 + get_chain_depth(implied, visited.copy())
                max_depth = max(max_depth, depth)
            
            return max_depth
        
        # Calculer la profondeur pour chaque littéral
        depths = {}
        for lit in instance:
            depths[lit] = get_chain_depth(lit)
        
        # Trier par profondeur décroissante (plus généraaux en premier)
        sorted_literals = sorted(instance, 
                            key=lambda x: depths[x], 
                            reverse=True)
        
        return sorted_literals

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
#              print("to_feature",self.to_features((-7,-6)
# ))
#              exit(0)
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
        """Version de test de la fonction"""
        merged = []
        print("ordre_features",ordre_features)
        # Détermine l'ordre de parcours des features
        if not ordre_features:
            # Ordre par défaut du dictionnaire
            features_a_parcourir = chains_by_feature.keys()
        else:
            # Utilise l'ordre spécifié
            features_a_parcourir = ordre_features
        print("features_a_parcourir",features_a_parcourir)
        # 1) concatène et déduplique selon l'ordre des features
        for feature in features_a_parcourir:
            if feature in chains_by_feature:  # Vérification que la feature existe
                chain = chains_by_feature[feature]
                for lit in chain:
                    if lit not in merged:
                        merged.append(lit)
        
        # 2) ajoute le reste de l'instance
        for lit in instance:
            if lit not in merged:
                merged.append(lit)
        
        return merged
    def merge_chains_and_instance_multiple_orders(self, chains_by_feature, instance):
        """
        Génère tous les ordres possibles de concaténation des chaînes de features.
        
        Args:
            chains_by_feature: dict { feature: [lits ordonnés] }
            instance: iterable de littéraux (int)
        
        Returns:
            list: [ [lits_ordre1], [lits_ordre2], [lits_ordre3], ... ]
        """
        import itertools
        
        multiple_orders = []
        
        # Obtenir seulement les chaînes de littéraux (pas les noms)
        chains = list(chains_by_feature.values())
        
        print(f"Nombre de chaînes: {len(chains)}")
        print(f"Chaînes: {chains}")
        # if chains:
        # Générer toutes les permutations possibles des chaînes
        for i, permutation in enumerate(itertools.permutations(range(len(chains)))):
            merged = []
            
            # Concaténer les chaînes selon l'ordre de la permutation
            chains_order = []
            for chain_index in permutation:
                chain = chains[chain_index]
                chains_order.append(f"T{chain_index+1}")
                for lit in chain:
                    if lit not in merged:
                        merged.append(lit)
            
            # Ajouter le reste de l'instance
            for lit in instance:
                if lit not in merged:
                    merged.append(lit)
            
            multiple_orders.append(merged)
            
            print(f"Ordre {i+1}: {' + '.join(chains_order)} = {merged}")
        # else:
        #     # Générer toutes les permutations possibles de instance complexité trop élevé
        #     print("Aucune chaîne trouvée, génération des permutations de l'instance...")
        #     instance_list = list(instance)  # Convertir en liste si ce n'est pas déjà le cas
            
        #     for i, permutation in enumerate(itertools.permutations(instance_list)):
        #         multiple_orders.append(list(permutation))
        #         print(f"Permutation {i+1}: {list(permutation)}")
        return multiple_orders

    def get_suppression_order_multiple(self, instance, th, strategy="priority_order_all", seed=None):
        """
        Retourne tous les ordres de suppression possibles pour la stratégie priority_order
        
        Args:
            instance: Liste des littéraux
            th: Théorie (clauses)
            strategy: Stratégie de suppression à utiliser
            seed: Graine pour la stratégie aléatoire (optionnel)
        
        Returns:
            Liste de listes ordonnées des littéraux selon toutes les permutations possibles
        """
        if strategy == "priority_order_all":
            chains_by_feature = self.get_feature_chain_lists_with_positive_first(th, instance)
            all_priority_orders = self.merge_chains_and_instance_multiple_orders(chains_by_feature, instance)
            return all_priority_orders
        else:
            # Fallback vers l'ancienne méthode pour les autres stratégies
            return [self.get_suppression_order(instance, th, strategy, seed,ordre_features=None)]

    def sufficient_reason_all_priority_orders(self, *, n=1):
        if n != 'ALL' and not isinstance(n, int):
            raise ValueError("Le paramètre 'n' doit être un entier positif ou 'ALL'")
        """
        Extrait des explications avec tous les ordres de priorité possibles
        """
        th = tuple(self.get_theory())
        print(f"Stratégie utilisée: priority_order_all")
        print(f"Théorie: {th}")
        print("##########")
        
        if self._instance is None:
            raise ValueError("Instance is not set")
        
        # Obtenir tous les ordres de suppression possibles
        all_suppression_orders = self.get_suppression_order_multiple(
            list(self._binary_representation), th, "priority_order_all"
        )
        
        print(f"\nNombre total d'ordres à tester: {len(all_suppression_orders)}")
        # Liste pour stocker toutes les explications trouvées (sans doublons)
        all_explanations = []
        repeated_rules=0
        combinaison_feature=len(all_suppression_orders)
        # Tester chaque ordre de suppression
        for order_idx, suppression_order in enumerate(all_suppression_orders):
            print(f"\n{'='*60}")
            print(f"TEST DE L'ORDRE {order_idx + 1}/{len(all_suppression_orders)}")
            print(f"{'='*60}")
            print(f"Ordre de suppression: {suppression_order}")
            
            # Réinitialiser l'instance pour chaque ordre
            cnf = self._tree.to_CNF(self._instance, target_prediction=self.target_prediction)
            instance = list(self._binary_representation)  # COPIE FRAÎCHE à chaque ordre
            original_length = len(instance)
            
            print("Instance initiale:", instance)
            print("CNF:", cnf)
            
            # Parcourir chaque littéral selon l'ordre de priorité
            for literal_to_test in suppression_order:
                # Vérifier si le littéral est encore dans l'instance
                if literal_to_test not in instance:
                    continue
                    
                print(f"\n=== Test du littéral: {literal_to_test} ===")
                
                # Créer une instance temporaire sans ce littéral
                temp_instance = [lit for lit in instance if lit != literal_to_test]
                print(f"Instance temporaire sans littéral {literal_to_test}: {temp_instance}")
                
                # Tester si l'instance réduite implique encore toutes les clauses CNF
                can_remove_literal = True
                
                for j, clause in enumerate(cnf, 1):
                    print(f"Test clause {j}: {clause}")
                    
                    # Négation de la clause
                    negated_clause = [(-lit) for lit in clause]
                    
                    # Construire la formule de test
                    test_formula = list(th)
                    
                    # Ajouter la négation de la clause
                    for neg_lit in negated_clause:
                        test_formula.append((neg_lit,))
                    
                    # Ajouter l'instance temporaire
                    for inst in temp_instance:
                        test_formula.append((inst,))
                    
                    # Ajouter les contraintes d'exclusion des explications précédentes
                    # for prev_explanation in all_explanations:
                    #     exclusion_clause = tuple(-lit for lit in prev_explanation)
                    #     test_formula.append(exclusion_clause)
                    #     print(f"Contrainte d'exclusion ajoutée: {exclusion_clause}")
                    
                    test_formula = tuple(test_formula)
                    print(f"Formule de test: {len(test_formula)} clauses")
                    
                    # Tester avec le solveur SAT
                    try:
                        from pysat.solvers import Glucose3
                        glucose = Glucose3()
                        
                        for clause_to_add in test_formula:
                            if clause_to_add:  # Éviter les clauses vides
                                glucose.add_clause(list(clause_to_add))
                        
                        is_sat = glucose.solve()
                        print(f"Clause {j} - Formule SAT: {is_sat}")
                        
                        # Nettoyer le solveur après chaque test
                        glucose.delete()
                        
                        if is_sat:
                            can_remove_literal = False
                            print(f"Clause {j} n'est pas impliquée - le littéral {literal_to_test} ne peut pas être supprimé")
                            break
                        else:
                            print(f"Clause {j} est bien impliquée par l'instance réduite")
                            
                    except Exception as e:
                        print(f"Erreur avec le solveur: {e}")
                        can_remove_literal = False
                        break
                
                # Décision pour ce littéral
                if can_remove_literal:
                    instance.remove(literal_to_test)
                    print(f"✓ SUPPRESSION: Littéral {literal_to_test} supprimé définitivement")
                    print(f"Instance après suppression: {instance}")
                else:
                    print(f"✗ CONSERVATION: Littéral {literal_to_test} conservé")
            
            # Vérifier si on a trouvé une nouvelle explication
            current_explanation = tuple(sorted(instance))
            
            print(f"\n=== RÉSULTAT POUR L'ORDRE {order_idx + 1} ===")
            print(f"Explication candidate: {current_explanation}")
            print(f"Déjà dans la liste: {current_explanation in all_explanations}")
            
            if current_explanation and current_explanation not in all_explanations:
                all_explanations.append(current_explanation)
                print(f"✓ NOUVELLE EXPLICATION TROUVÉE (Ordre {order_idx + 1})")
                print(f"Explication: {current_explanation}")
                print(f"Nombre de littéraux: {len(current_explanation)}")
                print(f"Nombre de littéraux supprimés: {original_length - len(current_explanation)}")
            else:
                if not current_explanation:
                    print(f"✗ Explication vide (Ordre {order_idx + 1})")
                else:
                    print(f"✗ Explication déjà trouvée (Ordre {order_idx + 1})")
                    repeated_rules+=1
            
            # Arrêter si on a trouvé assez d'explications
            if n != 'ALL' and len(all_explanations) >= n:
                print(f"\nArrêt: {n} explications trouvées")
                break

        
        print(f"\n{'='*60}")
        print(f"RÉSUMÉ FINAL - {len(all_explanations)} EXPLICATIONS UNIQUES TROUVÉES")
        print(f"{'='*60}")
        
        for i, explanation in enumerate(all_explanations, 1):
            print(f"Explication {i}: {explanation} (taille: {len(explanation)})")
        
        return all_explanations,repeated_rules,combinaison_feature
    from pysat.solvers import Glucose3

    def sufficient_reason_single_strategy(self, *, n=1, strategy="priority_order", random_seed=42, ordre_features=None):
        """
        Extrait plusieurs explications (AXp) avec une stratégie donnée.
        Pour trouver 'n' explications différentes, on fait varier l'ordre de suppression à chaque itération.
        """
        th = tuple(self.get_theory())
        print(f"Stratégie utilisée: {strategy}")
        print(f"Théorie: {len(th)} clauses")
        print("##########")
        print("ordre_features", ordre_features)
        
        if self._instance is None:
            raise ValueError("Instance is not set")
        
        # Récupérer la cible (ce qu'on doit prouver)
        cnf_target = self._tree.to_CNF(self._instance, target_prediction=self.target_prediction)
        print(f"CNF Cible (à prouver): {cnf_target}")
        print("##################")

        all_explanations = []
        k = 0
        max_retries = n * 5  # Sécurité pour éviter boucle infinie si on ne trouve pas n explications
        attempts = 0

        # Boucle pour trouver n explications différentes
        while k < n and attempts < max_retries:
            attempts += 1
            print(f"\n{'='*50}")
            print(f"RECHERCHE EXPLICATION {k+1}/{n} (Tentative {attempts})")
            print(f"{'='*50}")
            
            instance = list(self._binary_representation)
            original_length = len(instance)
            print("instance initiale", instance)
            
            # IMPORTANT : On change la seed à chaque itération k pour espérer un ordre différent
            # et donc une explication différente (si la stratégie le permet)
            current_seed = (random_seed + attempts) if random_seed is not None else None
            
            suppression_order = self.get_suppression_order(
                instance, th, strategy, 
                seed=current_seed,
                ordre_features=ordre_features if strategy == "priority_order" else None
            )
            print("Ordre de suppression:", suppression_order)

            # --- Début Algorithme Glouton ---
            for literal_to_test in suppression_order:
                if literal_to_test not in instance:
                    # print(f"\n=== Littéral {literal_to_test} déjà supprimé ===")
                    continue
                    
                print(f"\n=== Test du littéral: {literal_to_test} ===")
                
                # Créer une instance temporaire sans ce littéral
                temp_instance = [lit for lit in instance if lit != literal_to_test]
                print(f"Instance temporaire: {temp_instance}")
                
                can_remove_literal = True
                
                # Optimisation: Création du solveur UNE FOIS pour ce test
                solver = Glucose3()
                # 1. Ajouter la Théorie
                for clause in th:
                    solver.add_clause(list(clause))
                # 2. Ajouter l'instance réduite (Hypothèse)
                for lit in temp_instance:
                    solver.add_clause([lit])
                
                # 3. Vérifier si (Théorie + Instance) => CNF Cible
                # On vérifie clause par clause
                for j, clause_cible in enumerate(cnf_target, 1):
                    # On teste : (Base) => Clause ?
                    # Équivalent à : (Base) AND NOT(Clause) est UNSAT ?
                    
                    # Négation de la clause (assumptions)
                    # Si clause = (A v B), on assume -A et -B
                    assumptions = [-lit for lit in clause_cible]
                    
                    # Résolution
                    if solver.solve(assumptions=assumptions):
                        # SAT = Contre-exemple trouvé -> L'implication est fausse
                        # Le modèle montre un cas où l'instance est respectée mais pas la cible
                        print(f"Clause {j} {clause_cible} NON impliquée (Contre-exemple trouvé)")
                        # model = solver.get_model()
                        # print(f"Modèle: {model}")
                        can_remove_literal = False
                        break # Pas la peine de tester les autres clauses
                    else:
                        # UNSAT = Implication Vraie pour cette clause
                        print(f"Clause {j} bien impliquée.")

                solver.delete() # Nettoyage mémoire
                
                # Décision
                if can_remove_literal:
                    instance.remove(literal_to_test)
                    print(f"✓ SUPPRESSION: Littéral {literal_to_test} supprimé")
                else:
                    print(f"✗ CONSERVATION: Littéral {literal_to_test} nécessaire")
            # --- Fin Algorithme Glouton ---

            # Vérification du résultat
            current_explanation = tuple(sorted(instance))
            
            if len(current_explanation) == 0:
                print("⚠ Explication vide trouvée (bizarre), on continue...")
                continue

            if current_explanation in all_explanations:
                print(f"\n⚠ Explication déjà connue retrouvée: {current_explanation}")
                print("  -> On relance avec un nouvel ordre (nouvelle seed).")
                # On n'incrémente pas k, on réessaie
            else:
                print(f"\n✓ NOUVELLE EXPLICATION TROUVÉE: {current_explanation}")
                all_explanations.append(current_explanation)
                k += 1 # On a trouvé une explication valide, on avance

        print(f"\n{'='*60}")
        print(f"RÉSUMÉ FINAL - {len(all_explanations)} EXPLICATIONS UNIQUES")
        print(f"{'='*60}")
        
        for i, explanation in enumerate(all_explanations, 1):
            print(f"Explication {i}: {explanation}")
        
        return all_explanations
    from pysat.solvers import Glucose3

    def sufficient_reason_single_strategyy(self, *, n=1, strategy="priority_order", random_seed=42, ordre_features=None):
        """
        Extrait 'n' explications (AXp) pour un Arbre de Décision (DT).
        Code nettoyé : affiche uniquement la progression et les suppressions réussies.
        """
        # 1. Préparation
        th = tuple(self.get_theory())
        if self._instance is None:
            raise ValueError("L'instance n'est pas définie.")

        # Pour un DT, la CNF cible représente le chemin ou la logique de prédiction
        cnf_target = self._tree.to_CNF(self._instance, target_prediction=self.target_prediction)
        
        print(f"--- Configuration ---")
        print(f"Stratégie : {strategy}")
        print(f"Théorie   : {len(th)} clauses")
        print(f"Cible CNF : {len(cnf_target)} clauses à satisfaire")
        print(f"Objectif  : Trouver {n} explication(s) distincte(s)")
        print(f"{'-'*25}")

        all_explanations = []
        k = 0
        attempts = 0
        max_retries = n * 5  # Sécurité anti-boucle infinie

        # Boucle principale pour trouver n explications
        while k < n and attempts < max_retries:
            attempts += 1
            print(f"\n> Recherche Explication {k+1}/{n} (Essai {attempts})")
            
            # On repart toujours de l'instance complète
            current_instance = list(self._binary_representation)
            
            # Variation de la seed pour changer l'ordre si on cherche plusieurs explications
            current_seed = (random_seed + attempts) if random_seed is not None else None
            
            # Récupération de l'ordre de test
            suppression_order = self.get_suppression_order(
                current_instance, th, strategy, 
                seed=current_seed,
                ordre_features=ordre_features if strategy == "priority_order" else None
            )

            # === ALGORITHME GLOUTON (Minimisation) ===
            for literal_to_test in suppression_order:
                if literal_to_test not in current_instance:
                    continue
                
                # On tente d'enlever ce littéral
                temp_instance = [lit for lit in current_instance if lit != literal_to_test]
                can_remove = True
                
                # Initialisation du solveur pour ce test
                # (Note : Pour un DT, recréer le solveur est acceptable car la théorie est petite)
                solver = Glucose3()
                for clause in th: solver.add_clause(list(clause))     # Théorie
                for lit in temp_instance: solver.add_clause([lit])    # Instance réduite
                
                # Vérification : Est-ce que l'instance réduite implique TOUTES les clauses de la cible ?
                # Test : (Base) AND NOT(Clause) doit être UNSAT pour toutes les clauses
                for clause_cible in cnf_target:
                    assumptions = [-lit for lit in clause_cible] # Négation de la clause
                    
                    if solver.solve(assumptions=assumptions):
                        # SAT = Contre-exemple trouvé -> L'implication est brisée
                        can_remove = False
                        break # Pas la peine de tester les autres, ce littéral est nécessaire

                solver.delete() # Nettoyage

                # Décision
                if can_remove:
                    current_instance.remove(literal_to_test)
                    print(f"  ✓ Supprimé : {literal_to_test}")
                # else:
                #     print(f"  . Gardé (Nécessaire)") # Décommenter si tu veux voir les échecs

            # === FIN DE L'ITÉRATION ===
            final_explanation = tuple(sorted(current_instance))
            
            if final_explanation not in all_explanations:
                all_explanations.append(final_explanation)
                print(f"✅ EXPLICATION VALIDÉE : {final_explanation} (Taille : {len(final_explanation)})")
                k += 1
            else:
                print("⚠ Doublon trouvé (déjà connue). On relance avec un nouvel ordre...")

        # Résumé final
        print(f"\n{'='*40}")
        print(f"RÉSUMÉ : {len(all_explanations)} explication(s) trouvée(s)")
        for i, expl in enumerate(all_explanations, 1):
            print(f"  {i}. {expl}")
        print(f"{'='*40}")
        
        return all_explanations
    def compare_strategies(self, *, n=1, random_seed=42):
        """
        Compare toutes les stratégies de suppression
        """
        strategies = ["priority_order", "beginning_to_end", "end_to_beginning", "random"]
        results = {}
        
        print(f"\n{'='*80}")
        print("COMPARAISON DES STRATÉGIES DE SUPPRESSION")
        print(f"{'='*80}")
        
        for strategy in strategies:
            print(f"\n{'#'*60}")
            print(f"TEST DE LA STRATÉGIE: {strategy.upper()}")
            print(f"{'#'*60}")
            
            try:
                explanations = self.sufficient_reason_single_strategy(
                    n=n, 
                    strategy=strategy, 
                    random_seed=random_seed
                )
                
                # Calculer les statistiques
                if explanations:
                    avg_size = sum(len(exp) for exp in explanations) / len(explanations)
                    min_size = min(len(exp) for exp in explanations)
                    max_size = max(len(exp) for exp in explanations)
                    
                    # Calculer les features pour chaque explication
                    features_list = []
                    for explanation in explanations:
                        try:
                            features = self.to_features(explanation)
                            features_list.append(features)
                        except Exception as e:
                            print(f"Erreur lors du calcul des features pour {explanation}: {e}")
                            features_list.append(None)
                else:
                    avg_size = min_size = max_size = 0
                    features_list = []
                
                results[strategy] = {
                    'explanations': explanations,
                    'count': len(explanations),
                    'avg_size': avg_size,
                    # 'min_size': min_size,
                    # 'max_size': max_size,
                    'features': features_list
                }
                
            except Exception as e:
                print(f"Erreur avec la stratégie {strategy}: {e}")
                results[strategy] = {
                    'explanations': [],
                    'count': 0,
                    'avg_size': 0,
                    # 'min_size': 0,
                    # 'max_size': 0,
                    'features': []
                }
        
        # Affichage du tableau comparatif
        print(f"\n{'='*100}")
        print("TABLEAU COMPARATIF DES RÉSULTATS")
        print(f"{'='*100}")
        
        # print(f"{'Stratégie':<20} {'Nb Expl.':<10} {'Taille Moy.':<12} {'Min':<6} {'Max':<6} {'Features':<45}")
        # print("-" * 100)
        print(f"{'Stratégie':<20} {'Nb Expl.':<10} {'Taille Moy.':<12} {'Features':<45}")
        print("-" * 100)
        for strategy, result in results.items():
            # Préparer l'affichage des features
            if result['features'] and any(f is not None for f in result['features']):
                # Compter les features valides
                valid_features = [f for f in result['features'] if f is not None]
                if valid_features:
                    features_display = f"{(valid_features)} "
                else:
                    features_display = "Aucune"
            else:
                features_display = "Aucune"
            
            # print(f"{strategy:<20} {result['count']:<10} "
            #     f"{result['avg_size']:<12.2f} {result['min_size']:<6} {result['max_size']:<6} {features_display:<45}")
            print(f"{strategy:<20} {result['count']:<10} "
                 f"{result['avg_size']:<12.2f}  {features_display:<45}")
        # Trouver la meilleure stratégie
        valid_strategies = {k: v for k, v in results.items() if v['count'] > 0}
        if valid_strategies:
            best_strategy = min(valid_strategies.keys(), 
                            key=lambda s: valid_strategies[s]['avg_size'])
            
            # print(f"\n Meilleure stratégie (plus petite taille moyenne): {best_strategy}")
            # print(f"   Taille moyenne: {results[best_strategy]['avg_size']:.2f}")
        
        # Afficher toutes les explications par stratégie
        print(f"\n{'='*100}")
        print("DÉTAIL DES EXPLICATIONS PAR STRATÉGIE")
        print(f"{'='*100}")
        
        for strategy, result in results.items():
            print(f"\n{strategy.upper()}:")
            if result['explanations']:
                for i, explanation in enumerate(result['explanations'], 1):
                    features_info = ""
                    if i <= len(result['features']) and result['features'][i-1] is not None:
                        features = result['features'][i-1]
                        # Affichage simplifié des features (adaptez selon le type de vos features)
                        if hasattr(features, 'shape'):
                            features_info = f" - Features: shape {features.shape}"
                        elif isinstance(features, (list, tuple)):
                            features_info = f" - Features: {len(features)} éléments"
                        else:
                            features_info = f" - Features: {type(features).__name__}"
                    
                    print(f"  {i}. {explanation} (taille: {len(explanation)}){features_info}")
            else:
                print("  Aucune explication trouvée")
        
        return results
    def sufficient_general_reason_pure_sat(self, *, n=1, time_limit=None):
        """
        Version pure énumération SAT + minimisation
        Plus fidèle à la suggestion originale
        """
        th = tuple(self.get_theory())
        print("Théorie:", th)
        
        if self._instance is None:
            raise ValueError("Instance is not set")
        
        cnf = self._tree.to_CNF(self._instance, target_prediction=self.target_prediction)
        print("CNF:", cnf)
        
        sufficient_reasons = []
        blocking_clauses = []
        k = 0
        
        while k < n:
            print(f"\n========== ÉNUMÉRATION SAT #{k+1} ==========")
            print("instance", self._binary_representation)
            print("Théorie:", th)
            
            # ÉTAPE 1: Obtenir un modèle SAT
            try:
                from pysat.solvers import Glucose3
                glucose = Glucose3()
                
                # Ajouter théorie + CNF + blocking clauses
                for clause in th:
                    if clause:
                        glucose.add_clause(list(clause))
                
                for clause in cnf:
                    if clause:
                        glucose.add_clause(list(clause))
                
                for blocking_clause in blocking_clauses:
                    if blocking_clause:
                        glucose.add_clause(list(blocking_clause))
                
                if not glucose.solve():
                    print(f"Aucune nouvelle solution trouvée après {k} raisons")
                    glucose.delete()
                    break
                
                # Récupérer le modèle complet
                full_model = glucose.get_model()
                print(f"Modèle SAT complet: {full_model}")
                
                # Extraire les littéraux positifs pertinents
                # Adapter selon votre domaine - ici je prends les littéraux de l'instance originale
                original_vars = set(abs(lit) for lit in self._binary_representation) if self._binary_representation else set()
                
                if not original_vars and cnf:
                    # Si pas d'instance originale, utiliser les variables de la CNF
                    original_vars = set(abs(lit) for clause in cnf for lit in clause)
                
                candidate_model = []
                for lit in full_model:
                    if lit is not None and abs(lit) in original_vars:
                        candidate_model.append(lit)
                
                glucose.delete()
                
                if not candidate_model:
                    print("Aucun littéral pertinent dans le modèle SAT")
                    break
                    
                print(f"Modèle candidat: {candidate_model}")
                
            except Exception as e:
                print(f"Erreur énumération SAT: {e}")
                break
            
            # ÉTAPE 2: Minimiser en prime implicant
            prime_implicant = self._minimize_prime_implicant_v2(candidate_model, th, cnf)
            
            if not prime_implicant:
                print("Échec de la minimisation")
                break
            
            # Vérification que tous les littéraux du prime_implicant sont bien dans l'instance d'origine
            implicant_valide = True
            for lit in prime_implicant:
                if lit not in self._binary_representation:
                    print(f"⛔ Littéral {lit} n'est pas dans l'instance d'origine")
                    implicant_valide = False
                    break

            if not implicant_valide:
                # Ajouter blocking clause pour éviter de retrouver ce prime_implicant
                blocking_clause = tuple(-lit for lit in prime_implicant)
                blocking_clauses.append(blocking_clause)
                
                print(f"Prime implicant rejeté: {prime_implicant}")
                print(f"Blocking clause ajoutée: {blocking_clause}")
                continue  # On repart sans incrémenter k

            # ÉTAPE 3: Prime implicant valide - l'ajouter aux résultats
            blocking_clause = tuple(-lit for lit in prime_implicant)
            blocking_clauses.append(blocking_clause)
            sufficient_reasons.append(prime_implicant)
            
            print(f"✅ Prime implicant #{k+1} valide: {prime_implicant}")
            print(f"Blocking clause: {blocking_clause}")
            
            k += 1
        
        print(f"\n=== RÉSULTATS FINAUX (Pure SAT) ===")
        for i, reason in enumerate(sufficient_reasons, 1):
            print(f"Raison #{i}: {reason}")
        
        return sufficient_reasons

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
