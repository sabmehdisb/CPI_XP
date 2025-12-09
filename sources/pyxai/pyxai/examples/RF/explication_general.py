"""
Module d'analyse d'explications de modèles de machine learning
Utilise PyXAI pour générer et étendre des raisons d'explication
"""

import pandas as pd
from pyxai import Learning, Explainer, Tools


class TheoryExtender:
    """Classe pour étendre et manipuler les théories d'explication"""
    
    @staticmethod
    def negate_first(theory):
        """
        Parcourt la liste de tuples 'theory' et pour chaque tuple,
        remplace le premier élément par sa négation.
        
        Args:
            theory: Liste de tuples (old, new)
            
        Returns:
            Liste de tuples avec le premier élément négé: [(-old, new), ...]
            
        Exemple:
            Pour theory = [(-1, -5), (-6, 7)],
            retourne [(1, -5), (6, 7)]
        """
        return [(-old, new) for (old, new) in theory]
    
    @staticmethod
    def extend_theory(reason, theory):
        """
        Étend une raison basée sur une théorie donnée
        
        Args:
            reason: Liste des littéraux de la raison initiale
            theory: Liste de tuples de remplacement
            
        Returns:
            Liste étendue des littéraux
        """
        new_theory = TheoryExtender.negate_first(theory)
        reason_set = set(reason)  # Pour vérification rapide
        ordered_result = list(reason)  # On commence par les raisons initiales

        # Construction du dictionnaire de remplacements
        replacement_map = {}

        for old, new in new_theory:
            if old < 0:
                replacement_map.setdefault(old, set()).add(new)
                replacement_map.setdefault(-new, set()).add(-old)
            else:
                replacement_map.setdefault(old, set()).add(new)
                replacement_map.setdefault(-new, set()).add(-old)

        # Appliquer les remplacements
        for literal, new_literals in replacement_map.items():
            if literal in reason_set:
                for new_lit in new_literals:
                    if new_lit not in reason_set and new_lit not in ordered_result:
                        ordered_result.append(new_lit)
                        reason_set.add(new_lit)  # Marquer comme déjà ajouté

        return ordered_result
    
    @staticmethod
    # def minimize_reason(explainer, reason):
    #     """
    #     Minimise une raison en retirant les éléments non nécessaires
        
    #     Args:
    #         explainer: Instance d'Explainer
    #         reason: Liste des littéraux à minimiser
            
    #     Returns:
    #         Raison minimisée
    #     """
    #     general_reason = reason[:]
        
    #     for literal in reason[:]:  # Copie pour éviter la modification pendant l'itération
    #         temp_reason = general_reason[:]
    #         if literal in temp_reason:
    #             temp_reason.remove(literal)
                
    #             if explainer.is_reason(temp_reason):
    #                 general_reason.remove(literal)
        
    #     return general_reason

    def minimize_reason(explainer, reason):
        """
        Minimise une raison en retirant les éléments non nécessaires.
        
        Args:
            explainer: Instance d'Explainer.
            reason: Liste des littéraux à minimiser.
            
        Returns:
            Liste des littéraux minimisés (raison minimale).
        """
        general_reason = reason.copy()

        # On parcourt une copie pour ne pas modifier la liste pendant l'itération
        for literal in reason:
            # On tente de retirer le littéral
            candidate = [lit for lit in general_reason if lit != literal]
            
            # Si sans le littéral c'est encore une raison, on peut l'enlever
            if explainer.is_reason(candidate):
                general_reason = candidate

        return general_reason

class MLExplainerAnalyzer:
    """Classe principale pour l'analyse des explications de modèles ML"""
    
    def __init__(self, dataset_name=None, nb_instances=50):
        """
        Initialise l'analyseur
        
        Args:
            dataset_name: Nom du dataset (par défaut depuis Tools.Options.dataset)
            nb_instances: Nombre d'instances à analyser
        """
        self.dataset_name = dataset_name or Tools.Options.dataset
        self.nb_instances = nb_instances
        self.theory_extender = TheoryExtender()
        
        # Initialisation des attributs
        self.learner = None
        self.model = None
        self.explainer = None
        self.training_data = None
        self.good_instances = []
        
    def setup_model(self):
        """Initialise et entraîne le modèle de machine learning"""
        print("Configuration du modèle...")
        
        # Machine learning part
        self.learner = Learning.Scikitlearn(
            self.dataset_name + '.csv', 
            learner_type=Learning.CLASSIFICATION
        )
        
        self.model = self.learner.evaluate(
            method=Learning.HOLD_OUT, 
            output=Learning.DT
        )
        
        instance, prediction = self.learner.get_instances(self.model, n=1, correct=True)
        self.explainer = Explainer.initialize(
            self.model, 
            instance, 
            features_type=self.dataset_name + '.types'
        )
        
        print("Modèle configuré avec succès.")
    
    def prepare_training_data(self):
        """Prépare les données d'entraînement binarisées"""
        print("Préparation des données d'entraînement...")
        
        instances_dt_training = self.learner.get_instances(
            self.model, 
            n=None, 
            indexes=Learning.ALL, 
            details=True
        )
        
        # Préparation des données
        train_data = []
        train_labels = []
        
        for instance_dict in instances_dt_training:
            train_data.append(instance_dict["instance"])
            train_labels.append(instance_dict["label"])
        
        # Binarisation
        binarized_training = []
        nb_features = len(self.explainer.binary_representation)
        
        for i, instance in enumerate(train_data):
            self.explainer.set_instance(instance)
            binarized_row = [0 if l < 0 else 1 for l in self.explainer.binary_representation]
            binarized_row.append(train_labels[i])
            binarized_training.append(binarized_row)
        
        # Création du DataFrame
        columns = [f"X_{i}" for i in range(1, nb_features + 1)] + ['y']
        self.training_data = pd.DataFrame(binarized_training, columns=columns)
        
        # Sélection des bonnes instances
        self.good_instances = []
        for id_instance, instance_dict in enumerate(binarized_training):
            if len(self.good_instances) >= self.nb_instances:
                break
            self.good_instances.append(train_data[id_instance])
        
        print(f"Données préparées: {len(self.good_instances)} instances sélectionnées.")
    def analyze_explanations(self):
        """Analyse les explications pour chaque instance et stocke les cas modifiés"""
        print("Analyse des explications...")
        
        reasons = []
        new_reasons = []
        changed_explanations = []  # Stocke les cas où les features changent
        
        for i, instance in enumerate(self.good_instances):
            print(f"\n--- Instance {i+1}/{len(self.good_instances)} ---")
            
            self.explainer.set_instance(instance)
            
            # Génération de la raison suffisante
            #reason = self.explainer.minimal_sufficient_reason()
            reason =self.explainer.sufficient_reason_single_strategy(n=1, strategy="priority_order")
            reason=reason[0]
            #reason = self.explainer.sufficient_reason2()
            features_original = self.explainer.to_features(reason)
            print(f"Raison originale: {reason}")
            print(f"Features: {features_original}")
            #print(f"Est une raison valide: {self.explainer.is_sufficient_reason(reason)}")
            print(f"Est une raison valide: {self.explainer.is_reason(reason)}")
            
            # Extension de la théorie
            theory = self.explainer.get_theory()
            extended_reason = self.theory_extender.extend_theory(reason, theory)
            print(f"Raison étendue: {extended_reason}")
            
            # Minimisation
            general_reason = self.theory_extender.minimize_reason(self.explainer, extended_reason)
            features_general = self.explainer.to_features(general_reason)
            print(f"Raison general: {general_reason}")
            print(f"Features general: {features_general}")
            #print(f"Est une raison valide: {self.explainer.is_sufficient_reason(reason)}")
            print(f"Est une raison valide: {self.explainer.is_reason(general_reason)}")
            
            reasons.append(reason)
            new_reasons.append(general_reason)
            
            # Enregistrement si features ont changé
            if set(features_original) != set(features_general):
                changed_explanations.append({
                    "instance_id": i,
                    "instance": instance,
                    "original_reason": reason,
                    "Raison étendue":extended_reason,
                    "general_reason": general_reason,
                    "original_features": features_original,
                    "general_features": features_general,
                    "is_sufficient_theory_reason":self.explainer.is_sufficient_reason(general_reason)
                })

        return reasons, new_reasons, changed_explanations

    def evaluate_coverage(self, reasons, new_reasons):
        """Évalue la couverture des raisons sur toutes les instances"""
        print("\nÉvaluation de la couverture...")

        coverage_original = [0] * len(reasons)
        coverage_new = [0] * len(new_reasons)

        for i, (original_reason, new_reason) in enumerate(zip(reasons, new_reasons)):
            for test_instance in self.good_instances:
                self.explainer.set_instance(test_instance)

                if self.explainer.is_reason(original_reason):
                    coverage_original[i] += 1

                if self.explainer.is_reason(new_reason):
                    coverage_new[i] += 1

        return coverage_original, coverage_new

    def run_analysis(self):
        """Exécute l'analyse complète"""
        print("=== Début de l'analyse ===")
        
        try:
            # Configuration
            self.setup_model()
            self.prepare_training_data()
            
            # Analyse des explications
            reasons, new_reasons,changed_explanations = self.analyze_explanations()
            
            # # Évaluation de la couverture
            # coverage_original, coverage_new = self.evaluate_coverage(reasons, new_reasons)
            
            # # Affichage des résultats
            # print("\n=== RÉSULTATS ===")
            # print("Couverture des raisons originales:")
            # print(coverage_original)
            # print("\nCouverture des nouvelles raisons:")
            # print(coverage_new)
            
            # # Statistiques
            # avg_original = sum(coverage_original) / len(coverage_original)
            # avg_new = sum(coverage_new) / len(coverage_new)
            for item in changed_explanations:
                print(item)
                print("##########################")  # ← Ajoute un saut de ligne entre chaque élément
                  # Vérification avec is_sufficient_reason
                self.explainer.set_instance(item["instance"])
                assert self.explainer.is_sufficient_reason(item["general_reason"]), "La raison générale n'est pas suffisante"
            # print(f"\nCouverture moyenne originale: {avg_original:.2f}")
            # print(f"Couverture moyenne nouvelle: {avg_new:.2f}")
            # print(f"Amélioration: {((avg_new - avg_original) / avg_original * 100):.2f}%")
            print("nombre de raison total:",len(reasons))
            print("nombre de raison générale:",len(changed_explanations))
        except Exception as e:
            print(f"Erreur lors de l'analyse: {e}")
            raise


def main():
    """Fonction principale"""
    analyzer = MLExplainerAnalyzer(nb_instances=300)
    analyzer.run_analysis()


if __name__ == "__main__":
    main()