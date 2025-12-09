import numpy as np
import pandas
from pyxai import Learning, Explainer, Tools ,Builder
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier  # Import XGBoost classifier

class DecisionNode:
    def __init__(self, feature_index, left=None, right=None):
        self.feature_index = feature_index
        self.left = left
        self.right = right

    class LeafNode:
        def __init__(self, class_label):
            self.class_label = class_label
    def correct_instance(model, X_test1, y_test1):
        s = 0
        predictions = []
        for instance in X_test1:
            predicted_label = model.predict_instance(instance)
            predictions.append(predicted_label)
            s += 1

        correct_predictions = sum(1 for pred, true_label in zip(predictions, y_test1) if pred == true_label)

        correct = correct_predictions
        return correct

    def precision_model(model, X_test1, y_test1):
        s = 0
        predictions = []
        for instance in X_test1:
            predicted_label = model.predict_instance(instance)
            predictions.append(predicted_label)
            s += 1

        correct_predictions = sum(1 for pred, true_label in zip(predictions, y_test1) if pred == true_label)

        accuracy = correct_predictions / len(X_test1)
        return accuracy
    def trasforme_tuple_to_binaire(tupl,dt_model):
        s=[]
        for n in tupl:
            is_inside=False
            for e in dt_model.map_features_to_id_binaries:
                if e[0]==n: 
                    s.append((dt_model.map_features_to_id_binaries[e])[0])
                    is_inside=True
                elif e[0]==-n:
                    s.append(-(dt_model.map_features_to_id_binaries[e])[0])
                    is_inside=True
            if is_inside is False:
                s.append((abs(n),Builder.GT,0.5, True if n < 0 else False))
        return s

    def trasforme_list_tuple_to_binaire(tupl,dt_model):
        s=[]
        for k in tupl:
            for n in k:
                is_inside=False
                for e in dt_model.map_features_to_id_binaries:
                    if e[0]==n: 
                        s.append((dt_model.map_features_to_id_binaries[e])[0])
                        is_inside=True
                    elif e[0]==-n:
                        s.append(-(dt_model.map_features_to_id_binaries[e])[0])
                        is_inside=True
                if is_inside is False:
                    s.append((abs(n),Builder.GT,0.5, True if n < 0 else False))
        return s
    def list_to_tuple_pairs(lst):
        if len(lst) % 2 != 0:
            raise ValueError("La liste doit contenir un nombre pair d'éléments.")
        
        return [(lst[i], lst[i+1]) for i in range(0, len(lst), 2)]
    # Here, we use simplification to reduce the number of nodes in the tree, applying the theory mentioned in the paper.
    def simplify_tree_theorie(tree_tuple, glucose, stack):
        def is_node_consistent(node, stack):
            if isinstance(node, tuple):
                feature_index, left, right = node
                stack.append(-feature_index)

                # Check consistency on the left
                left_consistent = glucose.propagate(stack)[0]
                stack.pop()

                # Check consistency on the right
                stack.append(feature_index)
                right_consistent = glucose.propagate(stack)[0]
                stack.pop()
                return left_consistent, right_consistent

            # If it's a leaf, it's still consistent
            return True, True

        def _simplify_tree_theorie(node, stack):
            if isinstance(node, tuple):
                feature_index, left, right = node
                # Check consistency on the left
                left_consistent, right_consistent = is_node_consistent(node, stack)

                if left_consistent:
                    # The left part is consistent, simplify recursively
                    left_simplified = _simplify_tree_theorie(left, stack + [-feature_index])
                else:
                    # The left part is inconsistent, replace with the right
                    left_simplified = _simplify_tree_theorie(right, stack + [feature_index])

                # Reset the tmp list
                tmp = []

                if right_consistent:
                    # The right part is consistent, simplify recursively
                    right_simplified = _simplify_tree_theorie(right, stack + [feature_index])
                else:
                    # The right part is inconsistent, replace with the left
                    right_simplified = _simplify_tree_theorie(left, stack + [-feature_index])

                # If both sides are identical, simplify by replacing with either side
                if str(left_simplified) == str(right_simplified):
                    return left_simplified

                return (feature_index, left_simplified, right_simplified)

            # If it's a leaf, do nothing
            return node

        return _simplify_tree_theorie(tree_tuple, stack)
    
    
    #Transform an decision tree into tuple form with (condition, left child, right child).
    def parse_decision_tree(tree, feature_names, node_index=0):
        # Check if the node is a leaf
        if tree.children_left[node_index] == tree.children_right[node_index]:
            # Get the value of the leaf
            leaf_value = tree.value[node_index].argmax()
            return leaf_value

        # Get the index of the feature used for the split
        feature_index = tree.feature[node_index]
        feature_name = feature_names[feature_index]

        # Splitting condition for the current node
        condition = feature_name

        # Recursion for the right child (if the condition is true)
        right_child_value = DecisionNode.parse_decision_tree(tree, feature_names, tree.children_right[node_index])

        # Recursion for the left child (if the condition is false)
        left_child_value = DecisionNode.parse_decision_tree(tree, feature_names, tree.children_left[node_index])

        if isinstance(right_child_value, int):
            right_child_value = (feature_name, right_child_value)
        if isinstance(left_child_value, int):
            left_child_value = (feature_name, left_child_value)

        return (condition, left_child_value, right_child_value)
    

    #Transforms the condition of a decision tree from (X_i, left_child, right_child) to (i, left_child, right_child).
    def transform_tree(tree):
        # If the tree is a tuple, extract condition and recursively transform left and right children.
        if isinstance(tree, tuple):
            condition, left_child, right_child = tree
            condition = int(condition.split('_')[1]) # Extracting the numeric value from the condition
            left_child = DecisionNode.transform_tree(left_child)
            right_child = DecisionNode.transform_tree(right_child)
            return (condition, left_child, right_child)
        else:
            leaf_value = tree
            if isinstance(leaf_value, str):
                return int(leaf_value.split('_')[1]) # Extracting the numeric value from the leaf
            else:
                return leaf_value
            

    #classifies an instance using a decision tree represented as a tuple.
    def classify(tree, instance):
        if isinstance(tree, tuple):
            # If the current node is a tuple, extract the feature, left, and right children.
            feature, left, right = tree
            #classify the left or right subtree.
            if instance[feature-1] == 0:
                return DecisionNode.classify(left, instance)
            else:
                return DecisionNode.classify(right, instance)
        else:
            # If the current node is a leaf node, return the classification result.
            return tree
        

    #Split into two lists the positive rules and the negative rules that are in the form of tuples.
    def split_list(liste_tuples):
        liste_0,liste_1 = [],[]
        for tuple_item in liste_tuples:
            valeur = tuple_item[1]
            if valeur == 0:
                liste_0.append(tuple_item)
            elif valeur == 1:
                liste_1.append(tuple_item)
        return liste_0, liste_1
      
    #Calculate the accuracy of a decision tree model by comparing predictions with true labels.
    def precision(vn, X_test1, y_test1):
        s = 0
        predictions = []
        for instance in X_test1:
            predicted_label = DecisionNode.classify(vn, instance)
            predictions.append(predicted_label)
            s += 1

        correct_predictions = sum(1 for pred, true_label in zip(predictions, y_test1) if pred == true_label)

        accuracy = correct_predictions / len(X_test1)
        return accuracy
    def f1_score(model, X_test, y_test):
        # Collect predictions
        predictions = []
        for instance in X_test:
            predicted_label = model.predict_instance(instance)
            predictions.append(predicted_label)
        
        # Calculate true positives, false positives, and false negatives
        # This implementation assumes a binary classification (0 and 1)
        true_positives = sum(1 for pred, true_label in zip(predictions, y_test) 
                            if pred == 1 and true_label == 1)
        
        false_positives = sum(1 for pred, true_label in zip(predictions, y_test) 
                            if pred == 1 and true_label == 0)
        
        false_negatives = sum(1 for pred, true_label in zip(predictions, y_test) 
                            if pred == 0 and true_label == 1)
        
        # Calculate precision and recall
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        
        # Calculate the F1-score
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        return f1
    #Calculate the number of nodes of a decision tree in tuple form;
    def count_nodes(tree):
        if isinstance(tree, tuple):
            count = 1  # Count the current node
            for subtree in tree[1:]:
                count += DecisionNode.count_nodes(subtree)  # Recursively count nodes in the subtrees
            return count
        else:
            return 1
        
    #Calculate the depth of a decision tree in tuple form; here, in depth, we also count the root node.
    def tree_depth(tree):
        if not tree:
            return 0  # The tree is empty, so the depth is 0

        # Check if the elements of the tree are indeed tuples
        if not isinstance(tree, tuple):
            return 0  # If the leaf is a number, the depth is 1
        condition, left_child_value, right_child_value = tree

        left_depth= DecisionNode.tree_depth(left_child_value)
        right_depth = DecisionNode.tree_depth(right_child_value)

        # The depth of this tree is the maximum depth between the left subtree and the right subtree, plus 1 for the root
        return max(left_depth, right_depth) + 1
    
        # Calculate the number of correct instances of a decision tree model by comparing predictions with true labels.
    def correct_instance2(model, X_test1, y_test1):
        s = 0
        predictions = []
        for instance in X_test1:
            predicted_label = model.predict_instance(instance)
            predictions.append(predicted_label)
            s += 1
        correct_predictions = sum(1 for pred, true_label in zip(predictions, y_test1) if pred == true_label)
        correct = correct_predictions
        return correct
    # Calculate the number of correct instances of a decision tree model by comparing predictions with true labels.
    def correct_instance(tree, X_test1, y_test1):
        s = 0
        predictions = []
        for instance in X_test1:
            predicted_label =DecisionNode.classify(tree, instance)
            predictions.append(predicted_label)
            s += 1
        correct_predictions = sum(1 for pred, true_label in zip(predictions, y_test1) if pred == true_label)
        correct = correct_predictions
        return correct
    # Function to check if an instance meets the conditions specified by a tuple
    def instance_satisfies_conditions(instance, conditions):
        for condition in conditions:
            col_index = abs(condition) - 1  # The index of the column in the instance (0-indexed)
            col_value = 1 if condition > 0 else 0  # Expected value for this column
            column_name = f'X_{abs(condition)}'
            if instance[column_name] != col_value:
                return False
        return True
    def instance_satisfies_any_conditions2(instance, conditions_list1):
        satisfied_conditions = []
        for conditions_list in conditions_list1:
            for conditions, expected_output in conditions_list:
                if DecisionNode.instance_satisfies_conditions(instance, conditions):
                    satisfied_conditions.append((conditions, expected_output))
        return satisfied_conditions
    #Searching for the best hyperparameters for the boosted tree to achieve high accuracy.
    def tuning(dataset):
        def load_dataset(dataset):
            data = pandas.read_csv(dataset).copy()

            # extract labels
            labels = data[data.columns[-1]]
            labels = np.array(labels)

            # remove the label of each instance
            data = data.drop(columns=[data.columns[-1]])

            # extract the feature names
            feature_names = list(data.columns)

            return data.values, labels, feature_names

        X, Y, names = load_dataset(dataset)
        model1 = XGBClassifier()
        param_grid = {
        'max_depth': [3, 4, 5, 6, 7, 8, 9],
        'n_estimators': [50, 100, 150, 200],
        'learning_rate': [0.02, 0.1,0.2, 0.3],
    }
        gridsearch1 = GridSearchCV(model1,
                                param_grid=param_grid,
                                scoring='balanced_accuracy', refit=True, cv=3,
                                return_train_score=True, verbose=10)

        gridsearch1.fit(X, Y)
        return gridsearch1.best_params_
    #Searching for the best hyperparameters for the decision_tree to achieve high accuracy.
    def tuning2(dataset):
        def load_dataset(dataset):
            data = pandas.read_csv(dataset).copy()

            # extract labels
            labels = data[data.columns[-1]]
            labels = np.array(labels)

            # remove the label of each instance
            data = data.drop(columns=[data.columns[-1]])

            # extract the feature names
            feature_names = list(data.columns)

            return data.values, labels, feature_names

        X, Y, names = load_dataset(dataset)
        model = DecisionTreeClassifier()
        param_grid = {
            'max_depth': [3, 4, 5, 6, 8, 9,15],
            
        }
        gridsearch = GridSearchCV(model,
                                param_grid=param_grid,
                                scoring='balanced_accuracy', refit=True, cv=3,
                                return_train_score=True, verbose=10)

        gridsearch.fit(X, Y)
        return gridsearch.best_params_