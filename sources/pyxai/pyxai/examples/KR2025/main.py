from decisionNode import DecisionNode
import numpy as np
from pyxai import Learning, Explainer, Tools ,Builder
import pandas as pd
from pysat.solvers import Glucose3
import random
import time
import json
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import os
import statistics
#Tools.set_verbose(0)
glucose = Glucose3()
# I load the dataset
path=Tools.Options.dataset
treeOptimized =Tools.Options.types
data = pd.read_csv(path+'.csv')
name=path
# Dividing the DataFrame into training, testing, and validation sets
train_df, validation_df = train_test_split(data, test_size=0.3, random_state=42)
train_df.columns=data.columns
validation_df.columns=data.columns
# Save the DataFrames to CSV files
train_df.to_csv('train_data.csv', index=False)
validation_df.to_csv('validation_data.csv', index=False)
best_parameters = DecisionNode.tuning('train_data.csv')
bt_learner = Learning.Xgboost('train_data.csv', learner_type=Learning.CLASSIFICATION) # 70%
# I create a xgboost model: the expert
bt_model = bt_learner.evaluate(method=Learning.HOLD_OUT, output=Learning.BT,seed=0,**best_parameters)
instance, prediction = bt_learner.get_instances(bt_model, n=1)
# I need an explainer BT
bt_explainer = Explainer.initialize(bt_model, instance, features_type=path +".types")
# I need the theory.... Currently, we collect clauses related to a binarised instance
# Récupérer la liste des arbres

# I need to collect the theory related to boolean variables....
for clause in bt_model.get_theory(bt_explainer.binary_representation):
    glucose.add_clause(clause)

binarized_training,raw_validation,label_validation,binarized_validation,binarized_data= [],[],[],[],[]
nb_features = len(bt_explainer.binary_representation)  # nb binarized features
# Iterating through the training set to binarize it
for i, instance in enumerate(bt_learner.data):
    bt_explainer.set_instance(instance)
    binarized_training.append([0 if l < 0 else 1 for l in bt_explainer.binary_representation] +  [bt_learner.labels[i]])

# Iterating through the validation set to binarize it
for i, instance in validation_df.iterrows():
    bt_explainer.set_instance(instance[:-1])
    raw_validation.append(instance[:-1])
    label_validation.append(instance[-1])
    binarized_validation.append([0 if l < 0 else 1 for l in bt_explainer.binary_representation] +  [instance[-1]])
for i, instance in data.iterrows():
    bt_explainer.set_instance(instance[:-1])
    binarized_data.append([0 if l < 0 else 1 for l in bt_explainer.binary_representation] +  [instance[-1]])
training_data=pd.DataFrame(binarized_training, columns=[f"X_{i}" for i in range(1, nb_features + 1)] + ['y'])
dt_learner = Learning.Scikitlearn(training_data, learner_type=Learning.CLASSIFICATION)
# We create K fold cross validation models
#optimisied configuration
opti="_default"
if treeOptimized=="True":
    opti="_optimised"
    best_parameters2 = DecisionNode.tuning2('train_data.csv')
    dt_models = dt_learner.evaluate(method=Learning.K_FOLDS, output=Learning.DT,seed=0,**best_parameters2)
#default configuration
else:
    dt_models = dt_learner.evaluate(method=Learning.K_FOLDS, output=Learning.DT,seed=0)
feature_names=dt_learner.get_details()[0]['feature_names']
all_scikit = dt_learner.get_raw_models()
# Declare the lists and dictionaries that we will use
trees,number_of_nodes_decision_tree,number_of_nodes_decision_tree0,tuple_of_instance_predictions_boosted_tree,precision_decision_tree_before_correction_on_the_validation_set,X_train1,y_train1,X_test,y_test,X_train_folds,y_train_folds,precision_of_the_bosted_tree_on_the_validation_set,tree_depth,precision_for_each_tree,f1_score_for_each_tree,number_of_different_predictions,elapsed_time_specific_reason_and_different_instances_for_all_trees=[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[]
# Dictionary to store the reasons
reasons_with_predictions_dict,reasons_with_predictions_dict1 = {},{}
# Iterating through the 10 decision tree models created with PyXAI
random_forest_models=[]
for i, dt_model in enumerate(dt_models) :
    random_forest_models.append(dt_model)
    # I take scikitLearn model
    clf = all_scikit[i]
    number_of_nodes_for_a_single_tree0=clf.tree_.node_count
    # Access the decision tree
    tree = clf.tree_
    # Get the decision tree in tuple form
    tree_tuple = DecisionNode.parse_decision_tree(tree, feature_names)
    transformed_tree = DecisionNode.transform_tree(tree_tuple)
    # Simplify the obtained decision tree
    simplified_tree=DecisionNode.simplify_tree_theorie(transformed_tree, glucose, [])
    trees.append(simplified_tree)
    depth_for_a_single_tree=clf.get_depth()
    #depth_for_a_single_tree=DecisionNode.tree_depth(transformed_tree)
    tree_depth.append(depth_for_a_single_tree)
    number_of_nodes_for_a_single_tree=DecisionNode.count_nodes(transformed_tree)
    number_of_nodes_decision_tree.append(number_of_nodes_for_a_single_tree0)
    number_of_nodes_decision_tree0.append(number_of_nodes_for_a_single_tree)
    # I collect  all instances from training set
    instances_dt_training = dt_learner.get_instances(dt_model, n=None, indexes=Learning.TRAINING, details=True)
    # I collect  all instances from test set
    instances_dt_test= dt_learner.get_instances(dt_model, n=None, indexes=Learning.TEST, details=True)
    X_train1,y_train1,X_1,y_1=[],[],[],[]
    ert = Explainer.initialize(dt_model)
    # Store instances and their labels in the lists X1 and Y1 of the test set
    for instance_dict in instances_dt_test:
        instance_dt = instance_dict["instance"]
        label_dt = instance_dict["label"]
        ert.set_instance(instance_dt)
        X_1.append(instance_dt)
        y_1.append(label_dt)
    # Store instances and their labels in the lists X_train1 and y_train1 of the training set that we will use for retraining
    for instance_dicti in instances_dt_training:
        instance_dt = instance_dicti["instance"]
        label_dt = instance_dicti["label"]
        X_train1.append(instance_dt)
        y_train1.append(label_dt)
    X_train_folds.append(X_train1)
    y_train_folds.append(y_train1)
# Calculate the accuracy of each decision tree
    precision_for_a_single_tree=DecisionNode.precision(transformed_tree, X_1, y_1)
    f1_score_for_a_single_tree=DecisionNode.f1_score(dt_model, X_1, y_1)
    # print(bt_learner.get_details()[0]["metrics"]['f1_score'])
    # print(f1_score_for_a_single_tree)
    # print(precision_for_a_single_tree)
    precision_for_each_tree.append(precision_for_a_single_tree*100)
    f1_score_for_each_tree.append(f1_score_for_a_single_tree*100)
    reasons_with_predictions,reasons_with_predictions1,single_tuple_instance_prediction,instancemalclasee,sufficient_reason,X_test1,y_test1 = [],[],[],[],[],[],[]
    nb,sufficient_reason_numeric2 = 0,0
    # Store instances and their labels in the lists X_test1 and y_test1 of the validation set
    elapsed_time_specific_reason_and_different_instances=0
    for id_instance,instance_dict in enumerate(binarized_validation):
        start_time=time.time()
        instance_dt = instance_dict[:-1]
        label_dt = instance_dict[-1]
        X_test1.append(instance_dt)
        prediction_dt=DecisionNode.classify(simplified_tree,instance_dt)
        bt_explainer.set_instance(raw_validation[id_instance])
        y_test1.append(bt_explainer.target_prediction)# y_test1 is the label, and it is considered as the prediction of the decision tree with 100% confidence
        # Display the instances
        assert [0 if l < 0 else 1 for l in bt_explainer.binary_representation] == list(instance_dt), "ca va pas!"
        if(bt_explainer.target_prediction==label_dt):
            sufficient_reason_numeric2+=1 # However, we still calculate the number of times the decision tree had a correct prediction with the real label of the validation set
        if prediction_dt != bt_explainer.target_prediction:
            ert.set_instance(instance_dt)
           # Store misclassified instances
            instancemalclasee.append(instance_dt)
            # Extract a specific explanation from the decision tree
            #start_time=time.time()
            tree_specific_reason = bt_explainer.tree_specific_reason(n_iterations=50)
            end_time=time.time()
            elapsed_time_specific_reason_and_different_instances+=(end_time - start_time)
            # Propagate the obtained tree-specific explanations (to avoid creating impossible instances in the retraining)
            propagations = glucose.propagate(tree_specific_reason)
            assert(propagations[0])
            tree_specific_reason1 = propagations[1]
            tree_specific_reason1=tuple(tree_specific_reason1)
            # Obtain a specific explanation and its prediction in tuple form
            reason_with_prediction1 = (tree_specific_reason1, bt_explainer.target_prediction)
            reasons_with_predictions1.append(reason_with_prediction1)
            reason_with_prediction = (tree_specific_reason, bt_explainer.target_prediction)
            reasons_with_predictions.append(reason_with_prediction)
            # Retrieve instances where the decision tree and the boosted tree give different predictions, and keep the boosted tree's prediction
            tuple_dataa = (tuple(instance_dt), bt_explainer.target_prediction)
            single_tuple_instance_prediction.append(tuple_dataa)
            nb += 1 # Calculate the number of times the boosted tree and decision tree give different predictions
    precision_for_a_single_bosted_tree = (sufficient_reason_numeric2 / len(X_test1))*100
    # f1_score_for_a_single_bosted_tree=DecisionNode.f1_score(bt_model, X_test1, y_test1)
    # print("f1_score_for_a_single_bosted_tree",f1_score_for_a_single_bosted_tree)
    precision_of_the_bosted_tree_on_the_validation_set.append(precision_for_a_single_bosted_tree)
    elapsed_time_specific_reason_and_different_instances_for_all_trees.append(elapsed_time_specific_reason_and_different_instances)
    number_of_different_predictions.append(nb)
    # Store the specific explanation and its prediction in a dictionary for rectification
    reasons_with_predictions_dict[f'reasons_with_predictions{i}'] = reasons_with_predictions 
    # Store the propagated specific explanation and its prediction in a dictionary for retraining
    reasons_with_predictions_dict1[f'reasons_with_predictions1{i}'] = reasons_with_predictions1
    X_test.append(X_test1) # Store all instances from all decision trees
    y_test.append(y_test1) # Store all labels from all decision trees (which is actually the label of the boosted tree)
    tuple_of_instance_predictions_boosted_tree.append(single_tuple_instance_prediction)
    locf = DecisionNode.precision(simplified_tree, X_test1, y_test1) # Calculate the accuracy of the decision tree before correction

    precision_decision_tree_before_correction_on_the_validation_set.append(locf)
unique_reasons_with_predictions1,unique_reasons_with_predictions = {},{}  # Dictionary to store the reasons
# Keep only unique rules, meaning no repetition of rules
for i in range(len(trees)):
    unique_reasons_with_predictions1_i = list(set(reasons_with_predictions_dict1[f'reasons_with_predictions1{i}']))
    unique_reasons_with_predictions1[f'unique_reasons1_{i}'] = unique_reasons_with_predictions1_i

    unique_reasons_with_predictions_i = list(set(reasons_with_predictions_dict[f'reasons_with_predictions{i}']))
    unique_reasons_with_predictions[f'unique_reasons_{i}'] = unique_reasons_with_predictions_i

# Iterate through each list of tuple_of_instance_predictions_boosted_tree
for i, l in enumerate(tuple_of_instance_predictions_boosted_tree):
   # Use a temporary set to track unique tuples
    unique_tuples = set()
    unique_list = []

    # Iterate through the tuples in the current list
    for tup in l:
        # If the tuple is not already in the unique set, add it to the unique list and the set
        if tup not in unique_tuples:
            unique_tuples.add(tup)
            unique_list.append(tup)

   # Replace the original list with the unique list without duplicates
    tuple_of_instance_predictions_boosted_tree[i] = unique_list
# Transform rules in the form of conditions into values: if X1 >= 0.5, assign value 1; otherwise, assign value -1
results,results1,data_frames = [],[],[]  # Create an empty list to store the results
for i in range(len(tuple_of_instance_predictions_boosted_tree)):
    # Create an empty list to store the results of the current iteration
    sufficient_reason_numeric,sufficient_reason_numeric2 = [],[]  
    for j in range(len(unique_reasons_with_predictions[f'unique_reasons_{i}'])):
        result = unique_reasons_with_predictions[f'unique_reasons_{i}'][j]
        sufficient_reason_numeric2.append(result) 
        re = unique_reasons_with_predictions1[f'unique_reasons1_{i}'][j]
        sufficient_reason_numeric.append(re) 
    results.append(sufficient_reason_numeric2) 
    results1.append(sufficient_reason_numeric)
# Put the rules into a data frame for each tree
# Create a list to store the DataFrames
# Create a DataFrame from the list z
    df = pd.DataFrame(sufficient_reason_numeric, columns=['sufficient_reason_numeric', 'prediction'])  # Remplacez 'Column1' et 'Column2' par les noms de colonnes appropriés
    # Add the DataFrame to the list of DataFrames
    data_frames.append(df)
# Now you have a list of lists 'results' where each element is a list of results for an iteration of the outer loop.
# We put the rules into a data frame for each tree, filling columns not in the rule with random values
# Create a binary DataFrame with columns for instances and predictions
binary_data_frames = []
for i, df in enumerate(data_frames):
    df_binary = pd.DataFrame(columns=feature_names)
# Iterate through each row of the DataFrame "df" containing sufficient reasons and associated predictions
    for index, row in df.iterrows():
        # Retrieve the prediction associated with the sufficient reason
        y = row["prediction"]
        # Initialize the values of instances X to random values between 0 and 1
        X = np.random.rand(nb_features)
        # Modify the values of instances X based on the sufficient reason
        for feature in row["sufficient_reason_numeric"]:
            if feature < 0:
                X[abs(feature)-1] = 0
            else:
                X[feature-1] = 1
        # Add the values of instances X and the prediction y to a new row of the binary DataFrame
        df_binary.loc[len(df_binary)] = np.concatenate((X, [y]))
    binary_data_frames.append(df_binary)
    # Display the binary DataFrame
# Create a list to store the DataFrames of X_test
X_test_dataframes,y_test_dataframes = [],[]
nouvelles_colonnes = {ancien_nom: f"X_{ancien_nom + 1}" for ancien_nom in range(nb_features)}
nouvelles_colonness = {ancien_nom: f"y" for ancien_nom in range(nb_features)}
# Loop through each list in X_test
for x_test_list in X_test:
# Create a DataFrame from the current list
    x_test_df = pd.DataFrame(x_test_list)
    x_test_df.rename(columns=nouvelles_colonnes, inplace=True)
# Add the DataFrame to the list
    X_test_dataframes.append(x_test_df)
# X_test_dataframes will now contain a list of DataFrames, one for each list in X_test
# Create a list to store the DataFrames of y_test
# Loop through each list in y_test
for y_test_list in y_test:
# Create a DataFrame from the current list
    y_test_df = pd.DataFrame(y_test_list)
    y_test_df.rename(columns=nouvelles_colonness, inplace=True)
# Add the DataFrame to the list
    y_test_dataframes.append(y_test_df)
# y_test_dataframes will now contain a list of DataFrames, one for each list in y_test
# Create a list to store the DataFrames of X_train_folds
X_train_folds_dataframes,y_train_folds_dataframes = [],[]
# Loop through each list in X_train_folds
for X_train_list in X_train_folds:
    # Create a DataFrame from the current list
    X_train_df = pd.DataFrame(X_train_list)
    X_train_df.rename(columns=nouvelles_colonnes, inplace=True)
    # Add the DataFrame to the list
    X_train_folds_dataframes.append(X_train_df)
# X_train_folds_dataframes will now contain a list of DataFrames, one for each list in X_train_folds
# Create a list to store the DataFrames of y_train_folds
# Loop through each list in y_train_folds
for y_train_list in y_train_folds:
    # Create a DataFrame from the current list
    y_train_df = pd.DataFrame(y_train_list)
    y_train_df.rename(columns=nouvelles_colonness, inplace=True)
   # Add the DataFrame to the list
    y_train_folds_dataframes.append(y_train_df)# y_train_folds_dataframes will now contain a list of DataFrames, one for each list in y_train_folds
#############################################################################################################
# Here, we will perform correction through Retraining:
#############################################################################################################
# To use retraining, we will:
# - Add instances of each rule to our training set a certain number of times
# - Add misclassified instances from our test set to our training set, assigning them the correct labels
# - Retrain the model for each tree
# - Calculate accuracy at each addition of the rule and instance for each tree
# Create a list to store the modified binary DataFrames
modified_binary_data_frames,precisions_retraining,number_of_nodes_decision_tree_retraining,time_retraining_for_each_rule,decisiontree,depth_retraining_for_each_rule = [],[],[],[],[],[]
# Get column names from df_copy
colonnes = feature_names
# Create an empty DataFrame with the same column names
df_vide = pd.DataFrame(columns=colonnes)
number_of_different_predictions_for_all_decision_tree_retraining=[]
# Loop to perform retraining on the 10 decision trees
nbinstdeletealltree,nbinstaddalltree,nb_instance_restante,nb_instance_intiale,instances_to_deletess=[],[],[],[],[]
for i, df_binary_i in enumerate(binary_data_frames):
        nb_instance_intiale.append(len(X_train_folds_dataframes[i]))
        # Apply the code to remove instances that do not meet the conditions
        X_train_folds_dataframes[i] = pd.DataFrame(X_train_folds_dataframes[i])
        y_train_folds_dataframes[i] =  pd.DataFrame(y_train_folds_dataframes[i])
        length_of_the_instance=len(X_test[i])
        number_of_different_predictions_for_a_decision_tree=[number_of_different_predictions[i]]
        tree_depth_for_decision_tree=[tree_depth[i]]
        # Create a copy of the binary DataFrame for each iteration
        df_copy = df_binary_i.copy()
        iteration_precisions = [precision_decision_tree_before_correction_on_the_validation_set[i]*100]
        number_of_nodes_decision_tree_validation_set=[number_of_nodes_decision_tree[i]]
        time_unwind,nbinstadded,nbinstdeleted=[],[],[]
        elapsed=0
        for index, row in df_copy.iterrows():
            psintanpossible=nb_features-len(results1[i][index][0])
            inst_creer=2**psintanpossible
            at_least_one_instance= max(int(0.01 * inst_creer), 1)
            nb_repetition = min(at_least_one_instance, 1000)
            nb_instance=len(X_train_folds_dataframes[i])
            unique_instances = set()
            l = 0  # Initialize the counter for repetitions
            while l < nb_repetition:
                new_row = row.copy()
                # Generate a new instance by randomly modifying the columns
                for column in df_copy.columns:
                    if 0 < new_row[column] < 1:  # Modify if the value is between 0 and 1.
                        new_row[column] = random.choice([0, 1])
                # Convert the row into a tuple to check for uniqueness.
                instance_tuple = tuple(new_row)
                # Check for uniqueness and add to df_vide if unique.
                if instance_tuple not in unique_instances:
                    unique_instances.add(instance_tuple)  # Ajouter au set pour l'unicité
                    df_vide = pd.concat([df_vide, pd.DataFrame([new_row])], ignore_index=True)  # Ajouter l'instance unique
                    l += 1  # Increment only when a unique instance is added.
            # Extract the last column for y_train if necessary.
            df_vide_column = df_vide.iloc[:, -1].to_frame()
            start_time = time.time()
            # Update X_train and y_train.
            X_train_folds_dataframes[i] = pd.concat([X_train_folds_dataframes[i], df_vide.iloc[:, :-1]], ignore_index=True)
            y_train_folds_dataframes[i] = pd.concat([y_train_folds_dataframes[i], df_vide_column], axis=0)
            # Reset df_vide for the next iteration, if necessary.
            df_vide = pd.DataFrame(columns=colonnes)
            # Retrieve the index of the current tree (i) and the current rule (index)
            tree_index = i
            rule_index = index
            # Retrieve the corresponding instance and prediction from tuple_of_instance_predictions_boosted_tree
            instance, prediction = tuple_of_instance_predictions_boosted_tree[tree_index][rule_index]
            # Add the instance to X_train_folds[i] and the prediction to y_train_folds[i]
            # Add a new row to X_train_folds_dataframes[i]
            new_instance_df = pd.DataFrame([instance], columns=X_train_folds_dataframes[i].columns)
            X_train_folds_dataframes[i] = pd.concat([X_train_folds_dataframes[i], new_instance_df], ignore_index=True)
            # Add a new row to y_train_folds_dataframes[i]
            # Inside the loop, add the predictions to the same column
            prediction_df = pd.DataFrame([prediction], columns=y_train_folds_dataframes[i].columns)
            y_train_folds_dataframes[i] = pd.concat([y_train_folds_dataframes[i], prediction_df], ignore_index=True)
            instances_to_delete = []
            nb_instance2=len(X_train_folds_dataframes[i])
            for idx, instance in X_train_folds_dataframes[i].iterrows():
                satisfied_conditions = DecisionNode.instance_satisfies_any_conditions2(instance, [results[i]])
                if satisfied_conditions:
                    for conditions, expected_output in satisfied_conditions:
                        predicted_output = y_train_folds_dataframes[i].loc[idx]['y']
                        if not predicted_output==expected_output:
                            d=predicted_output
                            limo=expected_output
                            s=satisfied_conditions
                            ik=instance
                            cn=[results[i][rule_index]]
                            instances_to_delete.append(idx)
                            break
            instances_to_deletess.append(instances_to_delete)
            X_train_folds_dataframes[i].drop(index=instances_to_delete, inplace=True)
            y_train_folds_dataframes[i]=y_train_folds_dataframes[i].drop(index=instances_to_delete)
            nbinstdeleted.append(len(instances_to_delete))
            nbinstadded.append(nb_repetition+1)
            if treeOptimized=="True":
                clf=DecisionTreeClassifier(**best_parameters2)
            else:
                clf=DecisionTreeClassifier()
            clf.fit(X_train_folds_dataframes[i], y_train_folds_dataframes[i])
            end_time = time.time() # Save the end time
            elapsed = (end_time - start_time)  # Calculate the elapsed time
            time_unwind.append(elapsed)
            # Get the feature names
            feature_namess = X_train_folds_dataframes[i].columns.tolist()
            nb_nd=clf.tree_.node_count
            # Access the decision tree
            tree = clf.tree_
            # Get the decision tree in tuple form
            tree_tuple = DecisionNode.parse_decision_tree(tree, feature_namess)
            transformed_tree = DecisionNode.transform_tree(tree_tuple)
            # Simplify the obtained decision tree
            simplified_tree=DecisionNode.simplify_tree_theorie(transformed_tree, glucose, [])
            simplified_tree=transformed_tree
            depth=clf.get_depth()
            tree_depth_for_decision_tree.append(depth)
            number_of_nodes_decision_tree_validation_set.append(nb_nd)
            # Evaluate the accuracy of the retrained model on the validation set
            test_accuracy = clf.score(X_test[i], y_test[i])
            numbre_of_instance_correct=DecisionNode.correct_instance(simplified_tree, X_test[i], y_test[i])
            numbre_ofdifferentprediction=length_of_the_instance-numbre_of_instance_correct
            number_of_different_predictions_for_a_decision_tree.append(numbre_ofdifferentprediction)
            # Add the accuracy of this iteration to the list of accuracies for this iteration
            iteration_precisions.append(test_accuracy*100)
        nb_instance_restante.append(len(X_train_folds_dataframes[i]))
        # Add the modified copy to the list of modified binary DataFrames
        nbinstdeletealltree.append(nbinstdeleted)
        nbinstaddalltree.append(nbinstadded)
        modified_binary_data_frames.append(df_copy)
        # Add the list of accuracies for this iteration to the overall list of accuracies
        precisions_retraining.append(iteration_precisions)
        number_of_nodes_decision_tree_retraining.append(number_of_nodes_decision_tree_validation_set)
        time_retraining_for_each_rule.append(time_unwind)
        depth_retraining_for_each_rule.append(tree_depth_for_decision_tree)
        number_of_different_predictions_for_all_decision_tree_retraining.append(number_of_different_predictions_for_a_decision_tree)
################################################################################
# Here, we will perform correction through Rectification
################################################################################
# To use rectification, we need to retrieve a list of rules that give predictions of 0 and 1 for each tree
resultats_dict = {}
# Using rectification with the extracted rules for each tree
for i in range(len(results)):
    resultat_i = results[i]
    resultat_0, resultat_1 = DecisionNode.split_list(resultat_i)
    # Store the results in the dictionary
    resultats_dict[f"resultat_{i}"] = {"resultat_0": resultat_0, "resultat_1": resultat_1}
# Declare the lists that we will use to store the results
precision_decision_tree_after_rectification_for_all_trees,time_rectification_for_each_rule,number_of_nodes_for_all_trees,depth_rectification_for_each_rule,number_of_different_predictions_in_rectification_in_all_trees,elapsed_time_sufficient_reason_for_all_trees= [],[],[],[],[],[]
# Loop to perform rectification on the 10 decision trees
for b in range(len(results)):
    dt_model=random_forest_models[b]
    length_of_the_instance=len(X_test[b])
    number_of_different_predictions_in_rectification=[number_of_different_predictions[b]]
    depth_rectification_=[dt_model.depth()]
    precision_decision_tree_after_rectification = [precision_decision_tree_before_correction_on_the_validation_set[b]*100]
    number_of_nodes=[dt_model.n_nodes()]
    time_unwind= []
    tree_rectified = trees[b]
    # Use b as an index to access the results of the outer loop
    result_key = f"resultat_{b}"
    resultat_1_b = resultats_dict[result_key]["resultat_1"]
    resultat_0_b = resultats_dict[result_key]["resultat_0"]  
    ert=Explainer.initialize(dt_model)
    ths=bt_model.get_theory(bt_explainer.binary_representation)
    theorie=DecisionNode.trasforme_list_tuple_to_binaire(ths,dt_model)
    theorie_clause=ert.condi(conditions=theorie)
    theorie_clause=DecisionNode.list_to_tuple_pairs(theorie_clause)
    for j in resultat_1_b:
        i=DecisionNode.trasforme_tuple_to_binaire(j[0],dt_model)
        start_time = time.time()
        dt_model = ert.rectify(conditions=i, label=1, tests=False,theory_cnf=theorie_clause)
        end_time = time.time()
        elapsed_time = (end_time - start_time) 
        precision_tree_rectified=DecisionNode.precision_model(dt_model, X_test[b], y_test[b])
        numbre_of_instance_correct=DecisionNode.correct_instance2(dt_model, X_test[b], y_test[b])
        numbre_ofdifferentprediction = length_of_the_instance - numbre_of_instance_correct
        total_nodes=dt_model.n_nodes()
        tree_depth_rectification__=dt_model.depth()
        number_of_nodes.append(total_nodes)
        depth_rectification_.append(tree_depth_rectification__)
        time_unwind.append(elapsed_time)
        number_of_different_predictions_in_rectification.append(numbre_ofdifferentprediction)
        precision_decision_tree_after_rectification.append(precision_tree_rectified*100)

    for i in resultat_0_b:
         i=DecisionNode.trasforme_tuple_to_binaire(i[0],dt_model)
         start_time = time.time()
         dt_model = ert.rectify(conditions=i, label=0, tests=False,theory_cnf=theorie_clause)
         end_time = time.time()
         elapsed_time = (end_time - start_time)
         precision_tree_rectified=DecisionNode.precision_model(dt_model, X_test[b], y_test[b])
         numbre_of_instance_correct=DecisionNode.correct_instance2(dt_model, X_test[b], y_test[b])
         numbre_ofdifferentprediction = length_of_the_instance - numbre_of_instance_correct
         total_nodes=dt_model.n_nodes()
         tree_depth_rectification__=dt_model.depth()
         number_of_nodes.append(total_nodes)
         depth_rectification_.append(tree_depth_rectification__)
         precision_decision_tree_after_rectification.append(precision_tree_rectified*100)
         time_unwind.append(elapsed_time)
         number_of_different_predictions_in_rectification.append(numbre_ofdifferentprediction)
    precision_decision_tree_after_rectification_for_all_trees.append(precision_decision_tree_after_rectification)
    number_of_nodes_for_all_trees.append(number_of_nodes)
    depth_rectification_for_each_rule.append(depth_rectification_)
    time_rectification_for_each_rule.append(time_unwind)
    number_of_different_predictions_in_rectification_in_all_trees.append(number_of_different_predictions_in_rectification)
    explainn=Explainer.initialize(dt_model)
    elapsed_time_sufficient_reason=[]
    size_sufficient_reason=[]
    X_full = [row[:-1] for row in binarized_data] 
    y_full = [row[-1] for row in binarized_data]   
    for i in X_full[:100]:
        explainn.set_instance(i)
        start_time=time.time()
        sufficient_reason=explainn.sufficient_reason()
        end_time=time.time()
        print("time:", end_time - start_time,"classic reason:",len(sufficient_reason))
        elapsed_time_sufficient_reason.append(end_time - start_time)
        size_sufficient_reason.append(len(sufficient_reason))
    elapsed_time_sufficient_reason_for_all_trees.append(elapsed_time_sufficient_reason)

# Store the results to extract them into a JSON file
data_ = {
    "dataset name":name,
    #"bt_model.n_trees": bt_model.n_trees,
    # "bt_model.n_nodes":bt_model.n_nodes(),
    #"bt_model.depth":bt_model.depth(),
    "accuracy_of_the_boosted_tree_on_test_set":bt_learner.get_details()[0]["metrics"]["accuracy"],
    "f1_score_of_the_boosted_tree_on_test_set":bt_learner.get_details()[0]["metrics"]["f1_score"],
    "accuracy_of_the_boosted_tree_on_validation_set":precision_of_the_bosted_tree_on_the_validation_set[0],
    "columns_of_the_non_binarized_data_set":bt_learner.get_details()[0]['feature_names'],
    "columns_of_the_binarized_data_set":dt_learner.get_details()[0]['feature_names'],
    "accuracy_for_each_tree_on_test_set":precision_for_each_tree,
    "f1_score_for_each_tree_on_test_set":f1_score_for_each_tree,
    "accuracy_for_each_decision_tree_before_correction_on_validation_set_relative_to_the_boosted_tree":precision_decision_tree_before_correction_on_the_validation_set,
    "number_of_nodes_for_each_tree_before_correction":number_of_nodes_decision_tree,
    "accuracy_after_retraining_for_each_rule" : precisions_retraining,
    "accuracy_after_rectification_for_each_rule": precision_decision_tree_after_rectification_for_all_trees,
    "number_of_nodes_after_rectification_for_each_rule": number_of_nodes_for_all_trees,
    "number_of_nodes_after_retraining_for_each_rule": number_of_nodes_decision_tree_retraining,
    "depth_after_retraining_for_each_rule": depth_retraining_for_each_rule,
    "depth_after_rectification_for_each_rule": depth_rectification_for_each_rule,
    "time_retraining_for_each_rule":time_retraining_for_each_rule,
    "time_rectification_for_each_rule":time_rectification_for_each_rule,
    "elapsed_time_specific_reason_and_different_instances_for_all_trees":elapsed_time_specific_reason_and_different_instances_for_all_trees,
    "elapsed_time_sufficient_reason_for_all_trees_after_correction":elapsed_time_sufficient_reason_for_all_trees,
    "number_of_different_instances_rectification":number_of_different_predictions_in_rectification_in_all_trees,
    "number_of_different_instances_retraining":number_of_different_predictions_for_all_decision_tree_retraining,
    "nbinstance_initial":nb_instance_intiale,
    "nbinstance_final":nb_instance_restante,
    "nbinstaddalltree_for_each_rule":nbinstaddalltree,
    "nbinstdeletealltree_for_each_rule":nbinstdeletealltree
}
# Name of the JSON file where you want to save the data
file_name =name+opti
# Writing the data to the JSON file
with open(file_name + ".json", 'w') as file_json:
    json.dump(data_, file_json)