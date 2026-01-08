from pyxai import Learning, Explainer, Tools
import pandas as pd
import itertools
import pandas as pd
from cpi_explaine_BT import *

dataset_name = Tools.Options.dataset
# Tools.set_verbose(0)

df = pd.read_csv(dataset_name + '.csv')

learner = Learning.Xgboost(
    dataset_name + '.csv',
    learner_type=Learning.CLASSIFICATION,
)

model = learner.evaluate(method=Learning.HOLD_OUT, output=Learning.BT,max_depth=4)

instance, prediction = learner.get_instances(model, n=1, correct=True)
print("instance", instance)

explainer = Explainer.initialize(
    model,
    instance,
    features_type=dataset_name + '.types'
)

# th = explainer.get_theory()
target_prediction = explainer.target_prediction

# 3. Préparation des données binaires
v_binary = explainer.binary_representation
# Format (variable, valeur)
x_frozenset = frozenset((abs(lit), 1 if lit > 0 else 0) for lit in v_binary)

# 4. Calcul de l'explication CPI-Xp pour BT
# print("Calcul de la CPI-Xp pour Boosted Trees...")
cpi_xp = findCPIexplanation_BT(x_frozenset, explainer)
print("cpi_xp",cpi_xp)
if cpi_xp:
    reason4 = cpi_xp_to_sat_format(cpi_xp)
    print("cpi_xp_sat:", reason4)
else:
    print("Aucune explication trouvée.")
print("################")
ordre_features = []
reason = explainer.tree_specific_reason()

# print("binarized instance", explainer.binary_representation)

# tree_specific = explainer.tree_specific_reason()
# print("tree specific reason:", tree_specific)

# print("tree specific reason:", tree_specific)
# print("is implicant?:", explainer.is_implicant_BT((15,)
                                                  
# ))
# print("target_prediction:", target_prediction)
# print("instance", instance)
# exit(0)
reason2 = explainer.cpi_xp(
    n=1,
    strategy="priority_order",
    random_seed=None,
    ordre_features=ordre_features
)
reason3 = explainer.m_cpi_xp(n=1, strategy="priority_order", random_seed=None, ordre_features=ordre_features)

print("###############################################################")

features_original = explainer.to_features(reason)
print(f"Original PyXAI reason: {reason}")
print(f"PyXAI features: {features_original}")
print(f"Is a valid PyXAI reason: {explainer.is_reason(reason)}")
print("Is sufficient_reason PyXAI:", explainer.is_sufficient_reason(reason, n_samples=1000))
print("is implicant?:", explainer.is_implicant_BT(reason))

print("###############################################################")

# Generate 1 CPI-Xp explanation
features_original = explainer.to_features(reason2)
print(f"CPI-Xp reason: {reason2}")
print(f"CPI-Xp features: {features_original}")
print("CPI-Xp rule size:", len(explainer.to_features(reason2)))
print(f"Is a valid CPI-Xp reason: {explainer.is_reason(reason2)}")
print("Is sufficient_reason CPI-Xp:", explainer.is_sufficient_reason(reason2, n_samples=1000))
print("is implicant?:", explainer.is_implicant_BT(reason2))

print("###############################################################")

features_original = explainer.to_features(reason3)
print(f"mCPI-Xp reason: {reason3}")
print(f"mCPI-Xp features: {features_original}")
print("mCPI-Xp rule size:", len(explainer.to_features(reason3)))
print(f"Is a valid mCPI-Xp reason: {explainer.is_reason(reason3)}")
print("Is sufficient_reason mCPI-Xp:", explainer.is_sufficient_reason(reason3, n_samples=1000))
print("is implicant?:", explainer.is_implicant_BT(reason3))

print("######################################################################")
features_original = explainer.to_features(reason4)
print(f"Reason from procedure CPI-Xp (Explanations of Classifiers under Constraints): {reason4}")
print(f"CPI-Xp features: {features_original}")
print("CPI-Xp rule size:", len(explainer.to_features(reason4)))
print(f"Is a valid CPI-Xp reason: {explainer.is_reason(reason4)}")
print("Is sufficient_reason CPI-Xp:", explainer.is_sufficient_reason(reason4, n_samples=1000))
print("is implicant?:", explainer.is_implicant_BT(reason4))

print("######################################################################")

print("binarized instance", explainer.binary_representation)
print("instance", instance)
# On accède directement à l'objet boosted_trees interne
print("theory", explainer._boosted_trees.get_theory(explainer._binary_representation))
print("######################")

# print("cnf", explainer.to_CNF())
# print("number of rules", len(reason2[0]))
# print("number of possible feature combinations", reason2[2])
# print("number of duplicated rules with different feature combinations", reason2[1])
# print("number of smallest rules", smallest_rules)

# Display all columns (including the target)
# print("All columns:", df.columns.tolist())

print("Columns:", learner.get_details()[0]['feature_names'])

























# from pyxai import Learning, Explainer, Tools
# import pandas as pd
# import itertools
# import pandas as pd
# dataset_name = Tools.Options.dataset
# df = pd.read_csv(dataset_name + '.csv')
# learner = Learning.Scikitlearn(
#     dataset_name + '.csv',
#     learner_type=Learning.CLASSIFICATION
# )
# model = learner.evaluate(method=Learning.HOLD_OUT, output=Learning.DT)
# instance, prediction = learner.get_instances(model, n=1, correct=True)
# explainer = Explainer.initialize(
#     model,
#     instance,
#     features_type=dataset_name + '.types'
# )

# reason = explainer.sufficient_reason(n=all)
# # reason2 = explainer.sufficient_general_reason_pure_sat(n=2)
# reason2 = explainer.sufficient_reason_all_priority_orders(n="ALL")

# print("#################")
# for i in reason:
#     print("############################")
#     features_original = explainer.to_features(i)
#     print(f"Original reason: {i}")
#     print(f"Features: {features_original}")
#     print(f"Is a valid reason: {explainer.is_reason(i)}")
#     print("Is sufficient_reason:", explainer.is_sufficient_reason(i))

# print("########################################################################")

# # Find rules with minimal size
# rules_with_sizes = [(i, len(explainer.to_features(i))) for i in reason2[0]]
# min_size = min(size for _, size in rules_with_sizes)
# smallest_rules = [rule for rule, size in rules_with_sizes if size == min_size]

# # results = explainer.compare_strategies(n=1)
# for i in reason2[0]:
# # for i in smallest_rules:
#     print(i)
#     features_original = explainer.to_features(i)
#     print(f"Original reason 2: {i}")
#     print(f"Features 2: {features_original}")
#     print("Rule size:", len(explainer.to_features(i)))
#     print(f"Is a valid reason 2: {explainer.is_reason(i)}")
#     print("Is sufficient_reason 2:", explainer.is_sufficient_reason(i))

# print("########################################################################")
# print("binarized instance", explainer.binary_representation)
# print("instance", instance)
# print("theory", explainer.get_theory())
# print("number of rules 1", len(reason))
# print("number of rules 2", len(reason2[0]))
# print("number of possible feature combinations", reason2[2])
# print("number of duplicated rules with different feature combinations", reason2[1])
# print("most general rules", smallest_rules)

# for i in smallest_rules:
#     print("most general rules in features", explainer.to_features(i))

# # Display all columns (including the target)
# # print("All columns:", df.columns.tolist())

# columns_without_target = df.columns[:-1]  # All columns except the last
# print("Columns without the target:", columns_without_target)

