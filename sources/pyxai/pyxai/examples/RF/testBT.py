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
print("Calcul de la CPI-Xp pour Boosted Trees...")
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

print("###############################################################")

# Generate 1 CPI-Xp explanation
features_original = explainer.to_features(reason2)
print(f"CPI-Xp reason: {reason2}")
print(f"CPI-Xp features: {features_original}")
print("CPI-Xp rule size features:", len(explainer.to_features(reason2)))
print("CPI-Xp rule size:", len(reason2))
print(f"Is a valid CPI-Xp reason: {explainer.is_reason(reason2)}")
print("Is sufficient_reason CPI-Xp:", explainer.is_sufficient_reason(reason2, n_samples=1000))

print("###############################################################")

features_original = explainer.to_features(reason3)
print(f"mCPI-Xp reason: {reason3}")
print(f"mCPI-Xp features: {features_original}")
print("mCPI-Xp rule size:", len(explainer.to_features(reason3)))
print(f"Is a valid mCPI-Xp reason: {explainer.is_reason(reason3)}")
print("Is sufficient_reason mCPI-Xp:", explainer.is_sufficient_reason(reason3, n_samples=1000))

print("######################################################################")
features_original = explainer.to_features(reason4)
print(f"Reason from procedure CPI-Xp (Explanations of Classifiers under Constraints): {reason4}")
print(f"CPI-Xp features: {features_original}")
print("CPI-Xp rule size:", len(explainer.to_features(reason4)))
print(f"Is a valid CPI-Xp reason: {explainer.is_reason(reason4)}")
print("Is sufficient_reason CPI-Xp:", explainer.is_sufficient_reason(reason4, n_samples=1000))

print("######################################################################")

print("binarized instance", explainer.binary_representation)
print("instance", instance)
# On accède directement à l'objet boosted_trees interne
print("theory", explainer._boosted_trees.get_theory(explainer._binary_representation))
print("######################")

print("Columns:", learner.get_details()[0]['feature_names'])
