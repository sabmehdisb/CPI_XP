from pyxai import Learning, Explainer,Tools
import pandas as pd
import itertools
import pandas as pd
dataset_name=Tools.Options.dataset
df = pd.read_csv(dataset_name + '.csv')
learner = Learning.Scikitlearn(dataset_name+ '.csv', learner_type=Learning.CLASSIFICATION)
model = learner.evaluate(method=Learning.HOLD_OUT, output=Learning.RF,n_estimators=4)
instance, prediction = learner.get_instances(model, n=1, correct=True)
print("instancee",instance)
# instance=[0,0,0,1,0,0,0,0,1,0,0]
explainer = Explainer.initialize(model, instance,features_type=dataset_name + '.types')

reason =explainer.sufficient_reason()
# ordre_features=[ 'mcv', 'sgpt', 'sgot', 'alkphos', 'gammagt']
ordre_features=[]
# ordre_features=[ 'mcv']
#ordre_features=['Right-Distance', 'Left-Weight', 'Left-Distance', 'Right-Weight']
# reason2 =explainer.sufficient_general_reason_pure_sat(n=2)
#print("théorie", explainer.get_theory())
print("instance binarisé", explainer.binary_representation)
reason2 =explainer.sufficient_reason_single_strategyy( n=1, strategy="priority_order", random_seed=None,ordre_features=ordre_features)
print("###############################################################""")
features_original =explainer.to_features(reason)
print(f"Raison originale: {reason}")
print(f"Features: {features_original}")
print(f"Est une raison valide: {explainer.is_reason(reason)}")
print("is sufficient_reason: ", explainer.is_sufficient_reason(reason,n_samples=1000))
print("######################################################################")
# Trouver les règles de taille minimale
# rules_with_sizes = [(i, len(explainer.to_features(i))) for i in reason2[0]]
# min_size = min(size for _, size in rules_with_sizes)
# smallest_rules = [rule for rule, size in rules_with_sizes if size == min_size]
# results = explainer.compare_strategies(n=1)
for i in reason2:
# for i in smallest_rules:
    print("i",i)
    features_original =explainer.to_features(i)
    print(f"Raison CPI_X originale2: {i}")
    print(f"CPI_XP features: {features_original}")
    print("taille de la regles:",len(explainer.to_features(i)))
    print(f"Est une raison valide2: {explainer.is_reason(i)}")
    print("is sufficient_reason2: ", explainer.is_sufficient_reason(i,n_samples=1000))

print("######################################################################")
print("instance binarisé", explainer.binary_representation)
print("instance", instance)
print("théorie", explainer.get_theory())
print("######################")
#print("cnf",explainer.to_CNF())
# print("nombre de regles",len(reason2[0]))
# print("nombre de combinaison possible de features",reason2[2])
# print("nombre de régles répété avec les différente combinaisont de feature",reason2[1])
# print("nombre de regles petite",smallest_rules)
# # Afficher toutes les colonnes (y compris la cible)
# print("Toutes les colonnes :", df.columns.tolist())
colonnes_sans_cible = df.columns[:-1]  # Toutes les colonnes sauf la dernière
print("Colonnes  :",learner.get_details()[0]['feature_names'])





# from pyxai import Learning, Explainer,Tools
# import pandas as pd
# import itertools
# import pandas as pd
# dataset_name=Tools.Options.dataset
# df = pd.read_csv(dataset_name + '.csv')
# learner = Learning.Scikitlearn(dataset_name+ '.csv', learner_type=Learning.CLASSIFICATION)
# model = learner.evaluate(method=Learning.HOLD_OUT, output=Learning.DT)
# instance, prediction = learner.get_instances(model, n=1, correct=True)
# explainer = Explainer.initialize(model, instance,features_type=dataset_name + '.types')

# reason =explainer.sufficient_reason(n=all)
# # reason2 =explainer.sufficient_general_reason_pure_sat(n=2)
# reason2 =explainer.sufficient_reason_all_priority_orders(n="ALL")
# print("#################")
# for i in reason:
#     print("############################")
#     features_original =explainer.to_features(i)
#     print(f"Raison originale: {i}")
#     print(f"Features: {features_original}")
#     print(f"Est une raison valide: {explainer.is_reason(i)}")
#     print("is sufficient_reason: ", explainer.is_sufficient_reason(i))
# print("########################################################################")
# # Trouver les règles de taille minimale
# rules_with_sizes = [(i, len(explainer.to_features(i))) for i in reason2[0]]
# min_size = min(size for _, size in rules_with_sizes)
# smallest_rules = [rule for rule, size in rules_with_sizes if size == min_size]
# # results = explainer.compare_strategies(n=1)
# for i in reason2[0]:
# # for i in smallest_rules:
#     print(i)
#     features_original =explainer.to_features(i)
#     print(f"Raison originale2: {i}")
#     print(f"Features2: {features_original}")
#     print("taille de la regles:",len(explainer.to_features(i)))
#     print(f"Est une raison valide2: {explainer.is_reason(i)}")
#     print("is sufficient_reason2: ", explainer.is_sufficient_reason(i))

# print("########################################################################")
# print("instance binarisé", explainer.binary_representation)
# print("instance", instance)
# print("théorie", explainer.get_theory())
# print("nombre de regles1",len(reason))
# print("nombre de regles2",len(reason2[0]))
# print("nombre de combinaison possible de features",reason2[2])
# print("nombre de régles répété avec les différente combinaisont de feature",reason2[1])
# print(" les regles les plus générale",smallest_rules)
# for i in smallest_rules:
#     print("les regles les plus générale en features",explainer.to_features(i))
# # # Afficher toutes les colonnes (y compris la cible)
# # print("Toutes les colonnes :", df.columns.tolist())
# colonnes_sans_cible = df.columns[:-1]  # Toutes les colonnes sauf la dernière
# print("Colonnes sans la cible :", colonnes_sans_cible)
