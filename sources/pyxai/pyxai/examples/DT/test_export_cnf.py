from pyxai import Builder, Explainer

node_x4_1 = Builder.DecisionNode(4, left=0, right=1)
node_x4_2 = Builder.DecisionNode(4, left=0, right=1)
node_x4_3 = Builder.DecisionNode(4, left=0, right=1)
node_x4_4 = Builder.DecisionNode(4, left=0, right=1)
node_x4_5 = Builder.DecisionNode(4, left=0, right=1)

node_x3_1 = Builder.DecisionNode(3, left=0, right=node_x4_1)
node_x3_2 = Builder.DecisionNode(3, left=node_x4_2, right=node_x4_3)
node_x3_3 = Builder.DecisionNode(3, left=node_x4_4, right=node_x4_5)

node_x2_1 = Builder.DecisionNode(2, left=0, right=node_x3_1)
node_x2_2 = Builder.DecisionNode(2, left=node_x3_2, right=node_x3_3)

node_x1_1 = Builder.DecisionNode(1, left=node_x2_1, right=node_x2_2)

tree = Builder.DecisionTree(4, node_x1_1, force_features_equal_to_binaries=True)

explainer = Explainer.initialize(tree)
explainer.set_instance((1, 1, 1, 1))

print(explainer.to_CNF())
