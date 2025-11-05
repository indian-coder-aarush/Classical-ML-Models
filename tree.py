import numpy as np
import pandas as pd
import itertools

def gini_impurity(feature,target):
    df = pd.DataFrame({'feature':feature,'target':target})
    df.sort_values(by='feature',inplace = True)
    table = pd.crosstab(df['target'],df['feature'])
    if len(df['feature'].unique()) == 1:
        return 0, [list(df['feature'].unique()), []]
    combinations = []
    for i in range(1,len(df.feature.unique().tolist())):
        for combination in itertools.combinations(df.feature.unique().tolist(),i):
            combinations.append([list(combination),[x for x in df.feature.unique().tolist() if x not in combination]])
    gini_impurities = {}
    for combination in combinations:
        right = combination[0]
        left = combination[1]
        left_total = table.loc[:, left].sum().sum()
        right_total = table.loc[:, right].sum().sum()
        left_class_frequencies = table.loc[:, left]
        right_class_frequencies = table.loc[:, right]
        left_impurity = 1 - ((left_class_frequencies/left_total)**2).sum().sum()
        right_impurity = 1 - ((right_class_frequencies/right_total)**2).sum().sum()
        total_impurity = left_impurity*(left_total/(right_total+left_total)) + right_impurity*(right_total/
                                                                                               (right_total+left_total))
        gini_impurities[(tuple(left),tuple(right))] = total_impurity
    least_impurity_key = min(gini_impurities, key=gini_impurities.get)
    return gini_impurities[least_impurity_key] , [list(least_impurity_key[0]),list(least_impurity_key[1])]

class Node:

    def __init__(self,feature,target,max_depth,depth):
        self.feature = feature
        self.target = target
        self.max_depth = max_depth
        self.depth = depth
        self.left_child = None
        self.right_child = None
        self.split = None
        self.split_feature_index = None

    def make_children(self):
        feature = pd.DataFrame(self.feature)
        target = pd.DataFrame(self.target)
        gini_impurities = []
        splits = []
        valid_cols = [col for col in feature.columns if feature[col].nunique() > 1]
        for col in valid_cols:
            take_input_tuple = gini_impurity(feature[col].squeeze(), target.squeeze())
            gini_impurities.append(take_input_tuple[0])
            splits.append(take_input_tuple[1])
        if len(gini_impurities) == 0:
            self.left_child = LeafNode(self.target.squeeze())
            self.right_child = LeafNode(self.target.squeeze())
            self.left_child.calculate_best_label()
            self.right_child.calculate_best_label()
            return
        if len(gini_impurities) == 1:
            split_index = 0
        else:
            split_index = gini_impurities.index(min(gini_impurities))
        split = splits[split_index]
        self.split = split
        self.split_feature_index = valid_cols[split_index]
        mask_left = feature[self.split_feature_index].isin(split[0])
        mask_right = feature[self.split_feature_index].isin(split[1])
        if mask_left.sum() == 0 or mask_right.sum() == 0:
            self.left_child = LeafNode(self.target.squeeze())
            self.right_child = LeafNode(self.target.squeeze())
            self.left_child.calculate_best_label()
            self.right_child.calculate_best_label()
            return
        if self.depth == self.max_depth-1 or min(gini_impurities)==0:
            self.left_child = LeafNode(target[mask_left].squeeze())
            self.right_child = LeafNode(target[mask_right].squeeze())
            self.left_child.calculate_best_label()
            self.right_child.calculate_best_label()
            return
        else:
            self.left_child = Node(feature[mask_left],
                                   target[mask_left],
                                   self.max_depth, self.depth + 1)
            self.right_child = Node(feature[mask_right],
                                    target[mask_right],
                                    self.max_depth, self.depth + 1)
            self.left_child.make_children()
            self.right_child.make_children()


    def forward(self,features):
        if self.left_child is None or self.right_child is None:
            raise RuntimeError('You must call make_children first')
        if (self.right_child.__class__.__name__ == "LeafNode" and
                self.right_child.predicted_label == self.left_child.predicted_label):
            return self.right_child.predicted_label
        value = features[self.split_feature_index]
        if value in self.split[0]:
            return self.left_child.forward(features)
        elif value in self.split[1]:
            return self.right_child.forward(features)
        if len(self.left_child.target) >= len(self.right_child.target):
            return self.left_child.forward(features)
        else:
            return self.right_child.forward(features)

class LeafNode:

    def __init__(self,target):
        self.target = target
        self.predicted_label = None

    def calculate_best_label(self):
        unique_elements, counts = np.unique(self.target, return_counts=True)
        highest_value_element_index = np.argmax(counts)
        self.predicted_label = unique_elements[highest_value_element_index]

    def forward(self,features):
        if self.predicted_label is not None:
            return self.predicted_label
        else:
            raise RuntimeError('You must call calculate_best_label first')

class DecisionTreeClassifier:

    def __init__(self,max_depth):
        self.max_depth = max_depth
        self.node = None

    def fit(self,features,target):
        self.node = Node(features,target,self.max_depth,0)
        self.node.make_children()

    def predict(self,features):
        if self.node is None:
            raise RuntimeError('You must call fit first')
        result = self.node.forward(features)
        return result