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
        feature = pd.DataFrame(self.feature).reset_index(drop = True)
        target = pd.DataFrame(self.target.squeeze()).reset_index(drop = True)
        gini_impurities = []
        splits = []
        self.feature = self.feature[[col for col in feature.columns if feature[col].nunique() > 1]]
        for i in range(self.feature.shape[1]):
            take_input_tuple = gini_impurity(self.feature.iloc[:, i].squeeze(), self.target.squeeze())
            gini_impurities.append(take_input_tuple[0])
            splits.append(take_input_tuple[1])
        split_index = gini_impurities.index(min(gini_impurities))
        split = splits[split_index]
        self.split = split
        self.split_feature_index = split_index
        mask_left = feature[split_index].isin(split[0])
        mask_right = feature[split_index].isin(split[1])
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
            self.left_child = Node(feature[mask_left].reset_index(drop = True),
                                   target[mask_left].reset_index(drop = True),
                                   self.max_depth, self.depth + 1)
            self.right_child = Node(feature[mask_right].reset_index(drop = True),
                                    target[mask_right].reset_index(drop = True),
                                    self.max_depth, self.depth + 1)
            self.left_child.make_children()
            self.right_child.make_children()


    def forward(self,features):
        if self.left_child is None or self.right_child is None:
            raise RuntimeError('You must call make_children first')
        if features[self.split_feature_index] in self.split[0]:
            return self.left_child.forward(features)
        if features[self.split_feature_index] in self.split[1]:
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

class Tree:

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

if __name__ == '__main__':
    data = {
        0: ["Sunny", "Sunny", "Overcast", "Rain", "Rain", "Rain", "Overcast", "Sunny", "Sunny", "Rain", "Sunny", "Overcast", "Overcast", "Rain"],
        1: ["Hot", "Hot", "Hot", "Mild", "Cool", "Cool", "Cool", "Mild", "Cool", "Mild", "Mild", "Mild", "Hot", "Mild"],
        2: ["High", "High", "High", "High", "Normal", "Normal", "Normal", "High", "Normal", "Normal", "Normal", "High", "Normal", "High"],
        3: ["Weak", "Strong", "Weak", "Weak", "Weak", "Strong", "Strong", "Weak", "Weak", "Weak", "Strong", "Strong", "Weak", "Strong"]
    }

    # Target
    target = ["No", "No", "Yes", "Yes", "Yes", "No", "Yes", "No", "Yes", "Yes", "Yes", "Yes", "Yes", "No"]
    # Convert to DataFrame
    X = pd.DataFrame(data)
    y = pd.Series(target, name="Play")
    tree = Tree(max_depth=10)
    tree.fit(X, y)
    for i in range(14):
        print(tree.predict(X.loc[i]))