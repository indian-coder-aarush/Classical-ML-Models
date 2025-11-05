import numpy as np
import pandas as pd
import itertools


def gini_impurity(feature, target):
    df = pd.DataFrame({'feature': feature, 'target': target})
    df.sort_values(by='feature', inplace=True)
    table = pd.crosstab(df['target'], df['feature'])

    # only one unique category → no impurity
    if len(df['feature'].unique()) == 1:
        return 0, [list(df['feature'].unique()), []]

    combinations = []
    for i in range(1, len(df['feature'].unique())):
        for comb in itertools.combinations(df['feature'].unique(), i):
            left = list(comb)
            right = [x for x in df['feature'].unique() if x not in comb]
            combinations.append((left, right))

    gini_scores = {}
    for left, right in combinations:
        left_total = table.loc[:, left].sum().sum()
        right_total = table.loc[:, right].sum().sum()

        # skip invalid splits
        if left_total == 0 or right_total == 0:
            continue

        left_freq = table.loc[:, left]
        right_freq = table.loc[:, right]

        left_impurity = 1 - ((left_freq / left_total) ** 2).sum().sum()
        right_impurity = 1 - ((right_freq / right_total) ** 2).sum().sum()

        total_impurity = (
            left_impurity * (left_total / (left_total + right_total))
            + right_impurity * (right_total / (left_total + right_total))
        )
        gini_scores[(tuple(left), tuple(right))] = total_impurity

    if len(gini_scores) == 0:
        return 0, [list(df['feature'].unique()), []]

    best_split = min(gini_scores, key=gini_scores.get)
    return gini_scores[best_split], [list(best_split[0]), list(best_split[1])]


class LeafNode:
    def __init__(self, target):
        self.target = target
        self.predicted_label = None

    def calculate_best_label(self):
        unique, counts = np.unique(self.target, return_counts=True)
        self.predicted_label = unique[np.argmax(counts)]
        return self.predicted_label

    def forward(self, features):
        return self.predicted_label


class Node:
    def __init__(self, X, y, max_depth, depth):
        self.X = X
        self.y = y
        self.max_depth = max_depth
        self.depth = depth
        self.left_child = None
        self.right_child = None
        self.split = None
        self.split_feature_index = None

    def make_children(self):
        features = self.X.columns
        gini_scores = []
        splits = []

        # evaluate each feature for best split
        for feature in features:
            if self.X[feature].nunique() <= 1:
                continue
            impurity, split = gini_impurity(self.X[feature], self.y)
            gini_scores.append(impurity)
            splits.append((feature, split))

        # if no valid split found
        if len(gini_scores) == 0:
            self.split_feature_index = None
            self.left_child = LeafNode(self.y)
            self.right_child = LeafNode(self.y)
            self.left_child.calculate_best_label()
            self.right_child.calculate_best_label()
            return

        best_index = np.argmin(gini_scores)
        feature_name, split = splits[best_index]
        self.split_feature_index = feature_name
        self.split = split

        mask_left = self.X[feature_name].isin(split[0])
        mask_right = self.X[feature_name].isin(split[1])

        # if one side empty or max depth reached, make leaves
        if (
            mask_left.sum() == 0
            or mask_right.sum() == 0
            or self.depth == self.max_depth - 1
        ):
            self.left_child = LeafNode(self.y[mask_left])
            self.right_child = LeafNode(self.y[mask_right])
            self.left_child.calculate_best_label()
            self.right_child.calculate_best_label()
            return

        # recursion
        self.left_child = Node(
            self.X[mask_left].reset_index(drop=True),
            self.y[mask_left].reset_index(drop=True),
            self.max_depth,
            self.depth + 1,
        )
        self.right_child = Node(
            self.X[mask_right].reset_index(drop=True),
            self.y[mask_right].reset_index(drop=True),
            self.max_depth,
            self.depth + 1,
        )
        self.left_child.make_children()
        self.right_child.make_children()

    def forward(self, features):
        # if this is effectively a leaf node
        if self.split_feature_index is None:
            leaf = LeafNode(self.y)
            return leaf.calculate_best_label()

        value = features[self.split_feature_index]
        if value in self.split[0]:
            return self.left_child.forward(features)
        elif value in self.split[1]:
            return self.right_child.forward(features)
        else:
            # fallback if unseen value
            return self.right_child.forward(features)


class Tree:
    def __init__(self, max_depth=10):
        self.max_depth = max_depth
        self.root = None

    def fit(self, X, y):
        self.root = Node(X, y, self.max_depth, 0)
        self.root.make_children()

    def predict(self, x):
        return self.root.forward(x)


if __name__ == "__main__":
    # classic Play Tennis dataset
    data = {
        0: [
            "Sunny",
            "Sunny",
            "Overcast",
            "Rain",
            "Rain",
            "Rain",
            "Overcast",
            "Sunny",
            "Sunny",
            "Rain",
            "Sunny",
            "Overcast",
            "Overcast",
            "Rain",
        ],
        1: [
            "Hot",
            "Hot",
            "Hot",
            "Mild",
            "Cool",
            "Cool",
            "Cool",
            "Mild",
            "Cool",
            "Mild",
            "Mild",
            "Mild",
            "Hot",
            "Mild",
        ],
        2: [
            "High",
            "High",
            "High",
            "High",
            "Normal",
            "Normal",
            "Normal",
            "High",
            "Normal",
            "Normal",
            "Normal",
            "High",
            "Normal",
            "High",
        ],
        3: [
            "Weak",
            "Strong",
            "Weak",
            "Weak",
            "Weak",
            "Strong",
            "Strong",
            "Weak",
            "Weak",
            "Weak",
            "Strong",
            "Strong",
            "Weak",
            "Strong",
        ],
    }

    df = pd.DataFrame(data)
    target = pd.Series(
        [
            "No",
            "No",
            "Yes",
            "Yes",
            "Yes",
            "No",
            "Yes",
            "No",
            "Yes",
            "Yes",
            "Yes",
            "Yes",
            "Yes",
            "No",
        ],
        name="target",
    )

    tree = Tree(max_depth=10)
    tree.fit(df, target)

    preds = [tree.predict(df.iloc[i]) for i in range(len(df))]
    accuracy = np.mean(np.array(preds) == np.array(target))
    print(f"Accuracy: {accuracy * 100:.2f}%")
