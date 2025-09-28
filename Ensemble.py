from classification_tree import Tree

class RandomForestClassifier:

    def __init__(self, n_estimators, max_depth):
        self.n_estimators = n_estimators
        self.models = []
        self.max_depth = max_depth

    def fit(self,X,y):
        for i in range(self.n_estimators):
            self.models.append(Tree(max_depth=self.max_depth))
            data = X
            data['target'] = y
            data_length = len(data.index)
            data_sampled = data.sample(int(data_length*0.667))

