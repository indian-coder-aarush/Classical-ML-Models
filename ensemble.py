import pandas as pd
from tree import DecisionTreeClassifier

class RandomForestClassifier:

    def __init__(self, n_estimators, max_depth):
        self.n_estimators = n_estimators
        self.models = []
        self.max_depth = max_depth

    def fit(self,X,y):
        self.models = []
        for i in range(self.n_estimators):
            self.models.append(DecisionTreeClassifier(max_depth=self.max_depth))
            data = X.copy()
            data['target'] = y
            data_length = len(data.index)
            data_sampled = data.sample(int(data_length*0.667))
            self.models[i].fit(data_sampled.drop('target',axis=1), data_sampled['target'])

    def predict(self,X):
        predictions = []
        for i in self.models:
            predictions.append(i.predict(X))
        return pd.Series(predictions).mode()
