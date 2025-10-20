import pandas as pd
from classification_tree import Tree

class RandomForestClassifier:

    def __init__(self, n_estimators, max_depth):
        self.n_estimators = n_estimators
        self.models = []
        self.max_depth = max_depth

    def fit(self,X,y):
        self.models = []
        for i in range(self.n_estimators):
            self.models.append(Tree(max_depth=self.max_depth))
            data = X.copy()
            data['target'] = y
            data_length = len(data.index)
            data_sampled = data.sample(int(data_length*0.667))
            print(data_sampled)
            self.models[i].fit(data_sampled.drop('target',axis=1), data_sampled['target'])

    def predict(self,X):
        predictions = []
        for i in self.models:
            predictions.append(i.predict(X))
        return pd.Series(predictions).mode()

data = {
    0: ["Sunny", "Sunny", "Overcast", "Rain", "Rain", "Rain", "Overcast", "Sunny", "Sunny", "Rain", "Sunny", "Overcast",
        "Overcast", "Rain"],
    1: ["Hot", "Hot", "Hot", "Mild", "Cool", "Cool", "Cool", "Mild", "Cool", "Mild", "Mild", "Mild", "Hot", "Mild"],
    2: ["High", "High", "High", "High", "Normal", "Normal", "Normal", "High", "Normal", "Normal", "Normal", "High",
        "Normal", "High"],
    3: ["Weak", "Strong", "Weak", "Weak", "Weak", "Strong", "Strong", "Weak", "Weak", "Weak", "Strong", "Strong", "Weak",
        "Strong"]
}

if __name__ == '__main__':
    # Target
    target = ["No", "No", "Yes", "Yes", "Yes", "No", "Yes", "No", "Yes", "Yes", "Yes", "Yes", "Yes", "No"]

    # Convert to DataFrame
    X = pd.DataFrame(data)
    y = pd.Series(target, name="Play")

    tree = RandomForestClassifier(n_estimators = 200,max_depth=10)
    tree.fit(X, y)
    for i in range(14):
        print(tree.predict(X.loc[i]))