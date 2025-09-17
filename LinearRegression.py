import numpy as np

class LinearRegression:

    def __init__(self,size):
        self.coeff = np.zeros(size)
        self.intercept = np.array(0,dtype=float)

    def fit(self, X, y):
        for i in range(100):
            y_pred = X@self.coeff + self.intercept
            coeff_grad = X.T@(2*(y_pred - y))/len(X)
            intercept_grad = 2*(y_pred - y).sum()/len(X)
            self.coeff -= 0.01*coeff_grad
            self.intercept -= 0.01*intercept_grad

    def predict(self, X):
        return np.dot(X, self.coeff) + self.intercept

X = np.array([[1,2],[3,4],[5,6],[7,8]])
y = np.array([1,2,3,4])

model = LinearRegression(2)
model.fit(X,y)
for i in range(4):
    print(model.predict(X[i]))