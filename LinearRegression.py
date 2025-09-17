import numpy as np

class LinearRegression:

    def __init__(self,size):
        self.coeff = np.zeros(size)
        self.intercept = np.array(0,dtype=float)

    def fit(self, X, y):
        for i in range(100):
            for j in range(len(X)):
                y_pred = np.dot(X[j], self.coeff) + self.intercept
                coeff_grad = 2*(y_pred - y[j])*X[j]
                intercept_grad = 2*(y[j] - y[j])
                self.coeff -= 0.01*coeff_grad
                self.intercept -= 0.01*intercept_grad

    def predict(self, X):
        return np.dot(X, self.coeff) + self.intercept

X = np.array([[1,2],[3,4],[5,6],[7,8]])
y = np.array([1,2,3,4,5])

model = LinearRegression(2)
model.fit(X,y)
for i in range(4):
    print(model.predict(X[i]))