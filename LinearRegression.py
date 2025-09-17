import numpy as np
import pandas as pd

class LinearRegression:

    def __init__(self,size):
        self.coeff = np.zeros(size)
        self.intercept = np.array(0)

    def fit(self, X, y):
        for i in range(len(100)):
            for j in range(len(X)):
                y_pred = np.dot(X[j], self.coeff) + self.intercept
                coeff_grad = 2*(y_pred - y[j])*self.coeff
                intercept_grad = 2*(y[j] - self.intercept)
                self.coeff -= 0.01*coeff_grad
                self.intercept_grad = 0.01*intercept_grad