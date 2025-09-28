import numpy as np

class LinearRegression:

    def __init__(self):
        self.coeff = None
        self.intercept = np.array(0,dtype=float)

    def fit(self, X, y, lr = 0.1, method = "GradientDescent"):
        self.coeff = np.zeros(X.shape[1])
        if X.shape[0] != len(y):
            raise ValueError("X and y must have same number of data points")
        if method == "GradientDescent":
            for i in range(10000):
                y_pred = X@self.coeff + self.intercept
                coeff_grad = X.T@(2*(y_pred - y))/len(X)
                intercept_grad = 2*(y_pred - y).sum()/len(X)
                self.coeff -= lr*coeff_grad
                self.intercept -= lr*intercept_grad
        elif method == "LeastSquares":
            X_with_ones = np.zeros((X.shape[0],X.shape[1]+1))
            X_with_ones[:,1:] = X
            X_with_ones[:,0] = np.ones(X.shape[0])
            solution = np.linalg.inv(X_with_ones.T@X_with_ones)@X_with_ones.T@y
            self.intercept = solution[0]
            self.coeff = solution[1:]

    def predict(self, X):
        return np.dot(X, self.coeff) + self.intercept

import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    # ----- Generate noisy data (same as before) -----
    np.random.seed(42)
    n_samples = 50
    experience = np.random.randint(1, 11, size=n_samples)
    education = np.random.choice([0,1,2], size=n_samples)
    noise = np.random.normal(0, 5000, size=n_samples)

    y = 5000*experience + 10000*education + noise
    X = np.column_stack([experience, education])

    # ----- Train your model -----
    model = LinearRegression()
    model.fit(X, y, method="LeastSquares")

    # Predictions
    y_pred = model.predict(X)

    # ----- Plot y_true vs y_pred -----
    plt.figure(figsize=(7,7))
    plt.scatter(y, y_pred, color="blue", alpha=0.7, label="Predictions")
    plt.plot([y.min(), y.max()], [y.min(), y.max()], "r--", lw=2, label="y = y_pred")
    plt.xlabel("True Salary")
    plt.ylabel("Predicted Salary")
    plt.title("True vs Predicted Salaries")
    plt.legend()
    plt.grid(True)
    plt.show()