import numpy as np

class LinearRegression:

    def __init__(self,size):
        self.coeff = np.zeros(size)
        self.intercept = np.array(0,dtype=float)

    def fit(self, X, y):
        for i in range(10000):
            y_pred = X@self.coeff + self.intercept
            coeff_grad = X.T@(2*(y_pred - y))/len(X)
            intercept_grad = 2*(y_pred - y).sum()/len(X)
            self.coeff -= 0.01*coeff_grad
            self.intercept -= 0.01*intercept_grad

    def predict(self, X):
        return np.dot(X, self.coeff) + self.intercept

import numpy as np
import matplotlib.pyplot as plt

# ----- Generate noisy data (same as before) -----
np.random.seed(42)
n_samples = 50
experience = np.random.randint(1, 11, size=n_samples)
education = np.random.choice([0,1,2], size=n_samples)
noise = np.random.normal(0, 5000, size=n_samples)

y = 5000*experience + 10000*education + noise
X = np.column_stack([experience, education])

# ----- Train your model -----
model = LinearRegression(size=2)
model.fit(X, y)

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