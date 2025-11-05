import numpy as np

def mean_squared_error(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

def log_mean_squared_error(y_true, y_pred):
    return np.mean((np.log(y_true) - np.log(y_pred)) ** 2)

def accuracy(y_true, y_pred):
    return np.mean(y_true == y_pred)