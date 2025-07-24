# linear_regression_numpy.py

import numpy as np

# Sample data
X = np.array([1, 2, 3, 4, 5])
y = np.array([2, 4, 6, 8, 10])

# Add bias (intercept term)
X_b = np.c_[np.ones((X.shape[0], 1)), X]

# Closed-form solution: theta = (X.T * X)^-1 * X.T * y
theta = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)

print("Learned coefficients:", theta)
