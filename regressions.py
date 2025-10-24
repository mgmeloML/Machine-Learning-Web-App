# Import necessary libraries
import numpy as np

# Base class for the regression models
class Regression:
    # Takes in a list of coefficients and returns a string in LaTeX format representing the polynomial function
    def series(self, coeffs):
        n = len(coeffs)
        res = '$ f(x) = '
        for i in range(len(coeffs)):
            if n - i - 1 == 1:
                res += str(coeffs[i]) + "x+"
                if coeffs[i + 1] < 0:
                    res = res[:-1]
            elif n - i - 1 == 0:
                res += str(coeffs[i])
            else:
                res += str(coeffs[i]) + "x^{" + str(n - i - 1) + "}+"
                if coeffs[i + 1] < 0:
                    res = res[:-1]
        return res + "$"

# Subclass of Regression implementing linear regression
class LinearRegression(Regression):

    def __init__(self, learning_rate, epoch):
        # Initialize parameters
        self.learning_rate = learning_rate
        self.epoch = epoch
        self.m = 0 # gradient of the linear regression function
        self.c = 0 # constant term of the linear regression function

    def fit(self, X, y):
        m_grad = 0
        c_grad = 0

        n = len(X)

        # Loop through the specified number of epochs
        for iter in range(self.epoch):
            # Create a vector which contains values along the function f(x) = mx + c
            y_pred = np.dot(X, self.m) + self.c

            # Differentiate the loss function with respect to both m and c and calculate their gradients
            m_grad = (1 / n) * np.dot(X.T, (y_pred - y))
            c_grad = (1 / n) * np.sum(y_pred - y)

            # Update m and c using gradient descent
            self.m = self.m - self.learning_rate * m_grad
            self.c = self.c - self.learning_rate * c_grad

        return self.m, self.c

    # Uses the series method in the base class to return a LaTeX string representing the linear regression function
    def series(self, coeffs):
        return super().series(coeffs)

    # Predict values using the linear regression function
    def predict(self, X):
        return [num * self.m + self.c for num in X]

# Subclass of Regression implementing polynomial regression
class PolynomialRegression(Regression):

    def __init__(self, degree):
        self.degree = degree

    # Uses numpy's polyfit method to return a line of best fit of the specified degree
    def fit(self, X, Y):
        return np.polyfit(X, Y, self.degree)

    # Uses the series method in the base class to return a LaTeX string representing the polynomial regression function
    def series(self, coeffs):
        return super().series(coeffs)

    # Takes in a string that represents a mathematical function (something like f(x) = 3x + 2) and a value for x,
    # converts the string to a Python expression, and evaluates it with the given x value
    def evaluation(self, string):
        # Remove unnecessary text from the string and replace ^ with **
        string = string[9:-1]
        string = string.replace ("^", "**")
        string = string.replace ("{", "")
        string = string.replace ("}", "")
        # Replace 'x' with '*x' to properly multiply variables
        string = string.replace ("x", "*x")
        # Evaluate the expression with the given x value
        return string

    # Predicts values using the polynomial regression function
    def predict(self, function, X):
        f = lambda x: eval(function)
        return [f(num) for num in X]
