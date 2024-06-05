import numpy as np

class SimpleLinearRegression:
    def __init__(self, n_iter=1000,learning_rate = 0.01):
        self.W = 0
        self.b = 0
        self.n_iter = n_iter
        self.learning_rate = learning_rate

    def fit(self,X,y):
        self.X = np.array(X)
        self.y = np.array(y)
        if not len(X) == len(y):
            raise ValueError("X and y must have the same length.") 
        for _ in range(self.n_iter):
            self.gradient_descent()
        return self.compute_cost()

    def compute_cost(self):
        m = len(self.y)
        predictions = self.W * self.X + self.b
        cost = np.sum((self.y - predictions) ** 2) / (2 * m)
        return cost

    def gradient_descent(self):
        m = len(self.y)
        predictions = self.W * self.X + self.b
        self.W -= self.learning_rate * np.sum((predictions - self.y) * self.X) / m
        self.b -=  self.learning_rate * np.sum(predictions - self.y) / m


    def predict(self, X_test):
        X_test = np.array(X_test)
        test_pred = []
        for i in range(len(X_test)):
            test_pred.append(self.W * X_test[i] + self.b)
        return test_pred
