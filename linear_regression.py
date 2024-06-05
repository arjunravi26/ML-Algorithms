import numpy as np
from sklearn.metrics import r2_score

class LinearRegression:
    def __init__(self, learning_rate=0.01, n_iterations=1000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        self.X = np.array(X)
        self.y = np.array(y)
        self.n_samples, n_features = X.shape

      # Initialize weights and bias
        self.weights = np.zeros((n_features, 1))

        self.bias = 0        
        if not self.X.shape[0] == self.y.shape[0]:
            raise ValueError("X and y must have the same length.") 
        for _ in range(self.n_iterations):
            self.gradient_descent()
        return self.compute_cost()

    def gradient_descent(self):
        # Gradient descent
        for _ in range(self.n_iterations):
            y_predicted = np.dot(self.X, self.weights) + self.bias

            # Compute gradients
            dw = (1 / self.n_samples) * np.dot(self.X.T, (y_predicted - self.y))
            db = (1 / self.n_samples) * np.sum(y_predicted - self.y)

            # Update weights and bias
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
            
    def compute_cost(self):      
        predictions = np.dot(self.X, self.weights) + self.bias
        cost = np.sum((self.y - predictions) ** 2) / (2 * self.n_samples)
        return cost
    
    def predict(self, X):
        return np.dot(X, self.weights) + self.bias
