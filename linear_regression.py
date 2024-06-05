import numpy as np

class LinearRegression:
    def __init__(self, learning_rate: float = 0.01, n_iterations: int = 1000):
        """
        Initialize the linear regression model.

        Args:
            learning_rate (float): The learning rate for gradient descent.
            n_iterations (int): The number of iterations for gradient descent.
        """
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights: np.ndarray | None = None
        self.bias: float | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Fit the linear regression model to the training data.

        Args:
            X (np.ndarray): The training features.
            y (np.ndarray): The target values.

        Returns:
            float: The final cost after training.
        """
        self.X = X
        self.y = y
        self.n_samples, n_features = X.shape

        # Initialize weights and bias
        self.weights = np.zeros((n_features, 1))
        self.bias = 0

        if self.X.shape[0] != self.y.shape[0]:
            raise ValueError("X and y must have the same length.")

        for _ in range(self.n_iterations):
            self._gradient_descent()

        return self._compute_cost()

    def _gradient_descent(self) -> None:
        """
        Perform gradient descent to update the weights and bias.
        """
        y_predicted = np.dot(self.X, self.weights) + self.bias

        # Compute gradients
        dw = (1 / self.n_samples) * np.dot(self.X.T, (y_predicted - self.y))
        db = (1 / self.n_samples) * np.sum(y_predicted - self.y)

        # Update weights and bias
        self.weights -= self.learning_rate * dw
        self.bias -= self.learning_rate * db

    def _compute_cost(self) -> float:
        """
        Compute the cost function (mean squared error).

        Returns:
            float: The mean squared error.
        """
        predictions = np.dot(self.X, self.weights) + self.bias
        cost = np.sum((self.y - predictions) ** 2) / (2 * self.n_samples)
        return cost

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict the target values for new input data.

        Args:
            X (np.ndarray): The input features.

        Returns:
            np.ndarray: The predicted target values.
        """
        return np.dot(X, self.weights) + self.bias