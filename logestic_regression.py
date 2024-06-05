import numpy as np

class LogisticRegression:
    def __init__(self, learning_rate: float = 0.01, n_iterations: int = 1000):
        """
        Initialize the logistic regression model.

        Args:
            learning_rate (float): The learning rate for gradient descent.
            n_iterations (int): The number of iterations for gradient descent.
        """
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights: np.ndarray | None = None
        self.bias: float | None = None
    def _compute_loss(self) -> float:
        """
        Compute the binary cross-entropy loss.

        Returns:
            float: The binary cross-entropy loss.
        """
        y_predicted = self._sigmoid(np.dot(self.X, self.weights) + self.bias)
        loss = -(1 / self.n_samples) * (
            np.dot(self.y.T, np.log(y_predicted)) +
            np.dot((1 - self.y).T, np.log(1 - y_predicted))
        )
        return loss

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Fit the logistic regression model to the training data.

        Args:
            X (np.ndarray): The training features.
            y (np.ndarray): The target values (0 or 1).
            
         Returns:
            float: The final loss after training.
        """
        
        self.X = X
        self.y = y
        self.n_samples, n_features = X.shape

        # Initialize weights and bias
        self.weights = np.zeros(n_features)
        self.bias = 0

        # Gradient descent
        for _ in range(self.n_iterations):
            self._gradient_descent()
        return self._compute_loss()

    def _gradient_descent(self) -> None:
        """
        Perform gradient descent to update the weights and bias.
        """
        y_predicted = self._sigmoid(np.dot(self.X, self.weights) + self.bias)

        # Compute gradients
        dw = (1 / self.n_samples) * np.dot(self.X.T, (y_predicted - self.y))
        db = (1 / self.n_samples) * np.sum(y_predicted - self.y)

        # Update weights and bias
        self.weights -= self.learning_rate * dw
        self.bias -= self.learning_rate * db

    def _sigmoid(self, z: np.ndarray) -> np.ndarray:
        """
        Compute the sigmoid function.

        Args:
            z (np.ndarray): The input array.

        Returns:
            np.ndarray: The sigmoid of the input array.
        """
        return 1 / (1 + np.exp(-z))

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict the probabilities of the target values for new input data.

        Args:
            X (np.ndarray): The input features.

        Returns:
            np.ndarray: The predicted probabilities of the target values.
        """
        y_predicted = self._sigmoid(np.dot(X, self.weights) + self.bias)
        y_predicted_labels = (y_predicted > 0.5).astype(int)
        return y_predicted_labels