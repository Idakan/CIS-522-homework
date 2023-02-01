import numpy as np


class LinearRegression:
    """
    Linear Regression model that solves for weights using closed form approach
    """

    # w: np.ndarray
    # b: float

    def __init__(self):
        """ "
        Initialize weights and biases.
        """
        self.b = 0
        self.w = np.ndarray(1)
        self.losses = []

    # raise NotImplementedError()

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """ "
        Solve for parameters analytically, using closed-form solution
        derived in lecture
        """

        # Get optimal weights and bias term
        self.w = np.matmul(np.matmul(np.linalg.inv(np.matmul(X.T, X)), X.T), y)
        self.b = 0

        # We are not returning anything; just updating our weights
        return None

    def predict(self, X: np.ndarray) -> np.ndarray:
        """ "
        Using optimized weights and biases, calculate y_pred
        """
        print(self.w.shape)
        print(X.shape)
        # We are returning our predicted value/label using our optimized weight set
        return np.matmul(self.w.T, X.T) + self.b


class GradientDescentLinearRegression(LinearRegression):
    """
    A linear regression model that uses gradient descent to fit the model.
    """

    def fit(
        self, X: np.ndarray, y: np.ndarray, lr: float = 0.0000001, epochs: int = 50000
    ) -> None:

        """
        Over several iterations, adjust the weights and biases appropriately
        towards their most optimal values using gradient descent
        """
        # raise NotImplementedError()

        num_samples, num_features = X.shape
        self.w = np.zeros((num_features, 1))
        self.b = 0
        y = y.reshape(num_samples, 1)

        for i in range(epochs):
            y_hat = np.matmul(X, self.w) + self.b

            dL_dw = 0
            dL_db = 0

            for j in range(num_features):
                dL_db += y_hat[j] - y[j] / num_features
                dL_dw += (y_hat[j] - y[j]) * X[j] / num_features
            dL_dw = dL_dw.reshape(num_features, 1)

            self.w -= lr * dL_dw
            self.b -= -(lr * dL_db)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict the output for the given input.
        Arguments:
            X (np.ndarray): The input data.
        Returns:
            np.ndarray: The predicted output.
        """
        return np.matmul(X, self.w) + self.b
