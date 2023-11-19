import numpy as np
import autograd.numpy as ag_np
from autograd import grad
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

def sigmoid(z):
    """
    Calculate the sigmoid function.

    Parameters:
    - z (array-like): Input values.

    Returns:
    - array-like: Sigmoid of the input values.
    """
    return 1.0 / (1.0 + np.exp(-z))

def logistic_loss(y, y_pred):
    """
    Calculate the logistic loss.

    Parameters:
    - y (array-like): True labels.
    - y_pred (array-like): Predicted probabilities.

    Returns:
    - float: Logistic loss.
    """
    return -np.mean(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred))

def logistic_regression(X, y, learning_rate, batch_size=32, Niterations=1500, test_size=0.2):
    """
    Perform logistic regression using plain stochastic gradient descent.

    Parameters:
    - X (array-like): Input features.
    - y (array-like): True labels.
    - learning_rate (list): List of learning rates to try.
    - batch_size (int): Size of mini-batch for SGD.
    - Niterations (int): Number of iterations.
    - test_size (float): Fraction of the data to use for testing.

    Returns:
    - tuple: Best learning rate and corresponding beta.
    """
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    cost_values = []
    best_cost = float('inf')
    best_learning_rate = 0
    best_beta = None

    def predict_proba(X, beta):
        """
        Predict probabilities using the logistic function.

        Parameters:
        - X (array-like): Input features.
        - beta (array-like): Coefficients.

        Returns:
        - array-like: Predicted probabilities.
        """
        return sigmoid(X @ beta)

    def gradient(X, y, beta):
        """
        Compute the gradient of the logistic loss.

        Parameters:
        - X (array-like): Input features.
        - y (array-like): True labels.
        - beta (array-like): Coefficients.

        Returns:
        - array-like: Gradient of the logistic loss.
        """
        # Compute the gradient of the logistic loss with respect to the coefficients (beta)
        y_pred = predict_proba(X, beta)
        error = y_pred - y
        return (1.0 / X.shape[0]) * X.T @ error # # Compute the gradient of the logistic loss

    for eta in learning_rate:
        # Initialize random weights for each learning rate
        beta = np.random.randn(X.shape[1])

        cost_values_eta = []

        for epoch in range(Niterations):
            # Shuffle the training data to introduce randomness
            indices = np.random.permutation(X_train.shape[0])
            X_shuffled = X_train[indices]
            y_shuffled = y_train[indices]

            for i in range(0, X_train.shape[0], batch_size):
                X_mini_batch = X_shuffled[i:i+batch_size]
                y_mini_batch = y_shuffled[i:i+batch_size]

                # Compute the gradient using the current mini-batch
                grad_mini_batch = gradient(X_mini_batch, y_mini_batch, beta)
                # Update coefficients using gradient descent
                beta -= eta * grad_mini_batch

            y_pred = predict_proba(X_test, beta)
            cost = logistic_loss(y_test, y_pred)
            cost_values_eta.append(cost)

        cost_values.append(cost_values_eta)

        if min(cost_values_eta) < best_cost:
            best_cost = min(cost_values_eta)
            best_learning_rate = eta
            best_beta = beta
            
    # Compare with scikit-learn Logistic Regression
    clf = LogisticRegression(solver='lbfgs')
    clf.fit(X_train, y_train)
    y_pred_sklearn = clf.predict_proba(X_test)[:, 1]
    cost_sklearn = logistic_loss(y_test, y_pred_sklearn)
    print(f"Lowest cost score scikit-learn Logistic Regression {cost_sklearn}")

    # Print the best result from plain SGD
    print(f"Lowest cost score plain SGD {min(cost_values[learning_rate.index(best_learning_rate)])}, eta = {best_learning_rate}")

    # Plot the logistic loss over epochs for the best learning rate
    sns.set_style("darkgrid")
    plt.figure(figsize=(10, 6))
    plt.plot(range(Niterations), cost_values[learning_rate.index(best_learning_rate)], label=f'Logistic Loss (Learning Rate = {best_learning_rate})')
    plt.title(f'Plain SGD - Lowest Logistic Loss vs. Epochs (Learning Rate = {best_learning_rate})')
    plt.xlabel('Epochs')
    plt.ylabel('Logistic Loss')
    plt.legend()
    
    
    plt.show()

    return best_learning_rate, best_beta

# Set seed for reproducibility
np.random.seed(31)

#Input data
n = 100
x = np.random.rand(n, 1)
y = (x > 0.5).astype(int).flatten() # Convert values in array 'x' to binary (0 or 1) based on the condition (x > 0.5)
X = np.c_[np.ones((n, 1)), x]

# Define learning rates to try
learning_rates = [0.001, 0.01, 0.1]


logistic_regression(X, y, learning_rates)
