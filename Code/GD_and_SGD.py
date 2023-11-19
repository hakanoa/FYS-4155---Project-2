import numpy as np
import autograd.numpy as ag_np
from autograd import grad
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_squared_error

def CostOLS(y, X, theta):
    """
    Calculates the ordinary least squares (OLS) cost function.

    Parameters:
    - y: Target variable
    - X: Feature matrix
    - theta: Model parameters

    Returns:
    - OLS cost
    """
    return ag_np.sum((y - X @ theta) ** 2)

def CostRidge(y, X, theta, lamda = 0.1):
    """
    Calculates the Ridge regression cost function.

    Parameters:
    - y: Target variable
    - X: Feature matrix
    - theta: Model parameters
    - lamda: Regularization parameter

    Returns:
    - Ridge regression cost
    """
    return ag_np.sum((y - X @ theta) ** 2) + lamda * ag_np.sum(theta[1:] ** 2)

def GD_plain(X, y, learning_rates, momentum=None, Niterations=500):
    """
    Performs plain gradient descent optimization.

    Parameters:
    - X: Feature matrix
    - y: Target variable
    - learning_rates: List of learning rates to be tested
    - momentum: If not None, uses momentum in the optimization
    - Niterations: Number of iterations

    Returns:
    - Best learning rate and best beta
    """
    cost_values = []  # To store MSE values at each learning rate
    best_cost = float('inf')  # Initialize with a large value
    best_learning_rate = 0
    best_beta = None

    for eta in learning_rates:
        beta = np.random.randn(X.shape[1], 1)
        if momentum:
            v = np.zeros_like(beta)

        cost_values_eta = []  # To store MSE values at each epoch for the current learning rate

        for iter in range(Niterations):
            # Calculate the MSE
            cost = CostOLS(y, X, beta) #can also add CostRidge
            cost_values_eta.append(cost)
            #gradient_ridge = 2 * X.T(X @ beta - y) + lambda *beta
            gradient = (2.0 / X.shape[0]) * X.T @ (X @ beta - y) #Can change from gradient with OLS to gradient_ridge

            if momentum:
                v = momentum * v + eta * gradient
                beta -= v
            else:
                beta -= eta * gradient

        cost_values.append(cost_values_eta)

        if min(cost_values_eta) < best_cost:
            best_cost = min(cost_values_eta)
            best_learning_rate = eta
            best_beta = beta
            
    if momentum is None:
        print(f"Lowest cost score plain GD no momentum {min(cost_values[learning_rates.index(best_learning_rate)])}, eta = {best_learning_rate}")
    else:
        print(f"Lowest cost score plain GD with momentum {min(cost_values[learning_rates.index(best_learning_rate)])}, eta = {best_learning_rate}")

    # Plot the MSE as a function of epochs for the best learning rate
    sns.set_style("darkgrid")
    plt.figure(figsize=(10, 6))
    plt.plot(range(Niterations), cost_values[learning_rates.index(best_learning_rate)], label=f'MSE (Learning Rate = {best_learning_rate})')
    plt.title(f' Plain GD - Lowest MSE score vs. Epochs (Learning Rate = {best_learning_rate})')
    plt.xlabel('Epochs')
    plt.ylabel('MSE')
    plt.legend()
    plt.ylim([0, 3])
    plt.show()

    # Return the best learning rate and best beta
    return best_learning_rate, best_beta

def SGD_plain(X, y, learning_rates, momentum=None, batch_size=32, Niterations=500):
    """
    Performs stochastic gradient descent (SGD) optimization.

    Parameters:
    - X: Feature matrix
    - y: Target variable
    - learning_rates: List of learning rates to be tested
    - momentum: If not None, uses momentum in the optimization
    - batch_size: Size of mini-batches
    - Niterations: Number of iterations

    Returns:
    - Best learning rate and corresponding model parameters
    """
    cost_values = []  # To store MSE values at each learning rate
    best_cost = float('inf')  # Initialize with a large value
    best_learning_rate = 0
    best_beta = None

    for eta in learning_rates:
        beta = np.random.randn(X.shape[1], 1)
        if momentum:
            v = np.zeros_like(beta)

        cost_values_eta = []  # To store MSE values at each epoch for the current learning rate

        for epoch in range(Niterations):
            # Shuffle the data and split into mini-batches
            indices = np.random.permutation(X.shape[0])
            X_shuffled = X[indices]
            y_shuffled = y[indices]
        
            for i in range(0, X.shape[0], batch_size):
                X_mini_batch = X_shuffled[i:i+batch_size]
                y_mini_batch = y_shuffled[i:i+batch_size]
        
                # Calculate the gradient for the mini-batch
                gradient = (2.0 / X_mini_batch.shape[0]) * X_mini_batch.T @ (X_mini_batch @ beta - y_mini_batch)
        
                if momentum:
                    v = momentum * v + eta * gradient
                    beta -= v
                else:
                    beta -= eta * gradient
        
            # Calculate the MSE for the entire dataset after processing all mini-batches
            cost = CostOLS(y, X, beta)
            cost_values_eta.append(cost)

        cost_values.append(cost_values_eta)

        if min(cost_values_eta) < best_cost:
            best_cost = min(cost_values_eta)
            best_learning_rate = eta
            best_beta = beta
            
    if momentum is None:
        print(f"Lowest cost score plain SGD with no momentum {min(cost_values[learning_rates.index(best_learning_rate)])}, eta = {best_learning_rate}")
    else:
        print(f"Lowest cost score plain SGD with momentum {min(cost_values[learning_rates.index(best_learning_rate)])}, eta = {best_learning_rate}")

    # Plot the MSE as a function of epochs for the best learning rate
    sns.set_style("darkgrid")
    plt.figure(figsize=(10, 6))
    plt.plot(range(Niterations), cost_values[learning_rates.index(best_learning_rate)], label=f'MSE (Learning Rate = {best_learning_rate})')
    plt.title(f'Plain SGD - Lowest MSE score vs. Epochs (Learning Rate = {best_learning_rate})')
    plt.xlabel('Epochs')
    plt.ylabel('MSE')
    plt.legend()
    plt.ylim([0, 3])
    plt.show()

    # Return the best learning rate and model parameters
    return best_learning_rate, best_beta

def SGD(X, y, n_epochs=500, M=5, eta=0.01, optimizer="adam", momentum=None):
    """
    Performs stochastic gradient descent (SGD) optimization with optional optimizers.

    Parameters:
    - X: Feature matrix
    - y: Target variable
    - n_epochs: Number of epochs
    - M: Mini-batch size
    - eta: Learning rate
    - optimizer: String indicating the optimizer ("adam" or "rmsprop")
    - momentum: If not None, uses momentum in the optimization

    Returns:
    - Best weights and cost values
    """
    n = X.shape[0]
    m = n // M
    delta = 1e-8
    theta = np.random.randn(X.shape[1], 1)
    training_gradient = grad(CostOLS, 2)
    costs = []

    if optimizer == "adam":
        beta1 = 0.9
        beta2 = 0.999
        first_moment = 0.0
        second_moment = 0.0
        iter = 0
    elif optimizer == "rmsprop":
        rho = 0.99
        Giter = 0.0

    if momentum:
        prev_update = 0.0

    for epoch in range(n_epochs):
        for i in range(m):
            random_index = M * np.random.randint(m)
            xi = X[random_index:random_index + M]
            yi = y[random_index:random_index + M]
            gradients = (1.0 / M) * training_gradient(yi, xi, theta)

            if momentum:
                update = eta * gradients + prev_update
                prev_update = update
            else:
                update = eta * gradients

            if optimizer == "adam":
                iter += 1
                first_moment = beta1 * first_moment + (1 - beta1) * gradients
                second_moment = beta2 * second_moment + (1 - beta2) * gradients * gradients
                first_term = first_moment / (1.0 - beta1**iter)
                second_term = second_moment / (1.0 - beta2**iter)
                update = update + eta * first_term / (ag_np.sqrt(second_term) + delta)
            elif optimizer == "rmsprop":
                Giter = rho * Giter + (1 - rho) * gradients * gradients
                update = gradients*eta/(delta+np.sqrt(Giter))

            theta -= update

        cost = CostOLS(y, X, theta)
        costs.append(cost)
        
    # Plot the cost as a function of epochs
    sns.set_style("darkgrid")
    plt.plot(range(1, len(costs) + 1), costs)
    plt.xlabel('Epochs')
    plt.ylabel('Cost')
    plt.title(f'Cost as a Function of Epochs (No Momentum - optimizer: {optimizer})')
    plt.show()
    if momentum is None:
        print(f'Lowest cost score SGD with {optimizer} (No momentum): {min(costs)}')
    else:
        print(f'Lowest cost score SGD with {optimizer} (Momentum): {min(costs)}')

    return theta, costs

# Seed for reproducibility
np.random.seed(31)

# Generate synthetic data
n = 100
x = np.random.rand(n, 1)
y = 2.0 + 3 * x + 4 * x * x 
X = np.c_[np.ones((n, 1)), x, x * x]
theta_linreg = np.linalg.pinv(X.T @ X) @ (X.T @ y)

# Learning rates to be tested
learning_rates = [0.01, 0.08, 0.1, 0.5, 0.6]

# Plain GD without momentum
GD_plain(X, y, learning_rates, momentum=None)

# Plain GD with momentum
GD_plain(X, y, learning_rates, momentum=0.9)

# Plain SGD without momentum
SGD_plain(X, y, learning_rates, momentum=None)

# Plain SGD with momentum
SGD_plain(X, y, learning_rates, momentum=0.9)

# SGD with optimizer (rmsprop)
SGD(X, y, optimizer='rmsprop')

# SGD with optimizer (rmsprop) and momentum
SGD(X, y, optimizer='rmsprop', momentum=0.9)

# SGD with optimizer (adam)
SGD(X, y, optimizer='adam')

# Scikit SGD
sgdreg = SGDRegressor(max_iter=500, penalty=None, learning_rate='adaptive')
sgdreg.fit(X, y.ravel())
y_pred_sgd = sgdreg.predict(X)
mse_sgd = mean_squared_error(y, y_pred_sgd)
print(f"Minimum Cost Sklearn (SGD): {mse_sgd}")
