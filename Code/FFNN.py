import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
import pandas as pd

class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size, learning_rate, hidden_activation, output_activation):
        """
        Initialize a neural network with specified parameters.

        Parameters:
        - input_size (int): Number of input features.
        - hidden_size (int): Number of neurons in the hidden layer.
        - output_size (int): Number of neurons in the output layer.
        - learning_rate (float): Learning rate for weight updates.
        - hidden_activation (str): Activation function for the hidden layer ("sigmoid", "relu", "leaky_relu").
        - output_activation (str): Activation function for the output layer ("sigmoid", "relu", "leaky_relu").
        """
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate
        self.hidden_activation = hidden_activation
        self.output_activation = output_activation

        # Initialize weight matrices with correct dimensions
        self.hidden_weights = np.random.randn(self.input_size, self.hidden_size)
        self.hidden_bias = np.zeros(self.hidden_size) + 0.01
        self.output_weights = np.random.randn(self.hidden_size, self.output_size)
        self.output_bias = np.zeros(self.output_size) + 0.01

    def sigmoid(self, x):
        """
        Sigmoid activation function.

        Parameters:
        - x (numpy.ndarray): Input to the activation function.

        Returns:
        - numpy.ndarray: Output of the sigmoid activation.
        """
        x = np.clip(x, -500, 500)
        return 1 / (1 + np.exp(-x))

    def relu(self, x):
        """
        ReLU activation function.

        Parameters:
        - x (numpy.ndarray): Input to the activation function.

        Returns:
        - numpy.ndarray: Output of the ReLU activation.
        """
        return np.maximum(0, x)

    def leaky_relu(self, x, alpha=0.01):
        """
        Leaky ReLU activation function.

        Parameters:
        - x (numpy.ndarray): Input to the activation function.
        - alpha (float): Slope for negative values.

        Returns:
        - numpy.ndarray: Output of the Leaky ReLU activation.
        """
        return np.where(x > 0, x, alpha * x)

    def feed_forward(self, X):
        """
        Perform feedforward pass through the neural network.

        Parameters:
        - X (numpy.ndarray): Input data.

        Returns:
        - tuple: Tuple containing probabilities and hidden layer activations.
        """
        z_h = np.dot(X, self.hidden_weights) + self.hidden_bias

        if self.hidden_activation == "sigmoid":
            a_h = self.sigmoid(z_h)
        elif self.hidden_activation == "relu":
            a_h = self.relu(z_h)
        elif self.hidden_activation == "leaky_relu":
            a_h = self.leaky_relu(z_h)

        z_o = np.dot(a_h, self.output_weights) + self.output_bias

        if self.output_activation == "sigmoid":
            probabilities = self.sigmoid(z_o)
        elif self.output_activation == "relu":
            probabilities = self.relu(z_o)
        elif self.output_activation == "leaky_relu":
            probabilities = self.leaky_relu(z_o)

        return probabilities, a_h

    def train(self, X, y, n_epochs):
        """
        Train the neural network.

        Parameters:
        - X (numpy.ndarray): Training input data.
        - y (numpy.ndarray): Training labels.
        - n_epochs (int): Number of training epochs.

        Returns:
        - list: List of accuracy scores during training.
        """
        accuracy_scores = []

        for epoch in range(n_epochs):
            probabilities, a_h = self.feed_forward(X)

            y_pred = (probabilities > 0.5).astype(int)
            accuracy = accuracy_score(y, y_pred)
            accuracy_scores.append(accuracy)

            error = y.reshape(-1, 1) - probabilities

            d_output_weights = np.dot(a_h.T, error)
            d_hidden_weights = np.dot(X.T, error @ self.output_weights.T)

            self.output_weights += self.learning_rate * d_output_weights
            self.hidden_weights += self.learning_rate * d_hidden_weights

        return accuracy_scores


def train_and_evaluate(hidden_activation_func, output_activation_func, n_hidden_neurons, learning_rate, n_epochs, X_train, y_train, X_test, y_test):
    """
    Train and evaluate the neural network with specified hyperparameters.

    Parameters:
    - hidden_activation_func (str): Activation function for the hidden layer.
    - output_activation_func (str): Activation function for the output layer.
    - n_hidden_neurons (int): Number of neurons in the hidden layer.
    - learning_rate (float): Learning rate for weight updates.
    - n_epochs (int): Number of training epochs.
    - X_train (numpy.ndarray): Training input data.
    - y_train (numpy.ndarray): Training labels.
    - X_test (numpy.ndarray): Test input data.
    - y_test (numpy.ndarray): Test labels.

    Returns:
    - tuple: Tuple containing test accuracy and list of accuracy scores during training.
    """
    model = NeuralNetwork(n_features, n_hidden_neurons, n_categories, learning_rate, hidden_activation_func, output_activation_func)
    accuracy_scores = model.train(X_train, y_train, n_epochs)

    # Evaluate the model on the test set
    probabilities, _ = model.feed_forward(X_test)
    y_pred = (probabilities > 0.5).astype(int)
    test_accuracy = accuracy_score(y_test, y_pred)

    return test_accuracy, accuracy_scores


#Classification analysis

# Set seed for reproducibility
np.random.seed(31)
# Load breast cancer dataset
data = load_breast_cancer()
X = data.data
y = data.target

# Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Define neural network parameters
n_inputs, n_features = X_train.shape
n_categories = 1

# Define hyperparameter values to iterate over
hidden_activation_funcs = ["sigmoid"]
output_activation_funcs = ["sigmoid", "relu", "leaky_relu"]
n_hidden_neurons_values = [5, 10, 15]
learning_rate_values = [0.001, 0.01, 0.1]
n_epochs_values = [500, 1000, 1500]

# Iterate over hyperparameters and store results
results = []
results_sklearn = []

for hidden_activation_func in hidden_activation_funcs:
    for output_activation_func in output_activation_funcs:
        for n_hidden_neurons in n_hidden_neurons_values:
            for learning_rate in learning_rate_values:
                for n_epochs in n_epochs_values:
                    test_accuracy, accuracy_scores = train_and_evaluate(hidden_activation_func, output_activation_func, n_hidden_neurons, learning_rate, n_epochs, X_train, y_train, X_test, y_test)
                    results.append({
                        'Hidden Activation': hidden_activation_func,
                        'Output Activation': output_activation_func,
                        'Hidden Neurons': n_hidden_neurons,
                        'Learning Rate': learning_rate,
                        'Epochs': n_epochs,
                        'Test Accuracy': test_accuracy
                    })

                    model_sklearn = MLPClassifier(hidden_layer_sizes=n_hidden_neurons,
                                                 activation='relu',
                                                 learning_rate_init=learning_rate,
                                                 max_iter=n_epochs)
                    model_sklearn.fit(X_train, y_train)

                    test_accuracy_sklearn = model_sklearn.score(X_test, y_test)

                    results_sklearn.append({
                        'Hidden Layer Sizes Sk': n_hidden_neurons,
                        'Activation Function Sk': 'relu',
                        'Learning Rate Sk': learning_rate,
                        'Max Iterations Sk': n_epochs,
                        'Test Accuracy Sk': test_accuracy_sklearn
                    })

# Print or store the results as needed
for result in results:
    print(result)

for result_ in results_sklearn:
    print(result_)

# Create DataFrames and save results to Excel files
df_results = pd.DataFrame(results)
df_results_sorted = df_results.sort_values(by='Test Accuracy', ascending=False)
df_results_sorted.to_excel('results_sorted.xlsx', index=False)

df_results_sklearn = pd.DataFrame(results_sklearn)
df_results_sklearn_sorted = df_results_sklearn.sort_values(by='Test Accuracy Sk', ascending=False)
df_results_sklearn_sorted.to_excel('results_sklearn_sorted.xlsx', index=False)


#Regression analysis compared to OLS/Ridge

"""
np.random.seed(31)
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float64)
y = np.array([0, 1, 1, 0])

n_inputs, n_features = X.shape
n_hidden_neurons = 4
n_categories = 1
learning_rate = 0.1
n_epochs = 20000

# Activation function options
activation_functions = ["sigmoid", "relu", "leaky_relu"]

# Iterate over all combinations
for hidden_activation_func in activation_functions:
    for output_activation_func in activation_functions:
        model = NeuralNetwork(n_features, n_hidden_neurons, n_categories, learning_rate, hidden_activation_func, output_activation_func)

        
        # Check for NaN values in MSE scores
        try:
            mse_scores, _ = model.train(X, y, n_epochs)
            plt.figure(figsize=(12, 6))
            plt.plot(mse_scores)
            plt.xlabel('Epochs')
            plt.ylabel('MSE score')
            plt.title(f'MSE scores for Combination: Hidden: {hidden_activation_func}, Output: {output_activation_func} ')
            plt.show()
        except ValueError as e:
            print(f"Combination: Hidden: {hidden_activation_func}, Output: {output_activation_func}, Error: {e}")
            continue

        # Print the combination and its MSE scores
        print(f'Combination: Hidden: {hidden_activation_func}, Output: {output_activation_func}, Lowest MSE score: {min(mse_scores)}')

np.random.seed(8)
X = np.array([[0, 1], [0, 1], [1, 0], [1, 1]], dtype=np.float64)
y = np.array([0, 1, 1, 0])

# Standardize the data
scaler = StandardScaler()
X_std = scaler.fit_transform(X)

# Ordinary Least Squares (OLS) Regression
ols_mse = mean_squared_error(y, LinearRegression().fit(X_std, y).predict(X_std))

# Ridge Regression
ridge_mse = mean_squared_error(y, Ridge(alpha=0.5).fit(X_std, y).predict(X_std))

# Print the MSE scores for OLS and Ridge regression
print(f'OLS MSE score: {ols_mse}')
print(f'Ridge MSE score: {ridge_mse}')



"""

