import numpy as np
import matplotlib.pyplot as plt



class1_data = np.random.multivariate_normal([1, 1], [[1, 0], [0, 1]], 50)
class2_data = np.random.multivariate_normal([-3, -3], [[1, 0], [0, 1]], 50)
class3_data = np.random.multivariate_normal([-6, 2], [[1, 0], [0, 1]], 50)

X = np.vstack((class1_data, class2_data, class3_data))
y = np.hstack((np.zeros(50), np.ones(50), 2 * np.ones(50)))

X = np.c_[np.ones(X.shape[0]), X]


def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def compute_loss(X, y, theta):
    m = len(y)
    h = sigmoid(X.dot(theta))
    epsilon = 1e-5
    loss = (-1 / m) * (y.T.dot(np.log(h + epsilon)) + (1 - y).T.dot(np.log(1 - h + epsilon)))
    return loss

def gradient_descent(X, y, theta, learning_rate, iterations):
    m = len(y)
    loss_history = np.zeros(iterations)

    for i in range(iterations):
        h = sigmoid(X.dot(theta))
        gradient = X.T.dot(h - y) / m
        theta -= learning_rate * gradient
        loss_history[i] = compute_loss(X, y, theta)

    return theta, loss_history

theta = np.zeros(X.shape[1])

learning_rate = 0.01
iterations = 1000

decision_boundaries = []

for i in range(3):
    for j in range(i+1, 3):
      
        y_binary = np.where((y == i) | (y == j), 1, 0)
        
      
        theta_binary, _ = gradient_descent(X, y_binary, theta.copy(), learning_rate, iterations)
        decision_boundaries.append(theta_binary)

plt.scatter(X[:, 1], X[:, 2], c=y, cmap=plt.cm.Paired, edgecolor='k', s=30)
plt.title("Decision Boundaries for Three Classes")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")

for i in range(3):
    for j in range(i+1, 3):
        theta_binary = decision_boundaries.pop(0)
        x_boundary = np.linspace(X[:, 1].min(), X[:, 1].max(), 100)
        y_boundary = -(theta_binary[0] + theta_binary[1] * x_boundary) / theta_binary[2]
        plt.plot(x_boundary, y_boundary,color='k')

plt.legend()
plt.grid(True)
plt.show()