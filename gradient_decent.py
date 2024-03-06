import numpy as np
import matplotlib.pyplot as plt

X = np.array([[0, 1], [2, 0], [2, 3], [4, 2]])
y = np.array([0, 0, 1, 1])

X = np.c_[np.ones(X.shape[0]), X]


theta = np.zeros(X.shape[1])


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def loss_function(X, y, theta):
    m = len(y)
    h = sigmoid(X.dot(theta))
    return -(1/m) * np.sum(y * np.log(h) + (1-y) * np.log(1-h))

def gradient_descent(X, y, theta, learning_rate, epochs):


    for _ in range(epochs):
        h = sigmoid(np.dot(X,theta))
        gradient = -np.dot((y-h),X)
        theta -= learning_rate * gradient
 

    return theta

learning_rate = 0.01
epochs = 100000
theta = gradient_descent(X, y, theta, learning_rate, epochs)

print("Final Parameters:", theta)


plt.scatter(X[y == 0][:, 1], X[y == 0][:, 2], label='Class 1',color='b', marker='o')
plt.scatter(X[y == 1][:, 1], X[y == 1][:, 2], label='Class 2',color='r', marker='o')

x1_values = np.linspace(0, 5, 100)
x2_values = -(theta[0] + theta[1] * x1_values) / theta[2]

plt.plot(x1_values, x2_values, label='Decision Boundary', color='k')

plt.xlabel('X1')
plt.ylabel('X2')
plt.grid(True)
plt.legend()
plt.show()
