import numpy as np
import matplotlib.pyplot as plt

x1 = np.array([0, 1])
x2 = np.array([2, 0])

x3 = np.array([2, 3])
x4 = np.array([4, 2])


data = np.array([x1, x2, x3, x4])
labels = np.array([1, 1, -1, -1])


w = np.zeros(data.shape[1])
b = 0


def train_svm(data, labels, epochs=1000, learning_rate=0.1):
    global w, b  
    
    for epoch in range(epochs):
        for i, x in enumerate(data):
            
            if labels[i] * (np.dot(w, x) - b) < 1:
                w = w + learning_rate * ((labels[i] * x) - (2 * (1/epochs) * w))
                b = b - learning_rate * labels[i]
            else:
                w = w - learning_rate * (2 * (1/epochs) * w)

# Train SVM model
train_svm(data, labels)

# Plot the decision boundary
plt.scatter(data[:, 0], data[:, 1], c=labels, cmap=plt.cm.Paired, edgecolors='k')

xlim = plt.gca().get_xlim()
ylim = plt.gca().get_ylim()

xx, yy = np.meshgrid(np.linspace(xlim[0], xlim[1], 50),
                     np.linspace(ylim[0], ylim[1], 50))

Z = np.dot(np.c_[xx.ravel(), yy.ravel()], w) - b
Z = Z.reshape(xx.shape)

plt.contour(xx, yy, Z, colors='k', levels= 0, alpha=0.5, linestyles= '-')

plt.title('Linear SVM Decision Boundary')
plt.grid(True)
plt.show()
