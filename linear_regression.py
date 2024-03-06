import numpy as np
import matplotlib.pyplot as plt


sigma=float(input("Enter the sigma:\n"))
x = np.random.uniform(0, 8, 40)
noise = np.random.normal(0, np.sqrt(sigma), 40)
y = 3 + 2 * x + 0.2 * x **2+ noise
dataset = list(zip(x, y))

    

def linear_regression(dataset):
    data=np.array(dataset)
    X = data[:, 0].reshape(-1, 1)
    Y=data[:,1].reshape(-1,1)
    theta = np.hstack((np.ones((X.shape[0], 1)), X, X**2))
    theta_1=np.transpose(theta)
    z=np.dot(theta_1,theta)
    thetay=np.dot(theta_1,Y)
    z_inv=np.linalg.inv(z)
    weight=np.dot(z_inv,thetay)
    return weight
w=linear_regression(dataset)
print(w)
plt.figure(figsize=(10, 6))
plt.scatter(x, y, label='Data Points', color='blue', marker='o')
plt.xlabel('x')
plt.ylabel('y')
x_values = np.linspace(0, x.max(), 100)
y_values = w[0][0] + w[1][0] * x_values+w[2][0]*x_values**2
plt.plot(x_values, y_values, label='Linear Regression Line', color='red')
plt.legend()
plt.grid(True)
plt.show()
