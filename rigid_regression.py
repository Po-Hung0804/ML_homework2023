import numpy as np
import matplotlib.pyplot as plt
def rigid(x_bias,y):
    
    alpha=1
    phi=np.c_[x_bias[:,0]-np.mean(x_bias[:,0]),x_bias[:,1]-np.mean(x_bias[:,1])]
    w=np.zeros(phi.shape[1])
    y_bias=np.c_[y-np.mean(y)]
    w=np.dot(np.linalg.inv(np.dot(phi.T,phi)+np.identity(2)*alpha),np.dot(phi.T,y_bias))
    w0=np.mean(y)-np.mean(x_bias[:,0])*w[0]-np.mean(x_bias[:,1]*w[1])
    
    return w0,w
def show(sigma):
    x = np.random.uniform(0, 8, 30)
    noise = np.random.normal(0, np.sqrt(sigma), 30)
    y = 3 + 2 * x + 0.2 * x **2+ noise

    x_data=np.array(x)
    x_bias=np.c_[x_data,x_data*x_data]
    w0,w=rigid(x_bias,y)
    print("w[0]:",w0)
    print("w[1]:",w[0])
    print("w[2]:",w[1])
    plt.figure(figsize=(10, 6))
    plt.scatter(x, y, label='Data Points', color='blue', marker='o')
    plt.xlabel('x')
    plt.ylabel('y')
    x_values = np.linspace(0, x.max(), 100)
    y_values = w0 + w[0] * x_values+w[1]*x_values**2
    plt.plot(x_values, y_values, label=f'rigid Regression Line(rigid) sigma:{sigma}', color='red')
    plt.legend()
    plt.grid(True)
    plt.show()


show(0.1)
