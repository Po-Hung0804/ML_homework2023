import numpy as np
import matplotlib.pyplot as plt
def comp(phi,w,j,y):
    tmp=np.zeros(phi.shape[0])
    for i in range(0,phi.shape[1]):
        if i!=j:
            tmp+=w[i+1]*phi[:,i]
    cj=np.sum(phi[:,j]*(y-tmp))
    return cj
def lasso(x_bias,y):
    alpha=1
    phi=np.c_[x_bias[:,0]-np.mean(x_bias[:,0]),x_bias[:,1]-np.mean(x_bias[:,1])]
    
    w=np.zeros(phi.shape[1]+1)
    c=np.zeros(phi.shape[1])
    a=np.sum(phi*phi,axis=0)
    
    for _ in range(100000):
        w[0]=np.mean(y)-np.sum(np.mean(x_bias[:,0])*w[1]+np.mean(x_bias[:,1])*w[2])
        for j in range(0,phi.shape[1]):
            c[j]=comp(phi,w,j,y)
            if c[j]>alpha:
                w[j+1]=(c[j]-alpha)/a[j]
            elif c[j]<-alpha:
                w[j+1]=(c[j]+alpha)/a[j]
            else:
                w[j+1]=0
    return w
def show(sigma):
    x = np.random.uniform(0, 8, 30)
    noise = np.random.normal(0, np.sqrt(sigma), 30)
    y = 3 + 2 * x + 0.2 * x **2+ noise

    x_data=np.array(x)
    x_bias=np.c_[x_data,x_data*x_data]


    y_data=np.array(y)
    w=lasso(x_bias,y)
    print(w)
    plt.figure(figsize=(10, 6))
    plt.scatter(x, y, label='Data Points', color='blue', marker='o')
    plt.xlabel('x')
    plt.ylabel('y')
    x_values = np.linspace(0, x.max(), 100)
    y_values = w[0] + w[1] * x_values+w[2]*x_values**2
    plt.plot(x_values, y_values, label= f'lasso Regression Line(lasso) sigma:{sigma}', color='red')
    plt.legend()
    plt.grid(True)
    plt.show()

show(0.1)