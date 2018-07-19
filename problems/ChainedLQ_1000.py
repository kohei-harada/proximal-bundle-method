import numpy as np

def setParam(p):
    return p

def name():
    return "ChainedLQ_1000"

def ndim():
    return 1000

def init():
    return [-0.5 for i in range(1000)]

def calcObj(y):
    f = 0
    n = ndim()
    for i in np.arange(n-1):
        f1 = - y[i] - y[i+1]
        f2 = f1 + (y[i]**2 + y[i+1]**2 - 1)
        if f1 >= f2:
            f += f1
        else:
            f += f2
    return f

def calcGrad(y):
    n = ndim()
    g = np.zeros(n)
    for i in np.arange(n-1):
        g[i] -= 1.0
        g[i+1] -= 1.0
        ft = y[i]**2 + y[i+1]**2 - 1
        if ft >= 0:
            g[i] += 2.0*y[i]
            g[i+1] += 2.0*y[i+1]
    return g

def calcBoth(y):
    n = ndim()
    g = np.zeros(n)
    f = 0
    for i in np.arange(n-1):
        f += - y[i] - y[i+1]
        g[i] -= 1.0
        g[i+1] -= 1.0
        ft = y[i]**2 + y[i+1]**2 - 1
        if ft >= 0:
            g[i] += 2.0*y[i]
            g[i+1] += 2.0*y[i+1]
            f += ft
    return (f, g)

def calcAll(y, isCorrect):
    (f, ret) = calcBoth(y)
    return (f, f, ret)

if __name__ == "__main__":
    y = np.array([1.0,1.0,1.0,1.0,2.0])
    print calcObj(y)
    print calcGrad(y)
