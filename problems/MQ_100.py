import numpy as np
import scipy.io as io
import os

A = []
for i in range(5):
    A.append(io.loadmat(os.path.join("matFiles", ("matrix100_%d" % (i+1))))["A"])
B = io.loadmat(os.path.join("matFiles", "matrix100_B"))["B"]

def setParam(p):
    return p

def name():
    return "MQ_100"

def ndim():
    return 100

def init():
    return [1.0 for i in range(100)]

def calcObjs(x):
    f = np.zeros(5)
    for i in range(5):
        f[i] += np.dot(x, 0.5*A[i].dot(x) + B[i,:])
    return f
    
def calcObj(x):
    f = calcObjs(x)
    k = np.argmax(f)
    return f[k]

def calcGrad(x):
    n = ndim()
    g = np.zeros(n)
    f = calcObjs(x)
    k = np.argmax(f)    
    g += A[k].dot(x) + B[k,:]
    return g

def calcBoth(x):
    n = ndim()
    g = np.zeros(n)
    f = calcObjs(x)
    k = np.argmax(f)    
    g += A[k].dot(x) + B[k,:]
    return (f[k], g)

def calcAll(y, isCorrect):
    (f, ret) = calcBoth(y)
    return (f, f, ret)

if __name__ == "__main__":
    n = ndim()
    x = np.zeros(n)
    print calcObjs(x)
    print calcObj(x)
    print calcGrad(x)
