import numpy as np
import scipy.io as io
from numpy.linalg import norm
import os

A = io.loadmat(os.path.join("matFiles","matrix10_1"))["A"]

def setParam(p):
    return p

def name():
    return "TiltedNorm_10"

def ndim():
    return 10

def init():
    return [1.0 for i in range(10)]

def calcObj(x):
    Ax = A.dot(x)
    k = np.argmax(np.absolute(Ax))
    return 4.0*np.absolute(Ax[k]) + 3.0*Ax[0]

def calcGrad(x):
    n = ndim()
    g = np.zeros(n)
    Ax = A.dot(x)
    k = np.argmax(np.absolute(Ax))
    for i in range(n):
        # g[i] += 4.0*A[k][i] + 3.0*A[0][i]
        g[i] += 3.0*A[0][i]
        if Ax[k] >= 0:
            g[i] += 4.0*A[k][i]
        else:
            g[i] -= 4.0*A[k][i]
    return g

def calcBoth2(x):
    n = ndim()
    g = np.zeros(n)
    Ax = A.dot(x)
    k = np.argmax(np.absolute(Ax))
    for i in range(n):
        # g[i] += 4.0*A[k][i] + 3.0*A[0][i]
        g[i] += 3.0*A[0][i]
        if Ax[k] >= 0:
            g[i] += 4.0*A[k][i]
        else:
            g[i] -= 4.0*A[k][i]
    return (4.0*np.absolute(Ax[k]) + 3.0*Ax[0], g)

def calcBoth(x):
    n = ndim()
    g = np.zeros(n)
    Ax = A.dot(x)
    k = np.argmax(np.absolute(Ax))
    f = 4.0*norm(Ax) + 3.0*Ax[0]
    g = 4.0*A.dot(Ax)/norm(Ax) + 3.0*A[0]
    return (f, g)

def calcAll(y, isCorrect):
    (f, ret) = calcBoth(y)
    return (f, f, ret)

if __name__ == "__main__":
    # x = np.arange(10)
    n = ndim()
    x = np.zeros(n)
    print(calcObj(x))
    print(calcGrad(x))
