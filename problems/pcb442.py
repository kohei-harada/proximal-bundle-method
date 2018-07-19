import sys
import msp
import scipy.io as io
import numpy as np
import os

M = io.loadmat(os.path.join("matFiles","TSP_pcb442"))["M"]
n = M.shape[0]

def setParam(p):
    return p

def name():
    return "pcb442"

def ndim():
    return n

def init():
    return [0.0 for i in range(442)]

def update(M, x):
    M = M.copy()
    for i in range(n):
        for j in range(n):
            M[i][j] -= x[i]
            M[j][i] -= x[i]
    return M

def getMin2(v):
    (k1,k2) = (0,1)
    if v[0] > v[1]:
        (k1,k2) = (1,0)
    for i in range(2,len(v)):
        if v[i] < v[k1]:
            (k1,k2) = (i,k1)
        elif v[i] < v[k2]:
            k2 = i
    return (k1,k2)                     

def calcCostAndDegree(X):
    edges = msp.minimum_spanning_tree(X)
    f = 0
    g = np.array([2 for i in range(n-1)])
    for (i,j) in edges:
        f += X[i][j]
        g[i] -= 1
        g[j] -= 1
    return (f,g)

def calcBoth(x):
    X = update(M, x)
    v = X[0]
    # print v
    v = np.delete(v, 0)
    (k1,k2) = getMin2(v)    
    X = np.delete(X, 0, 0)
    X = np.delete(X, 0, 1)
    #edges = msp.minimum_spanning_tree(X)
    #print edges
    (f, g) = calcCostAndDegree(X)
    g = np.insert(g, 0, 0)
    g[k1+1] -= 1
    g[k2+1] -= 1
    f += v[k1] + v[k2]
    f += 2*np.sum(x)
    return (-f,-g)

def calcObj(x):
    (f, g) = calcBoth(x)
    return f

def calcGrad(x):
    (f, g) = calcBoth(x)
    return g

def calcAll(y, isCorrect):
    (f, ret) = calcBoth(y)
    return (f, f, ret)

if __name__ == "__main__":
    x = np.array([2 for i in range(n)])
    print calcObj(x)

    
