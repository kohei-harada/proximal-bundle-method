import autograd
import autograd.numpy as np
from numpy.random import *

m = 5
n = 10

def setParam(p):
    p.Lambda = 1.0e-3
    return p

def name():
    return "maxquad"

def ndim():
    return n

def init():
    return [0.0 for i in range(n)]

# A = [[[0.0 for k in range(n)] for i in range(n)] for j in range(np)]
A = np.zeros((m,n,n))
# A = np.zeros((5,10,10))
for j in range(m):
    for i in range(n):
        add = 0
        for k in range(i+1,n):
            A[j][i][k] = np.exp(float(i+1)/float(k+1))*np.cos(float((i+1)*(k+1)))*np.sin(float((j+1)))
            # print j, i, k, np.exp(float(i+1)/float(k+1))*np.cos(float((i+1)*(k+1)))*np.sin(float((j+1)))
            add += np.fabs(A[j][i][k])
        for k in range(i):
            A[j][i][k] = A[j][k][i]
            add += np.fabs(A[j][i][k])
        A[j][i][i] = np.fabs(np.sin(float(j+1)))*float(i+1)/float(n) + add

# b = [[0.0 for i in range(n)] for j in range(np)]
b = np.zeros((m,n))
for j in range(m):
    for i in range(n):
        b[j][i] = np.exp(float(i+1)/float(j+1))*np.sin(float(i+1)*float(j+1))

def calcObjs(y):
    # f = [0.0 for j in range(np)]
    # f = np.zeros(m)
    f = []
    for j in range(m):
        # f[j] = np.dot(y, np.dot(A[j], y)) - np.dot(b[j], y)
        f.append(np.dot(y, np.dot(A[j], y)) - np.dot(b[j], y))
    return np.array(f)

def calcObj(y):
    f = calcObjs(y)
    k = np.argmax(f)
    return f[k]

def calcGrad(y):
    g = autograd.grad(calcObj)
    return g(y)

def calcBoth(y):
    f = calcObjs(y)
    k = np.argmax(f)
    g = autograd.grad(calcObj)
    return (f[k], g(y))

def calcAll(y, isCorrect):
    (f, ret) = calcBoth(y)
    return (f, f, ret)

if __name__ == "__main__":
    for i in range(100000):
        rvec = (random(10) - 0.5)*1.0e-2
        (f, g) = calcBoth(rvec, True)
        if f <= 0:
            print i,f
