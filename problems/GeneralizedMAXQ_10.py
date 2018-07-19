import numpy as np

def setParam(p):
    return p

def name():
    return "GeneralizedMAXQ_10"

def ndim():
    return 10

def init():
    ret = [float(i) + 1.0 for i in range(10)]
    for i in range(5,10):
        ret[i] = - ret[i]
    return ret

def calcObj(y):
    ary = [t**2 for t in y]
    return max(ary)

def calcGrad(y):
    ary = np.array([t**2 for t in y])
    k = np.argmax(ary)
    n = ndim()
    ret = [0 for i in range(n)]
    ret[k] = 2*y[k]
    return ret

def calcBoth(y):
    ary = np.array([t**2 for t in y])
    k = np.argmax(ary)
    n = ndim()
    ret = [0 for i in range(n)]
    ret[k] = 2*y[k]
    return (ary[k], ret)

def calcAll(y, isCorrect):
    (f, ret) = calcBoth(y)
    return (f, f, ret)
