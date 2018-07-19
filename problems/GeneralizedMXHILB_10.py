import numpy as np

def setParam(p):
    return p

def name():
    return "GeneralizedMXHILB_10"

def ndim():
    return 10

def init():
    return [1.0 for i in range(10)]

def calcObjs(y):
    f = []
    n = ndim()
    for i in range(n):
        tary = [y[j]/(i+j+1) for j in range(n)]
        tmp = sum(tary)
        f.append(tmp)
    return f

def calcObj(y):
    f = calcObjs(y)
    f_abs = np.absolute(np.array(f))
    k = np.argmax(f_abs)
    return f_abs[k]

def calcGrad(y):
    n = ndim()
    g = np.zeros(n)    
    f = calcObjs(y)
    f_abs = np.absolute(np.array(f))
    k = np.argmax(f_abs)
    for j in range(n):
        g[j] = 1.0/(k+j+1)
    if f_abs[k] > f[k]:
        g = -g
    return g

def calcBoth(y):
    n = ndim()
    g = np.zeros(n)    
    f = calcObjs(y)
    f_abs = np.absolute(np.array(f))
    k = np.argmax(f_abs)
    for j in range(n):
        g[j] = 1.0/(k+j+1)
    if f_abs[k] > f[k]:
        g = -g
    return (f_abs[k], g)

def calcAll(y, isCorrect):
    (f, ret) = calcBoth(y)
    return (f, f, ret)
