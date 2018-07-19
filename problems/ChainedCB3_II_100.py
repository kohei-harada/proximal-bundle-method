import math
import numpy as np

def setParam(p):
    p.Lambda = 1.0e-3
    return p

def name():
    return "ChainedCB3_II_100"

def ndim():
    return 100

def init():
    return [2.0 for i in range(100)]

def calcObj(y):
    ret = 0
    n = ndim()
    ary1 = [y[i]**4 + y[i+1]**2 for i in range(n-1)]
    ary2 = [(y[i]-2)**2 + (y[i+1]-2)**2 for i in range(n-1)]
    ary3 = [2*np.exp(-y[i]+y[i+1]) for i in range(n-1)]
    return max([sum(ary1), sum(ary2), sum(ary3)])

def calcGrad(y):
    n = ndim()
    ret = [0 for i in range(n)]
    ary1 = [y[i]**4 + y[i+1]**2 for i in range(n-1)]
    ary2 = [(y[i]-2)**2 + (y[i+1]-2)**2 for i in range(n-1)]
    ary3 = [2*np.exp(-y[i]+y[i+1]) for i in range(n-1)]
    k = np.argmax(np.array([sum(ary1), sum(ary2), sum(ary3)]))
    for i in range(n-1):
        if k==0:
            ret[i] += 4*y[i]**3
            ret[i+1] += 2*y[i+1]
        elif k==1:
            ret[i] += 2*(y[i]-2)
            ret[i+1] += 2*(y[i+1]-2)
        elif k==2:
            ret[i] += -2*np.exp(-y[i]+y[i+1])
            ret[i+1] += 2*np.exp(-y[i]+y[i+1])
    return ret

def calcBoth(y):
    n = ndim()
    ret = [0 for i in range(n)]
    ary1 = [y[i]**4 + y[i+1]**2 for i in range(n-1)]
    ary2 = [(y[i]-2)**2 + (y[i+1]-2)**2 for i in range(n-1)]
    ary3 = [2*np.exp(-y[i]+y[i+1]) for i in range(n-1)]
    ary = np.array([sum(ary1), sum(ary2), sum(ary3)])
    k = np.argmax(ary)
    for i in range(n-1):
        if k==0:
            ret[i] += 4*y[i]**3
            ret[i+1] += 2*y[i+1]
        elif k==1:
            ret[i] += 2*(y[i]-2)
            ret[i+1] += 2*(y[i+1]-2)
        elif k==2:
            ret[i] += -2*np.exp(-y[i]+y[i+1])
            ret[i+1] += 2*np.exp(-y[i]+y[i+1])
    return (ary[k], ret)

def calcAll(y, isCorrect):
    (f, ret) = calcBoth(y)
    return (f, f, ret)
