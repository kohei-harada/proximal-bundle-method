import numpy as np
import param

def setParam(p):
    p.Lambda = 1.0e-2
    return p

def name():
    return "EVD52"

def ndim():
    return 3

def init():
    return [1.0 for i in range(3)]

def calcObj(y):
    f1 = y[0]*y[0] + y[1]*y[1] + y[2]*y[2] - 1
    f2 = y[0]*y[0] + y[1]*y[1] + (y[2]-2)*(y[2]-2)
    f3 = y[0] + y[1] + y[2] - 1
    f4 = y[0] + y[1] - y[2] + 1
    f5 = 2*y[0]**3 + 6*y[1]**2 + 2*(5*y[2]-y[0]+1)**2
    f6 = y[0]**2 - 9*y[2]
    ary = [f1, f2, f3, f4, f5, f6]
    return max(ary)

def calcGrad(y):
    n = ndim()
    ret = [0 for i in range(n)]
    f1 = y[0]*y[0] + y[1]*y[1] + y[2]*y[2] - 1
    f2 = y[0]*y[0] + y[1]*y[1] + (y[2]-2)*(y[2]-2)
    f3 = y[0] + y[1] + y[2] - 1
    f4 = y[0] + y[1] - y[2] + 1
    f5 = 2*y[0]**3 + 6*y[1]**2 + 2*(5*y[2]-y[0]+1)**2
    f6 = y[0]**2 - 9*y[2]
    ary = [f1, f2, f3, f4, f5, f6]
    k = np.argmax(np.array(ary))
    if k == 0:
        ret = [2*y[0], 2*y[1], 2*y[2]]
    elif k == 1:
        ret = [2*y[0], 2*y[1], 2*(y[2]-2)]
    elif k == 2:
        ret = [1, 1, 1]
    elif k == 3:
        ret = [1, 1, -1]
    elif k == 4:
        ret = [6*y[0]**2 - 4*(5*y[2]-y[0]+1), 12*y[1], 20*(5*y[2]-y[0]+1)]
    elif k == 5:
        ret = [2*y[0], 0, 9]
    return ret

def calcBoth(y):
    n = ndim()
    ret = [0 for i in range(n)]
    f1 = y[0]*y[0] + y[1]*y[1] + y[2]*y[2] - 1
    f2 = y[0]*y[0] + y[1]*y[1] + (y[2]-2)*(y[2]-2)
    f3 = y[0] + y[1] + y[2] - 1
    f4 = y[0] + y[1] - y[2] + 1
    f5 = 2*y[0]**3 + 6*y[1]**2 + 2*(5*y[2]-y[0]+1)**2
    f6 = y[0]**2 - 9*y[2]
    ary = [f1, f2, f3, f4, f5, f6]
    k = np.argmax(np.array(ary))
    if k == 0:
        ret = [2*y[0], 2*y[1], 2*y[2]]
    elif k == 1:
        ret = [2*y[0], 2*y[1], 2*(y[2]-2)]
    elif k == 2:
        ret = [1, 1, 1]
    elif k == 3:
        ret = [1, 1, -1]
    elif k == 4:
        ret = [6*y[0]**2 - 4*(5*y[2]-y[0]+1), 12*y[1], 20*(5*y[2]-y[0]+1)]
    elif k == 5:
        ret = [2*y[0], 0, 9]
    return (ary[k], ret)

def calcAll(y, isCorrect):
    (f, g) = calcBoth(y)
    return (f,f,g)
