import numpy as np

def setParam(p):
    p.Lambda = 1.0e-3
    return p

def name():
    return "ChainedCB3_I_10"

def ndim():
    return 10

def init():
    return [2.0 for i in range(10)]

def calcObj(y):
    ret = 0
    n = ndim()
    for i in range(n-1):
        ret += max([y[i]**4 + y[i+1]**2
                    ,(y[i]-2)**2 + (y[i+1]-2)**2
                    ,2*np.exp(-y[i]+y[i+1])])
    return ret

def calcGrad(y):
    n = ndim()
    ret = [0 for i in range(n)]
    for i in range(n-1):
        k = np.argmax(np.array([y[i]**4 + y[i+1]**2
                                ,(y[i]-2)**2 + (y[i+1]-2)**2
                                ,2*np.exp(-y[i]+y[i+1])]))
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
    f = 0
    ret = [0 for i in range(n)]
    for i in range(n-1):
        ary = np.array([y[i]**4 + y[i+1]**2
                        ,(y[i]-2)**2 + (y[i+1]-2)**2
                        ,2*np.exp(-y[i]+y[i+1])])
        k = np.argmax(ary)
        f += np.max(ary)
        if k==0:
            ret[i] += 4*y[i]**3
            ret[i+1] += 2*y[i+1]
        elif k==1:
            ret[i] += 2*(y[i]-2)
            ret[i+1] += 2*(y[i+1]-2)
        elif k==2:
            ret[i] += -2*np.exp(-y[i]+y[i+1])
            ret[i+1] += 2*np.exp(-y[i]+y[i+1])
    return (f, ret)

def calcAll(y, isCorrect):
    (f, ret) = calcBoth(y)
    return (f, f, ret)
