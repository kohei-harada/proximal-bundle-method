import numpy as np

A = np.array([[0.0, 0.0, 0.0, 0.0, 0.0],
              [2.0, 1.0, 1.0, 1.0, 3.0],
              [1.0, 2.0, 1.0, 1.0, 2.0],
              [1.0, 4.0, 1.0, 2.0, 2.0],
              [3.0, 2.0, 1.0, 0.0, 1.0],
              [0.0, 2.0, 1.0, 0.0, 1.0],
              [1.0, 1.0, 1.0, 1.0, 1.0],
              [1.0, 0.0, 1.0, 2.0, 1.0],
              [0.0, 0.0, 2.0, 1.0, 0.0],
              [1.0, 1.0, 2.0, 0.0, 0.0]])

b = np.array([1.0, 5.0, 10.0, 2.0, 4.0, 3.0, 1.7, 2.5, 6.0, 3.5])

def setParam(p):
    p.Lambda = 1.0e-2
    return p

def name():
    return "Shor"

def ndim():
    return 5

def init():
    ret = [0.0 for i in range(5)]
    ret[4] = 1.0
    return ret

def calcObjs(x):
    n = ndim()
    f = np.zeros(10)
    for i in range(10):
        for j in range(n):
            # print "%d,%d,%f" % (i,j,A[i][j])
            f[i] += (x[j] - A[i][j])**2
        f[i] *= b[i]
    return f

def calcObj(x):
    f = calcObjs(x)
    k = np.argmax(f)
    return f[k]

def calcGrad(x):
    # g = autograd.grad(calcObj)
    # return g(x)
    n = ndim()
    g = np.zeros(n)
    f = calcObjs(x)
    k = np.argmax(f)
    for j in range(n):
        g[j] += 2.0*b[k]*(x[j] - A[k][j])
    return g

def calcBoth(x):
    # g = autograd.grad(calcObj)
    # return g(x)
    n = ndim()
    g = np.zeros(n)
    f = calcObjs(x)
    k = np.argmax(f)
    for j in range(n):
        g[j] += 2.0*b[k]*(x[j] - A[k][j])
    return (f[k], g)

def calcAll(y, isCorrect):
    (f, ret) = calcBoth(y)
    return (f, f, ret)

if __name__ == "__main__":
    x = np.array([0.0, 0.0, 0.0, 0.0, 1.0])
    print(calcObj(x))
    print(calcObjs(x))
    print(calcGrad(x))

