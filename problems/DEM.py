import autograd
import autograd.numpy as np

def setParam(p):
    return p

def name():
    return "DEM"

def ndim():
    return 2

def init():
    return [1.0 for i in range(2)]

def calcObjs(x):
    f1 = 5*x[0] + x[1]
    f2 = -5*x[0] + x[1]
    f3 = x[0]*x[0] + x[1]*x[1] + 4*x[1]
    return np.array([f1,f2,f3])

def calcObj(x):
    f = calcObjs(x)
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
    x = np.array([1.0, 1.0])
    print(calcObj(x))
    print(calcObjs(x))
    print(calcGrad(x))

