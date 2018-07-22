import autograd
import autograd.numpy as np

def setParam(p):
    return p

def name():
    return "RosenSuzuki"

def ndim():
    return 4

def init():
    return [0.2 for i in range(4)]

def calcObjs(y):
    f1 = y[0]**2 + y[1]**2 + 2*y[2]**2 + y[3]**2 - 5*y[0] - 5*y[1] - 21*y[2] + 7*y[3]
    f2 = f1 + 10*(y[0]**2 + y[1]**2 + y[2]**2 + y[3]**2 + y[0] - y[1] + y[2] - y[3] - 8)
    f3 = f1 + 10*(y[0]**2 + 2*y[1]**2 + y[2]**2 + 2*y[3]**2 - y[0] - y[3] - 10)
    f4 = f1 + 10*(2*y[0]**2 + y[1]**2 + y[2]**2 + 2*y[0] - y[1] - y[3] - 5)
    return np.array([f1, f2, f3, f4])

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
    y = np.array([0.0, 1.0, 2.0, -1.0])
    print(calcObj(y))
    print(calcObjs(y))
    print(calcGrad(y))
