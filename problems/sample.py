import numpy as np

# Function setParam set original parameters for the problem.
# The argument p is a instance of class Param defined at param.py
# Usually, all you have to do is just returnning p as is, i.e. all parameters are default.
# If you would like to solve multiple problems one-shot and the necessary parameters
# are different one another, you need to use setParam individually.
# Otherwise, change param.py directly is prefered.
def setParam(p):
    # If you want to let the stopping criteria stronger(default value is 1.0e-5) for the problem
    # p.eps = 1.0e-6
    return p

# Function name returns the name of the problem
def name():
    return "sample"

# Function ndim returns the number of variables of the problem
def ndim():
    return 2

# Function init returns the initial point of the problem
# The length of the array must be equal to the return value of ndim
def init():
    return np.array([1.0, 2.0])

def calcObj(x):
    return 0.5*(x[0] - 2.0)*(x[0] - 2.0) + (x[1] + 1.0)*(x[1] + 1.0)

def calcGrad(x):
    return np.array([x[0] - 2.0, 2.0*(x[1] + 1.0)])

# Function calcBoth returns the objective value and a subgradient for a given point x
# This function plays the role of the "oracle" of the problem
def calcBoth(x):
    f = calcObj(x)
    g = calcGrad(x)
    return (f, g)
