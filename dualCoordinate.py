import sys
import re
import os
import numpy as np

def createInput(ffname, gfname, yfname, dir):
    f = []
    g = []
    y = []
    ycenter = []
    ffname = os.path.join(dir, ffname)
    gfname = os.path.join(dir, gfname)
    yfname = os.path.join(dir, yfname)    
    with open(ffname) as rf:
        rf.readline()
        for line in rf.readlines():
            ary = re.split(",", line.strip())
            f.append(float(ary[1]))
    with open(gfname) as rf:
        rf.readline()
        for line in rf.readlines():
            ary = re.split(",", line.strip())
            g.append(float(ary[2]))
            y.append(float(ary[3]))
    with open(yfname) as rf:
        rf.readline()
        for line in rf.readlines():
            ary = re.split(",", line.strip())
            ycenter.append(float(ary[1]))
    f = np.array(f)
    g = np.array(g)
    y = np.array(y)
    ycenter = np.array(ycenter)
    g = np.reshape(g, (len(f), len(ycenter)))
    y = np.reshape(y, (len(f), len(ycenter)))
    return (f, g, y, ycenter)

def calcIntercept(f, g, y):
    b = f - np.diag(np.dot(g, y.T))
    return b

# - Lambda_k ||g^t d||^2 t + <d, g(x_k - Lambda_k g^t theta) + b)> at t = 0
def calcDualDerivative0(Lambda, theta, g, d, b, ycenter):
    gtTheta = np.dot(g.T, theta)
    xdash = ycenter - Lambda * gtTheta
    gxPlusB = np.dot(g, xdash) + b
    return np.dot(d, gxPlusB)

def exactLineSearch(Lambda, gtdSquared, d_0):
    return d_0/(gtdSquared*Lambda)

def calcDual(Lambda, theta, ycenter, g, b):
    gtTheta = np.dot(g.T, theta)
    ret = - 0.5*Lambda*np.dot(gtTheta, gtTheta)
    gxPlusB = np.dot(g, ycenter) + b
    ret += np.dot(theta, gxPlusB)
    return ret

def calcPDgap(Lambda, theta, ycenter, g, b):
    gtTheta = np.dot(g.T, theta)
    xdash = ycenter - Lambda * gtTheta
    affines = np.dot(g, xdash) + b
    diff = max(affines) - affines
    ret = np.dot(theta, diff)
    return ret

def coordinateDescent(N, itrmax, Lambda, g, b, ycenter, gamma, logger):
    theta = np.ones(N)/float(N)
    for itr in range(itrmax):
        i = itr % N
        e_i = np.zeros(N)
        e_i[i] = 1.0
        d = e_i - theta
        d_0 = calcDualDerivative0(Lambda, theta, g, d, b, ycenter)
        if np.absolute(d_0) <= 1.0e-8:
            continue
        if d_0 < 0 and np.absolute(theta[i]) <= 1.0e-8:
            continue
        gtd = np.dot(g.T, d)
        gtdSquared = np.dot(gtd, gtd)
        if np.absolute(gtdSquared) <= 1.0e-12:
            continue
        else:
            tmax = 1.0
            tmin = - theta[i]/(1.0 - theta[i])
            t = exactLineSearch(Lambda, gtdSquared, d_0)
            if t >= tmax:
                t = tmax
            if t <= tmin:
                t = tmin
        theta += t*d
        gap = calcPDgap(Lambda, theta, ycenter, g, b)
        if gap <= gamma:
            logger.debug("reach the gap! " + str(itr))
            break
    gap = calcPDgap(Lambda, theta, ycenter, g, b)
    gtTheta = np.dot(g.T, theta)
    xdash = ycenter - Lambda * gtTheta
    affines = np.dot(g, xdash) + b
    mnext = np.dot(theta, affines)
    logger.debug("gap = " + str(gap) + " gamma = " + str(gamma))
    return(xdash, mnext, gtTheta)
    
if __name__ == "__main__":
    Lambda = 1.0
    (f, g, y, ycenter) = createInput("f.csv", "gy.csv", "ycenter.csv", sys.argv[1])
    N = len(f)
    b = calcIntercept(f, g, y)
    (theta, gap) = coordinateDescent(N, Lambda, g, b, ycenter)
    print(theta)
    print(gap)
    
