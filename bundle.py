import sys
import os
import re
import numpy as np
import scipy as sp
from numpy.random import *
import time
import param
import problems
from dualCoordinate import *
from copy import deepcopy
import cvxopt
import logging
from logging import getLogger, StreamHandler, Formatter

def descentTest(f, fnext, mnext, gamma, m):
    if f - fnext >= 0:
        return f - fnext >= m*(f - mnext + gamma)
    else:
        return False

def ipmByCvxopt(ycenter, b, g, Lambda, gamma, logger):
    n = len(ycenter)
    P = cvxopt.matrix(np.eye(n+1))/Lambda
    P[n,n] = 0.0
    ycenterPlus = np.insert(-ycenter/Lambda, n, 1.0)
    q = cvxopt.matrix(ycenterPlus)
    objAdded = 0.5*np.dot(ycenter, ycenter)
    k = len(b)
    A = cvxopt.matrix(np.hstack((g, - np.ones((k,1)))))
    bb = cvxopt.matrix(-b)
    cvxopt.solvers.options['show_progress'] = False
    cvxopt.solvers.options['abstol'] = gamma
    sol = cvxopt.solvers.qp(P, q, G=A, h=bb)
    logger.debug("iteration: %s" % sol['iterations'])
    yPlusR = np.array(sol['x']).flatten()
    gAggregatedTmp = np.dot(g.T, sol['z'])
    gAggregated = gAggregatedTmp.flatten()
    affineValues = np.dot(g, yPlusR[0:n]) + b
    return (yPlusR[0:n].tolist(), np.dot(affineValues, sol['z'])[0], gAggregated.tolist())

def isOptimal(fcenter, fnext, mnext, eps, n, ynext, ycenter, gamma, logger):    
    logger.debug("delta: " + str(fcenter - mnext + gamma)
                 + " fcenter: " + str(fcenter)
                 + " fnext: " + str(fnext)
                 + " mnext: " + str(mnext)
                 + " gamma: " + str(gamma))
    if fcenter - mnext + gamma <= 0:
        logger.debug("Negative delta detected!")
    deltaModified = np.absolute(fcenter - mnext + gamma)/(1.0 + np.absolute(fcenter))
    ydiff = np.array(ynext) - np.array(ycenter)
    logger.debug("delta/(1 + fcenter): " + str(deltaModified))
    return (deltaModified <= eps, deltaModified)

def frandomize(fnextLower, eta, isOracleRandomized):
    np.random.seed(1)
    if isOracleRandomized:
        fnextLower = fnextLower - eta*random()
    return fnextLower

def solve(prob, method, logger):
    print("Problem_Name                    %30s" % prob.name())
    status = False
    isIpm = False
    isCoordinate = False
    isExact = True
    isOracleRandomized = False
    objType = 0
    timeModelMin = 0.0
    coordItrMax = 100000
        
    para = param.Param()
    para = prob.setParam(para)
    if method == "ipm_exact":
        (isIpm, isExact) = (True, True)
    elif method == "ipm_inexact":
        (isIpm, isExact) = (True, False)
        (gamma, tau) = (para.gammaInit, 0.9)
    elif method == "cda_exact":
        (isCoordinate, isExact) = (True, True)
    elif method == "cda_inexact":
        (isCoordinate, isExact) = (True, False)
        (gamma, tau) = (para.gammaInit, 0.9)
    elif method == "cda_special":
        (isCoordinate, isExact) = (True, False)
        (gamma, tau) = (para.gammaInit, 0.8)
        coordItrMax = 100
    else:
        print("Method %s does not exist!" % method)
        sys.exit(10)
    imax = para.imax
    n = prob.ndim()
    Lambda = para.Lambda
    LambdaMin = para.LambdaMin
    eps = para.eps
    bmax = para.bmax
    gammaMin = para.gammaMin
    isOracleRandomized = para.isOracleRandomized
    # isDebug = para.isDebug
    if isExact == True:
        (gamma, tau) = (para.gammaMin, 1.0)        
    if isOracleRandomized == True:
        (eta, etau) = (para.etaInit, para.etau)
    else:
        (eta, etau) = (0.0, 1.0)

    # bundle information
    (y, f, g, b) = ([], [], [], [])
    gAggregated = []
    startTime = time.time()
    ycenter = prob.init()
    ynext = []
    (fcenter, gcenter) = prob.calcBoth(ycenter)
    if isOracleRandomized == True:
        fcenterLower = frandomize(fcenter, eta, isOracleRandomized)
    else:
        fcenterLower = fcenter
    fcenterUpper = fcenterLower + eta
    badd = fcenterLower - np.dot(np.array(ycenter), np.array(gcenter))
    for (ary, body) in zip([y,f,g,b], [ycenter,fcenterLower,gcenter,badd]):
        ary.append(body)        
    # It judges wether previous step is serious(True) or null(False)
    isBeforeSerious = False
    seriousStep = 0
    print("Preperation end.")
    sys.stdout.flush()
    
    for i in range(imax):
        startModelMin = time.time()
        if isCoordinate:
            (ynext, mnext, gAggregated) = coordinateDescent(len(y), coordItrMax, Lambda, np.array(g), np.array(b), np.array(ycenter), gamma, logger)
        elif isIpm:
            (ynext, mnext, gAggregated) = ipmByCvxopt(np.array(ycenter), np.array(b), np.array(g), Lambda, gamma, logger)
        endModelMin = time.time()
        timeModelMin += endModelMin - startModelMin
        (fnext, gnext) = prob.calcBoth(ynext)
        if isOracleRandomized == True:
            fnextLower = frandomize(fnext, eta, isOracleRandomized)
        else:
            fnextLower = fnext
        fnextUpper = fnextLower + eta
        eta *= etau
        bnext = fnextLower - np.dot(np.array(ynext), np.array(gnext))
        # Bundle management
        while len(y) >= bmax - 1:
            y.pop(0)
            f.pop(0)
            g.pop(0)
            b.pop(0)
        (optimal, deltaModified) = isOptimal(fcenterUpper, fnext, mnext, eps, n, ynext, ycenter, gamma, logger)
        if optimal:
            status = True
            break
        if descentTest(fcenterUpper, fnextUpper, mnext, gamma, 1.0e-3):
            logger.debug("Serious Step at %d" % i)
            if i%10 == 0:
                print("iteration %4d, delta = %g" % (i, deltaModified))
                sys.stdout.flush()
            seriousStep += 1
            # Enlarge Lambda if it isexpected to be too small
            if descentTest(fcenterUpper, fnextUpper, mnext, gamma, 1.0e-1):
                if isBeforeSerious == False:
                    Lambda = para.Lambda
                logger.debug("Enlarge Lambda!")
                Lambda *= 2.0
            isBeforeSerious = True            
            ycenter = ynext
            (fcenter, fcenterLower, fcenterUpper) = (fnext, fnextLower, fnextUpper)
            # bundle update
            for (ary, body) in zip([y,f,g,b], [ynext,fnextLower,gnext,bnext]):
                ary.append(body)    
        else:
            logger.debug("Null Step at %d" % i)
            if i%10 == 0:
                print("iteration %4d, delta = %g" % (i, deltaModified))
                sys.stdout.flush()
            isBeforeSerious = False
            if gamma >= gammaMin:
                gamma *= tau
            bAggregated = mnext - np.dot(np.array(gAggregated), np.array(ynext))
            # bundle update
            for (ary, body, aggregated) in zip([y,f,g,b], [ynext,fnextLower,gnext,bnext], [ynext,mnext,gAggregated,bAggregated]):
                ary.append(body)
                ary.append(aggregated)
            if Lambda >= LambdaMin:
                Lambda *= 0.8
    elapsedTime = time.time() - startTime
    return (fcenter, ycenter, i, seriousStep, status, elapsedTime, timeModelMin)

def printBoth(wf1, wf2, str):
    wf1.write(str)
    wf2.write(str)

if __name__ == "__main__":
    testset = problems.testset
    para = param.Param()
    mode = para.mode
    methods = para.methods

    # Preparation for logging
    logger = getLogger("bundle")
    stream_handler = StreamHandler()
    if para.isDebug == True:
        logger.setLevel(logging.DEBUG)
        stream_handler.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.ERROR)
        stream_handler.setLevel(logging.ERROR)
    # handler_format = Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler_format = Formatter('- %(levelname)s - %(message)s')
    stream_handler.setFormatter(handler_format)
    logger.addHandler(stream_handler)
    
    if mode == "develop":
        sf = time.strftime("_%Y_%m_%d_%H_%M", time.localtime())
        with open(os.path.join("output", "output" + sf + ".csv"), "w") as wf:
            wf.write("method,,")
            for method in methods:
                wf.write("%s,,," % method)
            wf.write("\n")
            wf.write("Problem, number of variables")
            for method in methods:
                wf.write(",OC(SC), CPU(MM), status")
            wf.write("\n")
            for prob in testset:
                print(prob.name(), method)
                wf.write("%s,%d" % (prob.name(), prob.ndim()))
                for method in methods:
                    (fcenter, ycenter, itr, seriousItr, status, elapsedTime, timeModelMin) = solve(prob, method, logger)
                    if status == True:
                        statusStr = "T"
                    else:
                        statusStr = "F"
                    wf.write(",%d(%d),%.1f(%.1f),%s" % (itr, seriousItr, elapsedTime, timeModelMin, statusStr))
                wf.write("\n")
    if mode == "normal":
        for prob in testset:
            for method in methods:
                (fcenter, ycenter, itr, seriousItr, status, elapsedTime, timeModelMin) = solve(prob, method, logger)
                if status == True:
                    statusStr = "Optimal"
                else:
                    statusStr = "Non_Optimal"
                print("[result]")
                with open(prob.name() + ".sol", "w") as wf:
                    printBoth(sys.stdout, wf, ("Problem_Name                    %30s\n" % prob.name()))
                    printBoth(sys.stdout, wf, ("Number_Of_Variables             %30d\n" % prob.ndim()))
                    printBoth(sys.stdout, wf, ("Objective_Value                 %30.6g\n" % fcenter))
                    printBoth(sys.stdout, wf, ("Status                          %30s\n" % statusStr))
                    printBoth(sys.stdout, wf, ("Oracle_Calls                    %30d\n" % itr))
                    printBoth(sys.stdout, wf, ("Serious_Steps                   %30d\n" % seriousItr))
                    printBoth(sys.stdout, wf, ("Model_Minimization_Algorithm    %30s\n" % method))
                    printBoth(sys.stdout, wf, ("Elapsed_Time(sec)               %30.2f\n" % elapsedTime))
                    printBoth(sys.stdout, wf, ("Model_Minimization_Time(sec)    %30.2f\n" % timeModelMin))
                    printBoth(sys.stdout, wf, ("Optimal_Point:\n"))
                    n = prob.ndim()
                    for i in range(n):
                        printBoth(sys.stdout, wf, ("[%4d]                          %30.6g\n" % (i, ycenter[i])))
