__author__ = 'Sony'

'''Standard python modules'''
import sys

'''For scientific computing'''
from numpy import *
# from scipy import *
import scipy.optimize

'''For plotting'''
from matplotlib import pyplot, cm
from mpl_toolkits.mplot3d import Axes3D
import pylab

import os
# print(os.path.dirname(os.path.realpath(__file__)))
# print(os.getcwd())

def plot(X, y):
    pyplot.plot(X, y, 'rx', markersize = 5)
    pyplot.ylabel('Profit in $10,000s')
    pyplot.xlabel('Population of City in 10,000s')

def hypo(X, theta):
    return X.dot(theta)


def computeCostLoop(X, y, theta):
    """Compute cost, loop version"""
    m = shape(X)[0]
    cum_sum = 0
    for i in range (0, m):
        error = hypo(X[i], theta) - y[i]
        cum_sum += error**2

    return (1/(2*m))*cum_sum


def computeCost(X, y, theta):
    """Compute cost, vectorized version"""
    m 	 = len(y)
    term = hypo(X, theta) - y
    return (term.T.dot(term) / (2 * m))[0, 0]         #?? [0, 0] ?? now that's pro!!


def gradDescLoop(X, y, theta, alpha, iters):
    grad = copy(theta)
    n = shape(X)[1]
    m = len(y)

    for i in range (0, iters):
        temp = [0 for x in range(0, n)]
        for j in range(0, n):
            for k in range(0, m):
                temp[j] = temp[j] + (hypo(X[k], grad) - y[k])*X[k][j];

        for j in range(0, n):
            grad[j] = grad[j] - (alpha/m)*(temp[j])
        # print(computeCostLoop(X, y, grad))

    return grad


def gradDescVect(X, y, theta, alpha, iters):
    m = shape(X)[0]
    grad = copy(theta)

    for c in range(0, iters):
        error_sum = hypo(X, grad) - y
        error_sum = X.T.dot(error_sum)
        grad -= (alpha/m)*error_sum

    return grad


def part1_1():
    data = genfromtxt(os.getcwd() + "\ex1data1.txt", delimiter = ',')
    X, y = data[:, 0], data[:, 1]
    m = shape(X)[0]
    plot(X, y)
    pyplot.show()


def part1_2():
    data = genfromtxt(os.getcwd() + "\ex1data1.txt", delimiter = ',')
    X, y = data[:, 0], data[:, 1]
    m = shape(X)[0]
    X = c_[ones((m, 1)), X]
    y = y.reshape(m, 1)
    alpha = 0.01
    iters = 1500

    # theta = zeros((2, 1))
    # theta = gradDescLoop(X, y, theta, alpha, iters)
    # print("theta found using gradDecsLoop() : ")
    # print(theta)

    theta = zeros((2, 1))
    # should be 32.07
    cost 	= computeCost(X, y, theta)
    print(cost)

    theta = gradDescVect(X, y, theta, alpha, iters)
    print("********************************")
    print("Results found using gradDecsVect() : ")

    cost 	= computeCost(X, y, theta)
    print("Cost : " + str(cost))
    print("Theta : " + str(theta))

    predict1 = array([1, 3.5]).dot(theta)
    predict2 = array([1, 7]).dot(theta)
    print ('prediction for [1, 3.5] : ' + str(predict1))
    print ('prediction for [1, 7] : ' + str(predict2))
    pyplot.text( 25, 38, 'alpha %f' % alpha )
    plot(X[:, 1], y)
    pyplot.plot(X[:, 1], X.dot(theta), 'b-', linewidth=0.1)
    pyplot.show(block = True)


def part1_3():
    data = genfromtxt(os.getcwd() + "/ex1data1.txt", delimiter=',')
    X, y = data[:, 0], data[:, 1]
    m 	 = len(y)
    y 	 = y.reshape(m, 1)
    X 	 = c_[ones((m, 1)), X]
    alpha = 0.01
    iterations = 1500

    theta0_vals = linspace(-10, 10, 500)
    theta1_vals = linspace(-4, 4, 500)

    J_vals = zeros((len(theta0_vals), len(theta1_vals)), dtype=float64)
    for i, v0 in enumerate(theta0_vals):
        for j, v1 in enumerate(theta1_vals):
            theta 		 = array((theta0_vals[i], theta1_vals[j])).reshape(2, 1)
            J_vals[i, j] = computeCost(X, y, theta)

    R, P = meshgrid(theta0_vals, theta1_vals)

    fig = pyplot.figure()
    ax 	= fig.gca(projection='3d')
    ax.plot_surface(R, P, J_vals)
    pyplot.show(block=True)

    theta = gradDescVect(X, y, theta, alpha, iterations) #find theta which minimizes errors J, this theta is ueful in code on line#161

    fig = pyplot.figure()
    # ax 	= fig.gca(projection='3d')
    pyplot.contour(R, P, J_vals.T, logspace(-2, 3, 20))
    pyplot.plot(theta[0], theta[1], 'rx', markersize = 10)
    pyplot.show(block=True)



def computeCostScipy(theta, X, y):
    """Compute cost, vectorized version"""
    m 	 = len(y)
    term = hypo(X, theta).reshape(m, 1) - y
    # print((term.T.dot(term) / (2 * m))[0, 0])
    return (term.T.dot(term) / (2 * m))[0, 0] #?? [0, 0] ??


""" using Simplex Downhill Algo to avoid Gradient Desc """
def findMinThetaSimplexDownhill(theta, X, y):
        result = scipy.optimize.fmin( computeCostScipy, x0=theta, args=(X, y), maxiter=500, full_output=True )	#Minimize a function using the downhill simplex algorithm.This algorithm only uses function values, not derivatives or second derivatives.
        return result[0], result[1]	#read the link or google - 'scipy.optimize.fmin - returns'

""" nonlinear conjugate gradient algorithm """
def run(theta, X, y ):
    result  = scipy.optimize.fmin_cg( computeCostScipy, x0=theta,  \
                                        args = (X, y), maxiter=100, disp=False, full_output=True )
    return result[0], result[1]


"""     **********              WOKING  BFGS                *********   """
def findMinThetaBFGS( theta, X, y):
        result = scipy.optimize.minimize( computeCostScipy, theta, args=(X, y),  method='BFGS', options={"maxiter":5000, "disp":True} )
        return result.x, result.fun


"""         USING OPTIMIZING ALGOS IMPLEMENTED IN SCIPY TO AVOID ITERATTIONS, HAVING TO CHOOSE ALPHA, NO. OF ITERS IN  GRADIENT DESCENT     """
def part1_4():
    data = genfromtxt(os.getcwd() + "/in2.txt", delimiter=',')
    X, y = data[:, 0], data[:, 1]
    m 	 = len(y)
    y 	 = y.reshape(m, 1)
    X 	 = c_[ones((m, 1)), X]

    theta = zeros((shape(X)[1], 1))
    print("***********************************")
    print("Results using BFGS Algo in Scipy module :")
    theta, cost = findMinThetaBFGS(theta, X, y)
    # cost 	= computeCostScipy(theta, X, y)
    # should be 32.07
    print("Cost : " + str(cost))
    print("Thteta : " + str(theta))

    predict1 = array([1, 3.5]).dot(theta)
    predict2 = array([1, 7]).dot(theta)
    print ('prediction for [1, 3.5] : ' + str(predict1))
    print ('prediction for [1, 7] : ' + str(predict2))

    plot(X[:, 1], y)
    pyplot.plot(X[:, 1], X.dot(theta), 'b-', linewidth=0.1)
    pyplot.show(block = True)

    print("***********************************")
    print("Results using Simplex Downhill Algo in Scipy module :")
    theta, cost = findMinThetaSimplexDownhill(theta, X, y)
    print("Cost : " + str(cost))
    print("Thteta : " + str(theta))

    predict1 = array([1, 3.5]).dot(theta)
    predict2 = array([1, 7]).dot(theta)
    print ('prediction for [1, 3.5] : ' + str(predict1))
    print ('prediction for [1, 7] : ' + str(predict2))

    plot(X[:, 1], y)
    pyplot.plot(X[:, 1], X.dot(theta), 'b-', linewidth=0.1)
    pyplot.show(block = True)


def main():
    # part1_1()
    # part1_2()
    # part1_3()
    part1_4()
if __name__ == '__main__':
    main()