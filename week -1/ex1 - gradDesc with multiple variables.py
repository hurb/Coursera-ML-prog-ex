__author__ = 'Sony'

'''Standard python modules'''
import sys

'''For scientific computing'''
from numpy import *
import scipy.optimize

'''For plotting'''
from matplotlib import pyplot, cm
from mpl_toolkits.mplot3d import Axes3D

import os
# print(os.path.dirname(os.path.realpath(__file__)))
# print(os.getcwd())


def hypo(X, theta):
    return X.dot(theta)


def featureNormalizeLoop(X):
    mu = []
    sigma = []
    norm_data = zeros(shape(X), X.dtype)

    for col in range (0, shape(X)[1]):
        mu.append(mean(X[:, col]))
        """ if ddof = 0 sigma will be uncorrected sample standard deviation"""
        sigma.append(std(X[:, col], ddof=1))
        norm_data[:, col] = list(map( lambda x: (x - mu[col]) / sigma[col], X[:, col] ))

    return norm_data, array(mu), array(sigma)


def featureNormalizeVect(X):
    mu = mean(X, axis=0)
    sigma = std(X, axis=0, ddof=1)
    norm_data = (X - mu)/sigma
    return norm_data, mu, sigma


def computeCostLoop(X, y, theta):
    """Compute cost, loop version"""
    m = shape(X)[0]
    cum_sum = 0
    for i in range (0, m):
        error = hypo(X[i], theta) - y[i]
        cum_sum += error**2

    return (1/(2*m))*cum_sum


def computeCostVect(X, y, theta):
    """Compute cost, vectorized version"""
    m 	 = len(y)
    term = hypo(X, theta) - y
    return (term.T.dot(term) / (2 * m))[0, 0] #?? [0, 0] ??


def gradDescVect(X, y, theta, alpha, iters):
    """Run a Vectorized gradient descent"""
    m = shape(X)[0]
    grad = copy(theta)
    J_history = []
    J_history.append(computeCostVect(X, y, theta))
    for c in range(1, iters):
        error_sum = hypo(X, grad) - y
        error_sum = X.T.dot(error_sum)
        grad -= (alpha/m)*error_sum
        # print(computeCostVect(X, y, grad))
        J_history.append(computeCostVect(X, y, grad))

    return J_history, grad


def part2_1():
    data = genfromtxt(os.getcwd() + "\ex1data2.txt", delimiter = ',')
    n = shape(data)[1]
    print(n)
    X, y = data[:, 0:n-1], data[:,n-1 : n]
    m = shape(X)[0]
    y = y.reshape(m, 1)

    # norm_data, mu, sigma = featureNormalizeLoop(X)
    norm_data, mu, sigma = featureNormalizeVect(X)
    norm_data = c_[ones((m, 1)), norm_data]
    # print(norm_data)
    # print(mu)
    # print(sigma)


def part2_2():
    data = genfromtxt(os.getcwd() + "\ex1data2.txt", delimiter=',')
    X = data[:, 0:2]
    y = data[:, 2:3]
    m = shape(X)[0]
    n = shape(X)[1]

    X, mu, sigma = featureNormalizeVect(X)
    X = c_[ones((m, 1)), X]

    alphas = [0.01, 0.03, 0.1, 0.3, 1]
    iters = 1500

    for alpha in alphas:
        print("results using alpha = " + str(alpha))
        theta = zeros((n + 1, 1))
        initialCost = computeCostVect(X, y, theta)
        J_history, theta = gradDescVect(X, y, theta, alpha, iters)
        print("theta : " + str(theta))
        print("cost : " + str(computeCostVect(X, y, theta)))

        # 1650 sq feet 3 bedroom house
        test = array([1.0, 1650.0, 3.0])
        """ test data must also be feature Normalized"""
        # but exclude intercept units
        test[1:] = (test[1:] - mu) / sigma
        print("prediction for 1650 sq feet 3 bedroom house (alpha used " + str(alpha) + ": ")
        print(test.dot( theta ))

        number_of_iters = array( [x for x in range( 1, iters + 1 )] ).reshape( iters, 1)
        pyplot.text(20, initialCost*0.95, 'alpha : %f' % alpha)
        pyplot.plot(number_of_iters, J_history, '-b')
        pyplot.xlim( [0, 50] )
        pyplot.xlabel("no. of iters")
        pyplot.ylabel("cost - J(theta)")
        pyplot.show()


"""     COST FUNCTION FOR SCIPY.OPTIMZE.MINIMIZE()      """
def computeCostScipy(theta, X, y):
    """Compute cost, vectorized version"""
    m 	 = len(y)
    term = hypo(X, theta).reshape(m, 1) - y
    return (term.T.dot(term) / (2 * m))[0, 0] #?? [0, 0] ??


def findMinTheta(theta, X, y):
    result = scipy.optimize.minimize( computeCostScipy, theta, args=(X, y),  method='BFGS', options={"maxiter":5000, "disp":True} )
    return result.x, result.fun


def part2_3():
    data = genfromtxt(os.getcwd() + "\ex1data2.txt", delimiter=',')
    X = data[:, 0:2]
    y = data[:, 2:3]
    m = shape(X)[0]
    n = shape(X)[1]

    X, mu, sigma = featureNormalizeVect(X)
    X = c_[ones((m, 1)), X]
    theta = zeros((n+1, 1))
    ausi = computeCostVect(X, y, theta)
    print("Results usning BFGS : ")
    theta, cost = findMinTheta(theta, X, y)
    print("theta : " + str(theta))
    print("cost : " + str(computeCostVect(X, y, theta)))

    # 1650 sq feet 3 bedroom house
    test = array([1.0, 1650.0, 3.0])
    """ test data must also be feature Normalized"""
    # but exclude intercept units
    test[1:] = (test[1:] - mu) / sigma
    print("prediction for 1650 sq feet 3 bedroom house (using BFGS : ")
    print(test.dot( theta ))


def main():
    part2_1()
    part2_2()
    part2_3()
if __name__ == '__main__':
    main()