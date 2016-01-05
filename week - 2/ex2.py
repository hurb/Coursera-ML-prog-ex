__author__ = 'Sony'

import sys
import scipy.optimize, scipy.special
from numpy import *
from matplotlib import pyplot, cm
from mpl_toolkits.mplot3d import Axes3D
import os


def plot(data):
    """positives stores entire row, whose element at index 2 is 1"""
    # print(data[:, 2] == 1)
    positives = data[data[:, 2] == 1]
    """negatives stores entire row, whose element at index 2 is 0"""
    negatives = data[data[:, 2] == 0]


    pyplot.xlabel("Exam 1 Score")
    pyplot.ylabel("Exam 2 Score")
    pyplot.xlim([25, 110])
    pyplot.ylim([25, 110])
    pyplot.scatter(positives[:, 0], positives[:, 1], c='b', marker='+', s=40, linewidths=1, label="Not admitted" )
    pyplot.scatter(negatives[:, 0], negatives[:, 1], c='r', marker='x', s=40, linewidths=1, label="Admitted")
    pyplot.legend()


def plotBoundary(data, X, theta):
    plot(data)
    plot_x = array([min(X[:, 1]), max(X[:, 1])])

    """ hypothesis = theta[0] + X[1]*theta[1] + theta[2]*theta[2] + . . . ., put this equal to zero to find the decision boundary, just like the way in lectures"""
    plot_y = (-1./ theta[2]) * (theta[1] * plot_x + theta[0])
    pyplot.plot( plot_x, plot_y )


def hypo(X, theta):
    return X.dot(theta)


def sigmoid(z):
    return scipy.special.expit(z)       # OR   -->    return 1.0 / (1.0 + exp( -z ))


def computeCost( theta, X, y):
    m = shape(X)[0]
    h = sigmoid(hypo(X, theta))
    J = - ( log(h).T.dot(y) + log(1 - h).T.dot(1 - y))
    return J/m


def gradientCost( theta, X, y ):
	m = shape(X)[0]
	return ( X.T.dot(sigmoid( X.dot( theta ) ) - y)  ) / m


"""http://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.optimize.fmin.html"""
def findMinTheta( theta, X, y ):
	result = scipy.optimize.fmin( computeCost, x0=theta, args=(X, y), maxiter=500, full_output=True )	#Minimize a function using the downhill simplex algorithm.This algorithm only uses function values, not derivatives or second derivatives.
	return result[0], result[1]	#read the link or google - 'scipy.optimize.fmin - returns'


"""classify test data according to theta learned"""
def predict( theta, X, binary=True ):
	prob = sigmoid( theta.dot( X ))
	if binary :
		return 1 if prob > 0.5 else 0
	else:
		return prob


def part2_1():
    data = genfromtxt(os.getcwd() + "\ex2data1.txt", delimiter=',')
    plot(data)
    pyplot.show()


def part2_2():
    data = genfromtxt(os.getcwd() + "\ex2data1.txt", delimiter=',')
    m, n = shape(data)[0], shape(data)[1]
    X = c_[ones((m, 1)), data[:, : n-1]]
    y = data[:, n-1 : n]
    theta = zeros((n, 1))

    print (computeCost(theta, X, y))
    theta, cost = findMinTheta( theta, X, y )
    plotBoundary( data, X, theta )
    pyplot.show(block = True)

    test = array([1, 45, 85])
    print ('prediction for [1, 45, 85] : ' + str(predict( test, theta )))

def main():
    part2_1()
    part2_2()


if __name__ == '__main__':
	main()