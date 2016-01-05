__author__ = 'Sony'
import sys
import scipy.optimize, scipy.special
from numpy import *

import pylab
from matplotlib import pyplot, cm
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.mlab as mlab
import os

def plot( data ):
	negatives = data[data[:, 2] == 0]
	positives = data[data[:, 2] == 1]

	pyplot.xlabel("Microchip test 1")
	pyplot.ylabel("Microchip test 2")
	pyplot.xlim([-1.0, 1.5])
	pyplot.ylim([-1.0, 1.5])

	pyplot.scatter( negatives[:,0], negatives[:,1], c='y', marker='o', linewidths=1, s=40, label='y=0' )
	pyplot.scatter( positives[:,0], positives[:,1], c='k', marker='+', linewidths=2, s=40, label='y=1' )

	pyplot.legend()


def mapFeature(X1, X2):
    degree = 6
    out = ones((shape(X1)[0], 1))
    for i in range(1, degree+1):
        for j in range(0, i+1):
            term1 = X1**(i-j)
            term2 = X2 ** (j)
            term  = (term1 * term2).reshape( shape(term1)[0], 1 )
            """note that here 'out[i]' represents mappedfeatures of X1[i], X2[i], ..........   out is made to store features of one set in out[i] horizontally """
            out   = hstack(( out, term ))
    return out


def sigmoid( z ):
	return scipy.special.expit(z)
	# return 1.0 / (1.0 + exp( -z ))


def computeCost( theta, X, y, lamda):
    m = shape(X)[0]
    h = sigmoid(X.dot(theta))
    J = - ( log(h).T.dot(y) + log(1 - h).T.dot(1 - y))
    """MORE PRECISELY -- J += (lamda/2)*(theta[1:].T.dot(thetatheta[1:]))"""
    J += (lamda/2)*(theta.T.dot(theta))
    return J/m


def gradCost(theta, X, y, lamda):
    m = shape(X)[0]
    error = sigmoid(X.dot(theta)) - y
    grad = X.T.dot(error)
    grad[1:] += lamda*(theta[1:])
    return grad/m


def costFunction( theta, X, y, lamda ):
        cost 	 = computeCost( theta, X, y, lamda )
        gradient = gradCost( theta, X, y, lamda )
        return cost


def findMinTheta( theta, X, y, lamda ):
	result = scipy.optimize.minimize( costFunction, theta, args=(X, y, lamda),  method='BFGS', options={"maxiter":500, "disp":True} )
	return result.x, result.fun


"""http://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.optimize.fmin.html"""
# def findMinTheta( theta, X, y ):
# 	result = scipy.optimize.fmin( computeCost, x0=theta, args=(X, y), maxiter=500, full_output=True )	#Minimize a function using the downhill simplex algorithm.This algorithm only uses function values, not derivatives or second derivatives.
# 	return result[0], result[1]	#read the link or google - 'scipy.optimize.fmin - returns'


def part2_1():
	data  = genfromtxt( os.getcwd() + '\ex2data2.txt', delimiter = ',' )
	plot( data )
	pyplot.show()


def part2_2():
	data  = genfromtxt( os.getcwd() + '\ex2data2.txt', delimiter = ',' )
	X 	  = mapFeature( data[:, 0], data[:, 1] )
	# print (X)


def part2_3():
    data  = genfromtxt( os.getcwd() + '\ex2data2.txt', delimiter = ',' )
    y 	  = data[:,2]
    X 	  = mapFeature( data[:, 0], data[:, 1] )
    theta = zeros(shape(X)[1] )
    lamda = 1.0
    print("theta found using lamda = 1 :")
    print(computeCost( theta, X, y, lamda ))

    theta, cost = findMinTheta( theta, X, y, lamda )


def part2_4():
    data  = genfromtxt( os.getcwd() + '\ex2data2.txt', delimiter = ',' )
    y 	  = data[:,2]
    X 	  = mapFeature( data[:, 0], data[:, 1] )
    theta = zeros(shape(X)[1] )
    lamdas = [0, 1, 100]

    for lamda in lamdas:
            theta, cost = findMinTheta( theta, X, y, lamda )
            pyplot.text( 0.15, 1.4, 'Lamda %.1f' % lamda )
            plot( data )

            x1, x2 = linspace(-1, 1.5, 50), linspace(-1, 1.5, 50)
            x = zeros((len(x1), len(x2)))

            for i in range (0, len(x1)):
                for j in range(0, len(x2)):
                    map = mapFeature(array([x1[i]]), array([x2[j]]))
                    x[i][j] = map.dot(theta)

            x = x.transpose();
            x1, x2 = meshgrid(x1, x2)
            pyplot.contour( x1, x2, x, [0.0, 0.0], label='Decision Boundary' )
            pyplot.show()


def main():
	set_printoptions(precision=6, linewidth=200)
	part2_1()
	part2_2()
	part2_3()
	part2_4()


if __name__ == '__main__':
	main()