__author__ = 'Sony'


from math import sqrt
from numpy import *
from os import getcwd
import scipy.misc, scipy.io, scipy.optimize, scipy.special
from scipy.misc import toimage
from matplotlib import pyplot
import pylab


def unrollParams( nn_params, input_layer_size, hidden_layer_size, num_labels):
    # print("here we are : looking at shape of nn_params")
    # print(shape(nn_params))
    theta1_elems = ( input_layer_size + 1 ) * hidden_layer_size
    theta1_size  = ( input_layer_size + 1, hidden_layer_size  )
    theta2_size  = ( hidden_layer_size + 1, num_labels )

    theta1 = nn_params[:theta1_elems].T.reshape( theta1_size ).T
    theta2 = nn_params[theta1_elems:].T.reshape( theta2_size ).T

    return (theta1, theta2)


def displayData(X, theta1=None, theta2=None):
    m, n = shape(X)
    width = sqrt(n)     #We wish to plot a square image from a feature vector of size n
    rows, cols = 10, 10
    out_image = zeros((rows*width, cols*width))
    rand_indices = random.permutation(m)[0 : rows*cols]
    counter=0

    for i in range (rows):
        for j in range (cols):
            out_image[i*width:(i+1)*width, j*width:(j+1)*width] = X[rand_indices[counter]].reshape(width, width).T
            counter += 1

    img = scipy.misc.toimage(out_image)
    fig = pyplot.figure()
    axes = fig.add_subplot(111)
    axes.imshow(img)

    if theta1 is not None and theta2 is not None:
        result_mat = predict(X[rand_indices, :], theta1, theta2)
        print(result_mat.reshape(rows, cols))

    pyplot.show()


def sigmoid(z):
    return scipy.special.expit(z)


def sigmoidGradient( z ):
	sig = sigmoid(z)
	return sig * (1 - sig)


def feedForward(X, y, theta1, theta2, X_bias):
    m, n = shape(X)
    a1 = c_[ones((m, 1)), X] if X_bias is None else X_bias        # m X 3
    z2 = a1.dot(theta1.T)                 # m X 2
    a2 = sigmoid(z2)
    a2 = c_[ones((m, 1)), a2]      # m X 3
    z3 = a2.dot(theta2.T)                # m X 2
    a3 = sigmoid(z3)
    return (a1, a2, a3, z2, z3)


def costFunc(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, lamda, X_bias=None, yk=None):
    m, n = shape(X)
    J = 0
    K = num_labels
    """IMPORTANT  :  THIS DEFINES THE HOW OUR NEURAL NETWORK LABELS THE OUTPUT DATA"""
    # y-index           :       0   1   2   3   4   5   6   7   8   9
    #label/output   :       1   2   3   4   5   6   7   8   9   10
    if yk is None:
        yk = zeros((m, K))
        for i in range(0, m):
            yk[i, y[i]-1] = 1

    theta1, theta2 = unrollParams(nn_params, input_layer_size, hidden_layer_size, num_labels)

    layers = feedForward(X, y, theta1, theta2, X_bias)
    a3 = layers[2]

    term1 		= -yk * (log( a3 ))
    term2 		= (1-yk) * (log( 1 - a3))
    J_unreg 	= sum(term1 - term2) / m
    reg_term = (lamda/(2*m))*(sum(theta1[:, 1:] **2) + sum(theta2[:, 1:] **2))

    J = J_unreg + reg_term

    return J

def gradFunc(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, lamda, X_bias, yk = None):
    m, n = shape(X)
    theta1, theta2 = unrollParams(nn_params, input_layer_size, hidden_layer_size, num_labels)

    # STEP 1 : FEED FWD THE NEURAL NET WITH a1 = X, For the hidden layers,
    a1 = c_[ones((m, 1)), X]      if X_bias is None else X_bias           # m X 3
    z2 = a1 . dot(theta1.T)       # m X 2
    a2 = sigmoid(z2)
    a2 = c_[ones((m, 1)), a2]                      # m X 3
    z3 = a2 . dot(theta2.T)      # m X 2
    a3 = sigmoid(z3)

    if yk is None:
        yk = zeros((m, num_labels))
        for i in range(0, m):
            yk[i, y[i]] = 1

    #STEP 2 : FIND DELTA FOR OUTPUT LAYER         IMPORTANT  """while training the neural network, this is how NN knows that faeatre vector i is to be labeled as y[i], since we add the cost of rest of the lables but y[i] """
    delta3 = a3 - yk       # m X 2

    #STEP 3 : FIND DELTA FOR HIDDEN LAYER
    delta2 = (delta3 . dot(theta2) ) * c_[ones((m, 1)), sigmoidGradient(z2)]      # (m X 2) . (2 X 3) * (m X 3)
    delta2 = delta2[:, 1:]                  # (m X 2)Taking care of delta0 (correspondig to bias unit) in the hidden layer

    #STEP 4 : ACCUMULATE GRADIENT FOR EACH LAYER, FOR THIS EXAMPLE
    theta1_grad = (delta2.T ) . dot (a1) /m      # (2 X m) X (m X 3) = shape(theta1)
    theta2_grad = (delta3.T ) . dot (a2)  /m    # (2 X m) X (m X 3)   = shape(theta2)

    theta1_grad[:, 1:] += (theta1[:,1:] * lamda)/m
    theta2_grad[:, 1:] += (theta2[:,1:] * lamda)/m

    g = array([theta1_grad.T.reshape(-1).tolist() + theta2_grad.T.reshape(-1).tolist()]).T

    return g.flatten()


"""THIS FUNCTION IS PERFECT !!, RETURNS COST AND GRADIENT CORRECTLY """
def NeuralNetCostFunc(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, lamda, X_bias=None, yk=None):
    m, n = shape(X)
    J = 0
    K = num_labels
    """IMPORTANT  :  THIS DEFINES THE HOW OUR NEURAL NETWORK LABELS THE OUTPUT DATA"""
    # y-index           :       0   1   2   3   4   5   6   7   8   9
    #label/output   :       1   2   3   4   5   6   7   8   9   10
    if yk is None:
        yk = zeros((m, K))
        for i in range(0, m):
            yk[i, y[i]-1] = 1

    theta1, theta2 = unrollParams(nn_params, input_layer_size, hidden_layer_size, num_labels)

    layers = feedForward(X, y, theta1, theta2, X_bias)
    a3 = layers[2]

    term1 		= -yk * (log( a3 ))
    term2 		= (1-yk) * (log( 1 - a3))
    J_unreg 	= sum(term1 - term2) / m
    reg_term = (lamda/(2*m))*(sum(theta1[:, 1:] **2) + sum(theta2[:, 1:] **2))

    J = J_unreg + reg_term


    """    VECTORIZED  BACKPROPAGATION"""
    #ACCORDING TO THE PDF y_predict[frist] represents 1, y_p[second] represents 2, . . . . . .  y_p[10] represents 10
    """ for this use yk (defined above)"""

    m, n = shape(X)
    theta1, theta2 = unrollParams(nn_params, input_layer_size, hidden_layer_size, num_labels)

    # STEP 1 : FEED FWD THE NEURAL NET WITH a1 = X, For the hidden layers,
    a1 = c_[ones((m, 1)), X]      if X_bias is None else X_bias           # m X 3
    z2 = a1 . dot(theta1.T)       # m X 2
    a2 = sigmoid(z2)
    a2 = c_[ones((m, 1)), a2]                      # m X 3
    z3 = a2 . dot(theta2.T)      # m X 2
    a3 = sigmoid(z3)

    if yk is None:
        yk = zeros((m, num_labels))
        for i in range(0, m):
            yk[i, y[i]] = 1

    #STEP 2 : FIND DELTA FOR OUTPUT LAYER         IMPORTANT  """while training the neural network, this is how NN knows that faeatre vector i is to be labeled as y[i], since we add the cost of rest of the lables but y[i] """
    delta3 = a3 - yk       # m X 2

    #STEP 3 : FIND DELTA FOR HIDDEN LAYER
    delta2 = (delta3 . dot(theta2) ) * c_[ones((m, 1)), sigmoidGradient(z2)]      # (m X 2) . (2 X 3) * (m X 3)
    delta2 = delta2[:, 1:]                  # (m X 2)Taking care of delta0 (correspondig to bias unit) in the hidden layer

    #STEP 4 : ACCUMULATE GRADIENT FOR EACH LAYER, FOR THIS EXAMPLE
    theta1_grad = (delta2.T ) . dot (a1) /m      # (2 X m) X (m X 3) = shape(theta1)
    theta2_grad = (delta3.T ) . dot (a2)  /m    # (2 X m) X (m X 3)   = shape(theta2)

    theta1_grad[:, 1:] += (theta1[:,1:] * lamda)/m
    theta2_grad[:, 1:] += (theta2[:,1:] * lamda)/m

    g = array([theta1_grad.T.reshape(-1).tolist() + theta2_grad.T.reshape(-1).tolist()]).T
    # print(shape(g))

    return (J, g)


def randomInitWeights(L_in, L_out):
    epsilon = sqrt(6) / (L_in + L_out)                                   # Ideally epsilon = sqrt(6) / (L_in + L_out)
    w = random.random((L_out, L_in + 1))* 2 * e - e
    return w


def debugInitWeights(fan_out, fan_in):
    w = zeros((fan_out, fan_in+1))
    w =[sin(x)/10 for x in range(1, w.size+1)]
    return asarray(w).reshape(fan_out, fan_in+1)


def computeNumericalGrad(costFunc, nn_params):
    epsilon = 1e-4
    delta = zeros(shape(nn_params))
    numGrad = zeros(shape(nn_params))
    theta = nn_params

    for i in range (0, len(theta)):
        delta[i] = epsilon
        loss1 = costFunc(theta - delta)[0]
        loss2 = costFunc(theta + delta)[0]
        numGrad[i] = (loss2 - loss1) / (2*epsilon)
        delta[i] = 0

    return numGrad


def gradChecking(lamda = None):
    if lamda is None:
        lamda = 0
    input_layer_size = 3
    hidden_layer_size = 5
    num_labels = 3
    m = 5
    X  = debugInitWeights(m, input_layer_size - 1)
    y  = asarray([1 + (i % num_labels) for i in range(0, m)]).reshape(m, 1)
    theta1 = debugInitWeights(hidden_layer_size, input_layer_size)
    theta2 = debugInitWeights(num_labels, hidden_layer_size)

    nn_params 	= array([theta1.T.reshape(-1).tolist() + theta2.T.reshape(-1).tolist()]).T
    cost_func_handle = lambda  param : NeuralNetCostFunc(param, input_layer_size, hidden_layer_size, num_labels, X, y, lamda)
    cost, grad = cost_func_handle(nn_params)
    numGrad = computeNumericalGrad(cost_func_handle, nn_params)

    print("compare the grads here :")
    print("     grad       numericGrad")
    compare_grad = c_[grad, numGrad]
    print(compare_grad)
    print('If your backpropagation implementation is correct, then the relative difference will be small (less than 1e-9).')
    diff = numGrad - grad
    sum =  numGrad + grad
    rel_diff = sqrt(diff.T . dot(diff) ) / sqrt(sum.T . dot(sum) )
    print(rel_diff)


def getTheta(nn_params, input_layer_size, hidden_layer_size, num_labels,  X, y, lamda, X_bias, yk):
    """USE THIS FUNC HANDLE"""
    # cost_func_handle = lambda  param : costFunc(param, input_layer_size, hidden_layer_size, num_labels, X, y, lamda)
    # theta1, theta2 = unrollParams(nn_params, input_layer_size, hidden_layer_size, num_labels)
    # unraveled 	= r_[theta1.T.flatten(), theta2.T.flatten()]
    result = scipy.optimize.fmin_cg( costFunc, fprime=gradFunc, x0=nn_params, \
									args=(input_layer_size, hidden_layer_size, num_labels, X, y, lamda, X_bias, yk), \
									maxiter=150, disp=True, full_output=True )
    theta1, theta2 = unrollParams( result[0], input_layer_size, hidden_layer_size, num_labels)
    return theta1, theta2


def predict(X, theta1, theta2):
    m = shape(X)[0]
    h1 = sigmoid(c_[ones((m, 1)), X] . dot(theta1.T))
    h2 = sigmoid(c_[ones((m, 1)), h1] . dot(theta2.T))
    # print('h2 is : ')
    # print(h2)
    """NOTE THAT LABEL i IS STORED AT INDEX i-1 """
    prediction = argmax(h2, 1)
    return prediction


def accuracy(y, prediction):
    # ans = c_[y, prediction]
    # c = 0
    # for i in range(0, y.size):
        # print(ans[i, :])
        # if y[i] == prediction[i] :
        #     c += 1
    prediction = prediction.reshape(y.size, 1)
    print('accuracy of the neural network : ')
    print(100 * sum(y == prediction)/y.size)


def part2_6(nn_params_optimal, input_layer_size, hidden_layer_size, num_labels, X, y, lamda, X_bias, yk):

    theta1, theta2 = getTheta(nn_params_optimal, input_layer_size, hidden_layer_size, num_labels, X, y, lamda, X_bias, yk)
    prediction = predict(X, theta1, theta2)
    accuracy(y, prediction)


def main():

    x = [[0, 0], [0, 1], [1, 0], [1, 1]]
    Y = [0, 1, 1, 0]

    """IMPORTANT        :       WE MUST HAVE ENOUGH EXAMPLES IN ORDER TO GENERALIZE THE BEHAVIOUR, ALTHOUGH I STILL NEED TO THINK MORE ON THIS ISSUE"""
    X = asarray([j for i in range(0, 100) for j in x]).reshape(400, 2)
    y =  asarray([j for i in range(0, 100) for j in Y]).reshape(400, 1)
    input_layer_size, hidden_layer_size, num_labels = 2, 2, 2

    theta1_rand, theta2_rand = randomInitWeights(2, 2), randomInitWeights(2, 2)
    nn_params_rand 	= array([theta1_rand.T.reshape(-1).tolist() + theta2_rand.T.reshape(-1).tolist()]).T

    m, n = shape(X)
    X_bias = c_[ones((m, 1)), X]
    yk = zeros((m, num_labels))
    for i in range(0, m):
            yk[i, y[i]] = 1
    lamda = 1.0

    part2_6(nn_params_rand, input_layer_size, hidden_layer_size, num_labels, X, y, lamda, X_bias, yk)

if __name__ == '__main__':
    main()