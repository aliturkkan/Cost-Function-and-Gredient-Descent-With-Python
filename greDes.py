from numpy import size

def gredientDescent(theta, X, y, alpha, iterator):

    # the length of the y component of the data set
    m = size(y)

    '''
        Compute Gredien Descent
        J := J - alpha/m(sum(h(x)-y)X
    '''
    for i in range(iterator):
        guesses = X.dot(theta).flatten()

        err1 = (guesses - y) * X[:, 0]
        err2 = (guesses - y) * X[:, 1]

        theta[0][0] = theta[0][0] - alpha * (1.0 / m) * err1.sum()
        theta[1][0] = theta[1][0] - alpha * (1.0 / m) * err2.sum()

    return theta

