from numpy import size, dot

def costFunction(theta, X, y):

    #the length of the y component of the data set
    #it used to find the total error
    m = size(y)

    '''
    Compute Cost Function
    J = 1/2m(sum(h(x)-y)^2
    '''
    guess = dot(X,theta).flatten()
    error = (guess - y) ** 2
    J = (1.0/(2.0*m)) * sum(error)

    return J