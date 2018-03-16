from matplotlib.pyplot import title, ylabel, xlabel, scatter, legend, show, plot, contour
from matplotlib.style import use
from numpy import size, ones, zeros, linspace, logspace, transpose, loadtxt
import costFunc
import greDes

#load dataset
veri = loadtxt('dataset.txt',delimiter=',')

#dataset's first col = x component
#dataset's second col = y component
x = veri[:,0]
y = veri[:,1]

#plotting the data
use('ggplot')
title("Data Set")
ylabel("X")
xlabel("y")
scatter(x,y,color="b")
legend


#the length of the y component of the data set
m = size(y)

#adding the ones col of x component
onesX = ones(shape=(m,2))
onesX[:, 1] = x

#required components for compute cost function and gredient descent
theta = zeros(shape=(2,1))
alpha = 0.01
iteration = 1500

#call cost function and print the result
J = costFunc.costFunction(theta, onesX, y)
print("Cost Function:\t{0}".format(round(J,4)))

#call gredient descent and print the result
theta = greDes.gredientDescent(theta, onesX, y, alpha, iteration)
print("Gredient Descent:\n{0}".format(theta))

#optimum straight line
result = onesX.dot(theta).flatten()
plot(veri[:, 0], result)
show()

#Values between -10 and 10 equidistant
theta0 = linspace(-10, 10, 100)
#Values between -1 and 4 equidistant
theta1 = linspace(-1, 4, 100)

#create a empty matris
J_val = zeros(shape=(len(theta0), len(theta1)))

#Fill J_val matris
for i, factor in enumerate(theta0):
    for j, factor2 in enumerate(theta1):
        thetaT = zeros(shape=(2, 1))
        thetaT[0][0] = factor
        thetaT[1][0] = factor2
        J_val[i, j] = costFunc.costFunction(thetaT, onesX, y)


J_valT = transpose(J_val)

#plot last calculation
contour(theta0, theta1, J_valT, logspace(-2, 3, 20), colors='g')
xlabel('theta 0')
ylabel('theta 1')
scatter(theta[0][0], theta[1][0])
show()
