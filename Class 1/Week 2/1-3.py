import numpy as np
import matplotlib.pyplot as plt
import pylab
from testCases import *
import sklearn
import sklearn.datasets
import sklearn.linear_model
from planar_utils import plot_decision_boundary, sigmoid, load_extra_datasets, load_planar_dataset

np.random.seed(1) # set a seed so that the results are consistent

X, Y = load_planar_dataset()

plt.scatter(X[0, :], X[1, :], c=Y[0, :], s=40, cmap=plt.cm.Spectral)
plt.show()

shape_X = X.shape
shape_Y = Y.shape
m = X.shape[1]

print('The shape of X is: ' + str(shape_X))
print('The shape of Y is: ' + str(shape_Y))
print('I have m = %d training examples!' %(m))