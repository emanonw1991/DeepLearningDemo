import numpy as np
import h5py
import matplotlib.pyplot as plt
import scipy
import pylab
from PIL import Image
from scipy import ndimage
from lr_utils import load_dataset

train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = load_dataset()
index = 23
plt.imshow(train_set_x_orig[index])
pylab.show()
print("y = " + str(train_set_y[:, index]) + ", it's a '" + classes[np.squeeze(train_set_y[:, index])].decode("utf-8") + "' picture.")

m_train = train_set_x_orig.shape[0]
m_test = test_set_x_orig.shape[0]
num_px = train_set_x_orig.shape[1]
print('Number of training examples: m_train = ' + str(m_train))
print('Number of testing examples: m_test = ' + str(m_test))
print("Height/Width of each image: num_px = " + str(num_px))
print('Each image is of size: ({0}, {1}, 3)'.format(num_px, num_px))
print('train_set_x shape: ' + str(train_set_x_orig.shape))
print('train_set_y shape: ' + str(train_set_y.shape))
print('test_set_x shape: ' + str(test_set_x_orig.shape))
print('test_set_y shape: ' + str(test_set_y.shape))

train_set_x_flatten = train_set_x_orig.reshape(m_train, -1).T
test_set_x_flatten = test_set_x_orig.reshape(m_test, -1).T
print('train_set_x_flatten shape: ' + str(train_set_x_flatten.shape))
print('test_set_x_flatten shape: ' + str(test_set_x_flatten.shape))
print('train_set_y shape: ' + str(train_set_y.shape))
print('test_set_y shape: ' + str(test_set_y.shape))
print('sanity check after reshaping: ' + str(train_set_x_flatten[0:5, 0]))

train_set_x = train_set_x_flatten / 255;
test_set_x = test_set_x_flatten / 255;