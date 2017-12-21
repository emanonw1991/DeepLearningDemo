import numpy as np
#1.1
def sigmoid(x):
    s = 1 / (1 + np.exp(-x))
    return s
#1.2
def sigmoid_derivative(x):
    s = sigmoid(x)
    d = s * (1 - s)
    return d

x = np.array([[1, 2, 3],
             [2, 3, 4]])
print(sigmoid_derivative(x))

#1.3
def image2vector(image):
    length = image.shape[0]
    height = image.shape[1]
    dim = length * height * 3
    image = image.reshape((dim, 1))
    return image

image = np.array ([[[1, 2, 3], [2, 3, 4], [3, 4, 5]],
          [[4, 5, 6], [5, 6, 7], [6, 7, 8]],
          [[7, 8, 9], [8, 9, 10], [9, 10, 11]]])
print(image2vector(image))

#1.4
def normalizeRows(x):
    norm = np.linalg.norm(x, axis=1, keepdims=True)
    print("norm's shape is: {0}".format(norm.shape))
    x_normalized = x / norm
    print("x_normalized's shape is: {0}".format(x_normalized.shape))
    return x_normalized

x = np.array([[1, 2, 3],
              [2, 3, 4]])
x_normalized = normalizeRows(x)
print(x_normalized)
print("x's shape is: {0}".format(x.shape))
print("normalized x's shape is: {0}".format(x_normalized.shape))

#1.5
def softmax(x):
    exp = np.exp(x)
    print("exp's shape is: {0}".format(exp.shape))
    x_sum = np.sum(exp, axis=1, keepdims=True)
    print("x_sum's shape is: {0}".format(x_sum.shape))
    s = exp / x_sum
    print("s's shape is: {0}".format(s.shape))
    return s

x = np.array([[1, 2],
              [3, 4]])
s = softmax(x)
print(s)

#2.1
def L1(yhat, y):
    loss = np.sum(np.abs(y - yhat))
    return loss

yhat = np.array([.9, 0.2, 0.1, .4, .9])
y = np.array([1, 0, 0, 1, 1])
print("L1 = " + str(L1(yhat, y)))

def L2(yhat, y):
    loss = np.sum(np.power((yhat - y), 2))
    return loss

print("L2 = " + str(L2(yhat, y)))
