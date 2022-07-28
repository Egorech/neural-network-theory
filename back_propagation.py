#  BACK PROPAGATION - BACK PROPAGATION LEARNING ALGORITHM

"""
The error back propagation method is a gradient calculation method
that is used when updating the weights of a multilayer perceptron.
"""

# training sample
epoch = [(-1, -1, -1, -1),
         (-1, -1, 1, 1),
         (-1, 1, -1, -1),
         (-1, 1, 1, 1),
         (1, -1, -1, -1),
         (1, -1, 1, 1),
         (1, 1, -1, -1),
         (1, 1, 1, -1)]

import numpy as np

def f(x):

    # hyperbolic tangent
    return 2/(1 + np.exp(-x)) - 1

def df(x):

    # derivative of the hyperbolic tangent
    return 0.5*(1 + x)*(1 - x)

W1 = np.array([[-0.2, 0.3, -0.4], [0.1, -0.3, -0.4]])
W2 = np.array([0.2, 0.3])

def go_forward(inp):
    sum = np.dot(W1, inp)
    out = np.array([f(x) for x in sum])

    sum = np.dot(W2, out)
    y = f(sum)
    return (y, out)

def train(epoch):
    global W2, W1
    lmd = 0.01          # learning step
    N = 10000           # number of iterations in training
    count = len(epoch)
    for k in range(N):
        x = epoch[np.random.randint(0, count)]
        y, out = go_forward(x[0:3])             # direct walkthrough of the NS and calculation of neuron output values
        e = y - x[-1]                           # error
        delta = e*df(y)                         # local gradient
        W2[0] = W2[0] - lmd * delta * out[0]    # first link weight adjustment
        W2[1] = W2[1] - lmd * delta * out[1]    # adjusting the weight of the second link

        delta2 = W2*delta*df(out)               # vector of 2 values of local gradients

        # correcting the links of the first layer
        W1[0, :] = W1[0, :] - np.array(x[0:3]) * delta2[0] * lmd
        W1[1, :] = W1[1, :] - np.array(x[0:3]) * delta2[1] * lmd


if __name__ == '__main__':

    train(epoch)

    for x in epoch:
        y, out = go_forward(x[0:3])
        print(f"Выходное значение НС: {y} => {x[-1]}")