# STRUCTURE AND OPERATION OF FULL-LINK NEURAL NETWORKS

"""
In a full-connected neural network, each neuron of the current layer is
connected to a neuron of the next layer, and direct propagation means
that the input signal propagates from input to output
"""

import numpy as np

def act(x):
    """
    activation function
    """

    return 0 if x < 0.5 else 1

def main(house, rock, attr):
    """
    neural network that looks to see if sympathy is formed or not
    """

    x = np.array([house, rock, attr]) # input signal vector
    w11 = [0.3, 0.3, 0] # weights to activate 1 hidden layer
    w12 = [0.4, -0.5, 1] # weights to activate 2 hidden layer
    weight1 = np.array([w11, w12]) # matrix 2x3
    weight2 = np.array([-1,1]) # vector 1x2

    # matrix multiplication: mxn * nxp = mxp

    sum_hidden = np.dot(weight1, x)
    print('Значение сумм на нейронах скрытого слоя: ' + str(sum_hidden))

    out_hidden = np.array([act(x) for x in sum_hidden])
    print('Значение на выходах нейронов скрытого слоя: ' + str(out_hidden))

    sum_end = np.dot(weight2, out_hidden)
    print('Конечная сумма: ' + str(sum_end))
    
    y =  act(sum_end)
    print('Выходное значение нейронной сети: '+ str(y))

    return y


if __name__ == '__main__':

    guy = [1 , 0 , 1]

    res = main(*guy)
    if res == 1:
        print('Симпатия')
    else:
        print('Отсутствие симпатии')























