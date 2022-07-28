# PERSEPTRON - IMAGE CLASSIFICATION CAPABILITIES

"""
In machine learning, the perceptron is an algorithm for supervised learning
of binary classifiers. A binary classifier is a function which can
decide whether or not an input, represented by a vector of numbers,
belongs to some specific class.
"""

import numpy as np
import matplotlib.pyplot as plt

N = 5

def main():
    """
    binary classification with a dividing line
    """

    # first class, points above the line
    x1 = np.random.random(N)
    x2 = x1 + [np.random.randint(10)/10 for i in range(N)]
    C1 = [x1,x2]

    # second class, points below the line
    x1 = np.random.random(N)
    x2 = x1 - [np.random.randint(10)/10 for i in range(N)] - 0.1
    C2 = [x1, x2]

    f = [0, 1]

    w = np.array([-0.3, 0.3])
    for i in range(N):
        x = np.array([C2[0][i], C2[1][i]])
        y = np.dot(w,x)
        if y >= 0:
            print('Класс C1')
        else:
            print('Класс C2')

        plt.scatter(C1[0][:], C1[1][:], s=10, c='red')
        plt.scatter(C2[0][:] , C2[1][:] , s = 10 , c = 'blue')
        plt.plot(f)
        plt.grid(True)
        plt.show()



if __name__ == '__main__':
    main()























































