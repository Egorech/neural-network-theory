# KERAS - INSTALLATION AND FIRST INTRODUCTION

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from keras.layers import Dense

def main():
    """
    neural network, learning with a teacher,
    conversion from fahrenheit to celsius
    """



    c = np.array([-40, -10, 0, 8, 15, 22, 38]) # celsius
    f = np.array([-40, 14, 32, 46, 59, 72, 100]) # farenheit

    model = keras.Sequential() # creating a multilayer neural network model
    model.add(Dense(units = 1, input_shape = (1, ), activation = 'linear')) # 1 neuron, 1 input, linear activation function f(x)=x
    model.compile(loss = 'mean_squared_error', optimizer = keras.optimizers.Adam(0.1))

    history = model.fit(c, f, epochs = 500, verbose = 0)

    plt.plot(history.history['loss'])
    plt.grid(True)
    plt.show()

    print(model.predict([100])) # neural network operation at a value of 100
    print(model.get_weights()) # the output of neuronka scales



if __name__ == '__main__':
    main()


