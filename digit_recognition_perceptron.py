import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# all libraries for nn
import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import mnist
from tensorflow import keras
from keras.layers import Dense, Flatten

def main():
    (x_train, y_train), (x_test, y_test) = mnist.load_data() # 60000 example for train sampling, and 10000 for test

    # standardization of input data
    x_train = x_train / 255
    x_test = x_test / 255

    # converting output values into vectors by category
    y_train_cat = keras.utils.to_categorical(y_train, 10)
    y_test_cat = keras.utils.to_categorical(y_test, 10)

    # displaying the first 25 images from the training sample
    plt.figure(figsize=(10,5))
    for i in range(25):
        plt.subplot(5,5,i+1)
        plt.xticks([])
        plt.yticks([])
        plt.imshow(x_train[i], cmap=plt.cm.binary)

    plt.show()

    model = keras.Sequential([
        Flatten(input_shape=(28, 28, 1)),
        Dense(128, activation='relu'),
        Dense(10, activation='softmax')
    ])

    print(model.summary())      # output the NN structure to the console

    model.compile(optimizer='adam',
                 loss='categorical_crossentropy',
                 metrics=['accuracy'])


    model.fit(x_train, y_train_cat, batch_size=32, epochs=5, validation_split=0.2)

    model.evaluate(x_test, y_test_cat)

    n = 1
    x = np.expand_dims(x_test[n], axis=0)
    res = model.predict(x)
    print( res )
    print( np.argmax(res) )

    plt.imshow(x_test[n], cmap=plt.cm.binary)
    plt.show()


if __name__ == '__main__':
    main()