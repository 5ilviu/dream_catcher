from random import randint

import datetime
import numpy as np
from matplotlib import animation
from numpy.random.mtrand import permutation

import alt_rbm
import import_data
import matplotlib.pyplot as plt


fig = plt.figure()
visual = None
im = plt.imshow(np.zeros((28, 28)), cmap=plt.get_cmap('gray'), animated=True)
i = 0


def get_index():
    global i
    i += 1
    if i >= len(visual):
        i = 0
        print 'restart'
    print(i)
    return i


def updatefig(*args):
    global visual
    mat = np.reshape(visual[get_index()], (3, 90, 90))
    im.set_array(mat)
    return im,


if __name__ == '__main__':
    global visual
    data = import_data.get_img()
    data1 = import_data.get_img('./res/cs_4_1_8.jpg')
    # data = np.resize(data, (50000, 784))
    # perm = permutation(50000)
    # data = data[perm]
    data = np.array([data, data1, data, data1])
    rbm = alt_rbm.RBM(24300, 200, learning_rate=0.1)
    rbm.train(data[0:4], 10)
    f = open("matrix{}".format(datetime.datetime.now()), "wb")
    np.save(f, rbm.weights)
    # good ones are 6 data[6] >
    initial = np.random.rand(201)
    im = plt.imshow(np.random.rand(28, 28), cmap=plt.get_cmap('gray'), animated=True)
    visual = rbm.daydream(2, initial)

    ani = animation.FuncAnimation(fig, updatefig, interval=2000, blit=True)
    plt.show()



