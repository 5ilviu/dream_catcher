from random import randint

import datetime
import numpy as np
from matplotlib import animation
from numpy.random.mtrand import permutation

import alt_rbm
import import_data
import matplotlib.pyplot as plt


if __name__ == '__main__':
    global visual
    data = import_data.get_img()
    # data = np.resize(data, (50000, 784))
    # perm = permutation(50000)
    # data = data[perm]
    # data = np.array([data, data, data, data, data])
    rbm = alt_rbm.RBM(24300, 200, learning_rate=0.1)
    rbm.train(data[0:4], 2000)
    f = open("matrix{}".format(datetime.datetime.now()), "wb")
    np.save(f, rbm.weights)
    # good ones are 6 data[6] >
    initial = np.random.rand(201)
    # im = plt.imshow(np.random.rand(28, 28), cmap=plt.get_cmap('gray'), animated=True)
    visual = rbm.daydream(2, initial)

    # ani = animation.FuncAnimation(fig, updatefig, interval=2000, blit=True)
    # plt.show()



