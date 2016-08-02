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
im = plt.imshow(np.zeros((90, 90)), cmap=plt.get_cmap('gray'), animated=True)
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
    mat = np.reshape(visual[get_index()], (90, 90, 3))
    im.set_array(mat)
    return im,


if __name__ == '__main__':
    global visual
    dataset = import_data.get_dataset()
    data = dataset[0]
    label = dataset[1]
    # data = np.resize(data, (50000, 784))
    # perm = permutation(50000)
    # data = data[perm]
    rbm = alt_rbm.RBM(24300, 400, learning_rate=0.1)
    perm = permutation(len(data))
    data = data[perm]
    label = label[perm]
    rbm.train(data, 10)
    f = open("matrix{}".format(datetime.datetime.now()), "wb")
    np.save(f, rbm.weights)
    # good ones are 6 data[6] >
    initial = np.random.rand(401)
    im = plt.imshow(np.random.rand(90, 90, 3), cmap=plt.get_cmap('gray'), animated=True)
    visual = rbm.daydream(100, initial)

    ani = animation.FuncAnimation(fig, updatefig, interval=500, blit=True)
    plt.show()



