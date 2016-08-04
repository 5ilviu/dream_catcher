from os import listdir
from os.path import join
import numpy as np
from scipy import misc
import matplotlib.pyplot as plt

data_path = './res/{}/{}/'


def get_img(image='./res/1/cs_1_1.jpg'):
    f = misc.imread(image).astype(dtype=float)
    return ((f - f.min())/(f.max() - f.min())).reshape(24300)


def get_dataset(nature='train'):
    data = []
    label = []
    i = 0
    for clazz in range(0, 10):
        path = data_path.format(nature, clazz)
        for f in listdir(path):
            # print join(path, f)
            img = get_img(join(path, f))
            data.append(np.array(img))
            # data.append(np.array(np.roll(img, 90)))
            # data.append(np.array(np.roll(img, 9000)))
            # data.append(np.array(np.roll(img, -90)))
            data.append(img + (np.random.rand(len(img)))*.1)
            data.append(img + (np.random.rand(len(img))-.5)*.1)
            data.append(img + (np.random.rand(len(img))-.5)*.2)
            data.append(img + (np.random.rand(len(img))-.5)*.3)
            data.append(img + (np.random.rand(len(img))-.5)*.4)
            data.append(img + (np.random.rand(len(img)))*.2)

            label.append(clazz)
            label.append(clazz)
            label.append(clazz)
            label.append(clazz)
            label.append(clazz)
            label.append(clazz)
            label.append(clazz)
            i += 1
    print 'read {} {} images'.format(nature, i)
    return [np.array(data), np.array(label)]




def show_img(f):
    plt.imshow(f.reshape(90, 90, 3))
    plt.show()


if __name__ == '__main__':
    get_dataset()
    # print show_img(get_img('./res/cs_2_1_1.jpg').reshape(90, 90, 3))
