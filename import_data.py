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
            data.append(img)
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

# from sklearn.metrics import confusion_matrix
# y_true = [2, 0, 2, 2, 0, 1]
# y_pred = [0, 0, 2, 2, 0, 2]
# print confusion_matrix(y_true, y_pred)
