
from scipy import misc
import matplotlib.pyplot as plt


def get_img(image='./res/cs_1_1.jpg'):
    f = misc.imread(image).astype(float)
    return ((f - f.min())/(f.max() - f.min())).reshape(24300)


def show_img(f):
    plt.imshow(f)
    plt.show()


if __name__ == '__main__':
    print show_img(get_img('./res/cs_2_1_1.jpg').reshape(90, 90, 3))

# from sklearn.metrics import confusion_matrix
# y_true = [2, 0, 2, 2, 0, 1]
# y_pred = [0, 0, 2, 2, 0, 2]
# print confusion_matrix(y_true, y_pred)
